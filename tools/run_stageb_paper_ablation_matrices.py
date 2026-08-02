#!/usr/bin/env python3
"""Launch sealed CVPR Table-B and Table-D Stage-B training matrices.

The interface mirrors ``run_stageb_token_ablation_matrix.py`` while adding the
contracts that are unique to TN-source and score-ownership ablations. ``run``
is deliberately fail-closed: every selected run root must be absent, inputs
are hashed before training, and completion requires a weights-only checkpoint
and scorer-initialization audit.

S3 is one logical run with three physical phases: a throwaway one-update
isolation probe, rank training for half of the declared update budget, and
confidence training for the other half. The confidence phase loads the rank
checkpoint through ``--pretrain_model_path`` and never resumes optimizer state.

Runtime environment overrides:

* ``PIVOT_PYTHON``, ``PIVOT_STAGE_A_INIT``, ``PIVOT_SCORER_WARMSTART``
* ``PIVOT_BATCH_SIZE`` and ``PIVOT_MAX_TRAIN_ITERS`` (total paper budget)
* ``PIVOT_ITER_CHECKPOINT_INTERVAL``, ``PIVOT_NUM_WORKERS``
* ``PIVOT_PREFETCH_FACTOR``, ``PIVOT_OMP_NUM_THREADS``, ``PIVOT_MIN_NOFILE``
* ``PIVOT_CUDA_VISIBLE_DEVICES``, ``PIVOT_DATA_ROOT``
* ``PIVOT_TN_OUTPUT_ROOT`` and ``PIVOT_SCORE_OUTPUT_ROOT``
* ``PIVOT_ORCHESTRATION_ROOT`` (``detach`` control artifacts)
* ``PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL`` (S0-S2; must be positive)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_stageb_token_ablation_matrix as token_launcher


DEFAULT_STAGE_A_INIT = token_launcher.DEFAULT_STAGE_A_INIT
DEFAULT_SCORER_WARMSTART = token_launcher.DEFAULT_SCORER_WARMSTART
DEFAULT_PYTHON = token_launcher.DEFAULT_PYTHON
DEFAULT_STAGE_A_SHA256 = token_launcher.DEFAULT_STAGE_A_SHA256
DEFAULT_SCORER_SHA256 = token_launcher.DEFAULT_SCORER_SHA256
DEFAULT_TN_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/tn_data_ablation"
DEFAULT_SCORE_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/score_decoupling"
DEFAULT_ORCHESTRATION_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/orchestration/paper_ablation_matrices"
)
DEFAULT_TABLE_D_DATASET = REPO_ROOT / "config/datasets_stageb_v21_single_edit_train.json"
TABLE_B_AUDIT = (
    REPO_ROOT
    / "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
)
TABLE_B_AUDIT_SHA256 = (
    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
)
MATCHED_TABLE_B_AUDIT = (
    REPO_ROOT / "data/ablations/stageb_tn_c2_parent_matched_20260717/audit.json"
)
MATCHED_TABLE_B_AUDIT_SHA256 = (
    "ca1c9c581fd78f1fe026397cc127d9b7448c60227b31c5e83148c91e9c61861e"
)
SEEDS = (17, 42, 73)


@dataclass(frozen=True)
class MatrixRow:
    row_id: str
    table: str
    config: str
    dataset: str
    tn_scope: str | None = None
    score_ownership: str | None = None
    objective_fidelity: str | None = None


ROWS: tuple[MatrixRow, ...] = (
    MatrixRow(
        "D0",
        "B",
        "config/ablations/cfg_stageb_v23_table_b_d0_no_tn.py",
        "config/datasets_stageb_table_b_d0_no_tn.json",
        tn_scope="none",
    ),
    MatrixRow(
        "D1",
        "B",
        "config/ablations/cfg_stageb_v23_table_b_d1_unverified_allneg.py",
        "config/datasets_stageb_table_b_d1_unverified_allneg.json",
        tn_scope="unverified_all_negative",
    ),
    MatrixRow(
        "D2",
        "B",
        "config/ablations/cfg_stageb_v23_table_b_d2_traceable_edits.py",
        "config/datasets_stageb_table_b_d2_traceable_edits.json",
        tn_scope="traceable_counterfactual_edit",
    ),
    MatrixRow(
        "D3",
        "B",
        "config/ablations/cfg_stageb_v23_table_b_d3_proposal_covered.py",
        "config/datasets_stageb_table_b_d3_proposal_covered.json",
        tn_scope="proposal_covered_verified",
    ),
    MatrixRow(
        "D2m",
        "B",
        "config/ablations/cfg_stageb_v24_table_b_d2m_matched.py",
        "config/datasets_stageb_table_b_d2m_matched_traceable.json",
        tn_scope="traceable_counterfactual_edit",
    ),
    MatrixRow(
        "D3m",
        "B",
        "config/ablations/cfg_stageb_v24_table_b_d3m_matched.py",
        "config/datasets_stageb_table_b_d3m_matched_proposal_covered.json",
        tn_scope="proposal_covered_verified",
    ),
    MatrixRow(
        "S0",
        "D",
        "config/ablations/cfg_stageb_v22_s0_shared_score.py",
        "config/datasets_stageb_v21_single_edit_train.json",
        score_ownership="shared_score",
        objective_fidelity="common_objective_ownership_ablation",
    ),
    MatrixRow(
        "S1",
        "D",
        "config/ablations/cfg_stageb_v22_s1_shared_trunk_two_heads.py",
        "config/datasets_stageb_v21_single_edit_train.json",
        score_ownership="shared_trunk_two_heads",
        objective_fidelity="common_objective_ownership_ablation",
    ),
    MatrixRow(
        "S2",
        "D",
        "config/ablations/cfg_stageb_v22_s2_independent_joint.py",
        "config/datasets_stageb_v21_single_edit_train.json",
        score_ownership="independent_decoders_joint",
        objective_fidelity="common_objective_ownership_ablation",
    ),
    MatrixRow(
        "S3",
        "D",
        "config/ablations/cfg_stageb_v22_s3_rank_phase.py",
        "config/datasets_stageb_v21_single_edit_train.json",
        score_ownership="independent_decoders_two_phase",
        objective_fidelity=(
            "common_objective_ownership_ablation_split_schedule"
        ),
    ),
    MatrixRow(
        "S2F",
        "D",
        "config/ablations/cfg_stageb_v22_s2_independent_joint_full.py",
        "config/datasets_stageb_v21_single_edit_train.json",
        score_ownership="independent_decoders_joint",
        objective_fidelity="full_v19_base_plus_gate_objective",
    ),
)
ROW_BY_ID = {row.row_id: row for row in ROWS}
ROW_BY_ID_CASEFOLD = {row.row_id.casefold(): row for row in ROWS}


def _table_b_audit_contract(row: MatrixRow) -> tuple[Path, str, str]:
    if row.row_id in {"D2m", "D3m"}:
        return (
            MATCHED_TABLE_B_AUDIT,
            MATCHED_TABLE_B_AUDIT_SHA256,
            "disabled_uniformly_D2m_D3m",
        )
    return TABLE_B_AUDIT, TABLE_B_AUDIT_SHA256, "disabled_uniformly_D1_D3"


@dataclass(frozen=True)
class Phase:
    phase_id: str
    config: str
    updates: int
    diagnostic_interval: int
    scorer_warmstart: bool
    contributes_to_budget: bool
    pretrain_source: str


@dataclass(frozen=True)
class Runtime:
    python: Path
    stage_a_init: Path
    scorer_warmstart: Path
    tn_output_root: Path
    score_output_root: Path
    data_root: Path
    batch_size: int
    total_train_iters: int
    iter_checkpoint_interval: int
    num_workers: int
    prefetch_factor: int
    omp_num_threads: int
    min_nofile: int
    cuda_visible_devices: str
    mp_sharing_strategy: str
    gradient_diagnostic_interval: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve(strict=True)
    else:
        found = shutil.which(value)
        if found is None:
            raise FileNotFoundError(f"runtime executable not found: {value}")
        resolved = Path(found).resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError(f"runtime executable is not executable: {resolved}")
    return resolved


def runtime_from_environment() -> Runtime:
    total = _env_int("PIVOT_MAX_TRAIN_ITERS", 1000, minimum=2)
    if total % 2:
        raise ValueError(
            "PIVOT_MAX_TRAIN_ITERS must be even so S3 rank/confidence phases "
            "receive exactly equal update budgets"
        )
    interval = _env_int(
        "PIVOT_ITER_CHECKPOINT_INTERVAL", total, minimum=1
    )
    sharing = os.environ.get("PIVOT_MP_SHARING_STRATEGY", "file_system")
    if sharing not in {"file_system", "file_descriptor", "none"}:
        raise ValueError(
            "PIVOT_MP_SHARING_STRATEGY must be file_system, file_descriptor, or none"
        )
    runtime = Runtime(
        python=_resolve_executable(
            os.environ.get("PIVOT_PYTHON", str(DEFAULT_PYTHON))
        ),
        stage_a_init=Path(
            os.environ.get("PIVOT_STAGE_A_INIT", str(DEFAULT_STAGE_A_INIT))
        ).expanduser().resolve(strict=True),
        scorer_warmstart=Path(
            os.environ.get(
                "PIVOT_SCORER_WARMSTART", str(DEFAULT_SCORER_WARMSTART)
            )
        ).expanduser().resolve(strict=True),
        tn_output_root=Path(
            os.environ.get("PIVOT_TN_OUTPUT_ROOT", str(DEFAULT_TN_OUTPUT_ROOT))
        ).expanduser().resolve(strict=False),
        score_output_root=Path(
            os.environ.get(
                "PIVOT_SCORE_OUTPUT_ROOT", str(DEFAULT_SCORE_OUTPUT_ROOT)
            )
        ).expanduser().resolve(strict=False),
        data_root=Path(
            os.environ.get(
                "PIVOT_DATA_ROOT",
                os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"),
            )
        ).expanduser().resolve(strict=False),
        batch_size=_env_int("PIVOT_BATCH_SIZE", 16, minimum=1),
        total_train_iters=total,
        iter_checkpoint_interval=interval,
        num_workers=_env_int("PIVOT_NUM_WORKERS", 8, minimum=0),
        prefetch_factor=_env_int("PIVOT_PREFETCH_FACTOR", 1, minimum=1),
        omp_num_threads=_env_int("PIVOT_OMP_NUM_THREADS", 8, minimum=1),
        min_nofile=_env_int("PIVOT_MIN_NOFILE", 65536, minimum=0),
        cuda_visible_devices=os.environ.get(
            "PIVOT_CUDA_VISIBLE_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        ),
        mp_sharing_strategy=sharing,
        gradient_diagnostic_interval=_env_int(
            "PIVOT_GRADIENT_DIAGNOSTIC_INTERVAL", 100, minimum=1
        ),
    )
    for label, path in (
        ("Stage-A initializer", runtime.stage_a_init),
        ("scorer warm-start", runtime.scorer_warmstart),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    return runtime


def _row_dataset(row: MatrixRow) -> Path:
    return (REPO_ROOT / row.dataset).resolve(strict=True)


def _row_config(row: MatrixRow) -> Path:
    return (REPO_ROOT / row.config).resolve(strict=True)


def output_directory(runtime: Runtime, row: MatrixRow, seed: int) -> Path:
    root = runtime.tn_output_root if row.table == "B" else runtime.score_output_root
    return root / row.row_id / f"seed{seed}"


def _phases(runtime: Runtime, row: MatrixRow) -> tuple[Phase, ...]:
    if row.row_id != "S3":
        diagnostic = (
            runtime.gradient_diagnostic_interval if row.table == "D" else 0
        )
        return (
            Phase(
                "joint",
                row.config,
                runtime.total_train_iters,
                diagnostic,
                True,
                True,
                "stage_a_initializer",
            ),
        )
    half = runtime.total_train_iters // 2
    return (
        Phase(
            "isolation_probe",
            "config/ablations/cfg_stageb_v22_s3_isolation_probe.py",
            1,
            1,
            True,
            False,
            "stage_a_initializer",
        ),
        Phase(
            "rank",
            "config/ablations/cfg_stageb_v22_s3_rank_phase.py",
            half,
            0,
            True,
            True,
            "stage_a_initializer",
        ),
        Phase(
            "confidence",
            "config/ablations/cfg_stageb_v22_s3_confidence_phase.py",
            half,
            0,
            False,
            True,
            "rank_phase_checkpoint",
        ),
    )


def _phase_output(run_root: Path, row: MatrixRow, phase: Phase) -> Path:
    return run_root if row.row_id != "S3" else run_root / phase.phase_id


def _expand_dataset_path(raw: str, *, dataset: Path, runtime: Runtime) -> Path:
    expanded = os.path.expandvars(raw.replace("${DATA_ROOT}", str(runtime.data_root)))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = dataset.parent / path
    legacy_root = Path("/home/user/PIVOT")
    try:
        relative = path.relative_to(legacy_root)
    except ValueError:
        return path.resolve(strict=False)
    return (REPO_ROOT / relative).resolve(strict=False)


def _validate_common_config(config: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "stage_b_v20_acc50_aligned_hard_negatives": True,
        "stage_b_v11_candidate_topk": 50,
        "stage_b_v11_positive_iou_threshold": 0.5,
        "stage_b_v11_negative_iou_threshold": 0.499,
        "stage_b_v21_token_objective": "edit_bce",
        "stage_b_v21_token_weight": 1.0,
        "stage_b_v21_token_positive_weight": 1.0,
        "stage_b_v21_token_shared_weight": 0.25,
        "stage_b_v21_token_edit_weight": 1.0,
        "stage_b_v21_token_focal_alpha": 0.25,
        "stage_b_v21_token_focal_gamma": 2.0,
        "stage_b_v11_predicate_tn_rank_weight": 1.0,
        "stage_b_v21_allow_legacy_token_diff_fallback": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"{label} violates fixed v19+Acc50+L4 contract: "
                f"{key} expected {value!r}, got {config.get(key)!r}"
            )


def _validate_table_d_comparison_block() -> dict[str, float]:
    """Reject Table D when a loss knob changes alongside score ownership."""

    configs = {
        "S0": "config/ablations/cfg_stageb_v22_s0_shared_score.py",
        "S1": "config/ablations/cfg_stageb_v22_s1_shared_trunk_two_heads.py",
        "S2": "config/ablations/cfg_stageb_v22_s2_independent_joint.py",
        "S3": "config/ablations/cfg_stageb_v22_s3_rank_phase.py",
    }
    trust_weights: dict[str, float] = {}
    for row_id, path in configs.items():
        config = runpy.run_path(str((REPO_ROOT / path).resolve(strict=True)))
        raw = config.get("stage_b_v15_tail_queue_positive_trust_weight")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(
                "Table-D ownership comparison is untrusted: "
                "stage_b_v15_tail_queue_positive_trust_weight must be a "
                f"finite number for {row_id}, got {raw!r}"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError(
                "Table-D ownership comparison is untrusted: "
                "stage_b_v15_tail_queue_positive_trust_weight must be finite "
                f"for {row_id}, got {raw!r}"
            )
        trust_weights[row_id] = value

    # S3's phase configs must not silently change the same loss knob after the
    # rank-phase comparison value has passed the cross-row check.
    for phase_name in ("confidence_phase", "isolation_probe"):
        path = REPO_ROOT / f"config/ablations/cfg_stageb_v22_s3_{phase_name}.py"
        config = runpy.run_path(str(path.resolve(strict=True)))
        raw = config.get("stage_b_v15_tail_queue_positive_trust_weight")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(
                f"Table-D S3 {phase_name} trust weight is invalid: {raw!r}"
            )
        value = float(raw)
        if not math.isfinite(value) or value != trust_weights["S3"]:
            raise RuntimeError(
                f"Table-D S3 {phase_name} trust weight {value!r} differs "
                f"from rank phase {trust_weights['S3']!r}"
            )
    if len(set(trust_weights.values())) != 1:
        raise RuntimeError(
            "Table-D ownership comparison is confounded: "
            "stage_b_v15_tail_queue_positive_trust_weight differs across rows "
            f"{trust_weights}. Equalize this loss knob for S0-S3 and test the "
            "trust term in a separate ablation before launching."
        )
    clean_weight = next(iter(trust_weights.values()))
    if clean_weight != 0.0:
        raise RuntimeError(
            "Table-D ownership comparison is untrusted: the S0-S3 clean "
            "objective requires stage_b_v15_tail_queue_positive_trust_weight=0.0, "
            f"got {clean_weight!r}"
        )
    return trust_weights


def _validate_table_d_row_objective(
    row: MatrixRow, config: Mapping[str, Any]
) -> None:
    expected = 1.0 if row.row_id == "S2F" else 0.0
    raw = config.get("stage_b_v15_tail_queue_positive_trust_weight")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RuntimeError(
            f"Table-D {row.row_id} trust weight must be a finite number, got {raw!r}"
        )
    value = float(raw)
    if not math.isfinite(value) or value != expected:
        raise RuntimeError(
            f"Table-D {row.row_id} objective contract requires "
            "stage_b_v15_tail_queue_positive_trust_weight="
            f"{expected!r}, got {raw!r}"
        )


def _validate_phase_config(row: MatrixRow, phase: Phase) -> dict[str, Any]:
    path = (REPO_ROOT / phase.config).resolve(strict=True)
    config = runpy.run_path(str(path))
    _validate_common_config(config, label=f"{row.row_id}/{phase.phase_id}")
    dependencies = token_launcher._config_dependencies(path)
    dependency_names = {dependency.name for dependency in dependencies}
    if "cfg_stageb_v19_full_text_base_plus_gate.py" not in dependency_names:
        raise RuntimeError(
            f"{row.row_id}/{phase.phase_id} does not inherit the v19 "
            "base_plus_gate implementation"
        )
    if row.table == "B":
        expected_audit, expected_audit_sha, provenance_contract = (
            _table_b_audit_contract(row)
        )
        expected_allowlist = [] if row.row_id == "D0" else [str(row.tn_scope)]
        expected = {
            "stage_b_v23_ablation_table": "B",
            "stage_b_v23_table_id": row.row_id,
            "stage_b_v23_objective_contract": (
                "v19_base_plus_gate_acc50_hardneg_v21_l4"
            ),
            "stage_b_v23_tn_token_provenance_contract": (
                provenance_contract
            ),
            "stage_b_v19_allow_scope_labeled_tn_ablation": row.row_id != "D0",
            "stage_b_v19_table_b_id": row.row_id,
            "stage_b_v19_table_b_scope_allowlist": expected_allowlist,
            "stage_b_v19_table_b_audit_sha256": expected_audit_sha,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"{row.row_id} Table-B config binding mismatch for {key}: "
                    f"expected {value!r}, got {config.get(key)!r}"
                )
        configured_audit = config.get("stage_b_v19_table_b_audit")
        if not isinstance(configured_audit, str) or (
            (REPO_ROOT / configured_audit).resolve(strict=True)
            != expected_audit.resolve(strict=True)
        ):
            raise RuntimeError(
                f"{row.row_id} Table-B config does not bind the sealed audit path"
            )
        if config.get("stage_b_v15_decoupled_confidence") is not True:
            raise RuntimeError(f"{row.row_id} must retain v19 decoupled confidence")
        if config.get("stage_b_v16_confidence_output_mode") != "base_plus_gate":
            raise RuntimeError(f"{row.row_id} must retain v19 base_plus_gate output")
    else:
        _validate_table_d_comparison_block()
        _validate_table_d_row_objective(row, config)
        weak_scope_expected = {
            "stage_b_v19_allow_scope_labeled_tn_ablation": True,
            "stage_b_v19_table_b_id": "D3",
            "stage_b_v19_table_b_scope_allowlist": [
                "proposal_covered_verified"
            ],
            "stage_b_v19_table_b_audit_sha256": TABLE_B_AUDIT_SHA256,
            "stage_b_v19_table_b_allow_single_edit_token_provenance": True,
        }
        for key, value in weak_scope_expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"{row.row_id}/{phase.phase_id} weak-scope binding mismatch "
                    f"for {key}: expected {value!r}, got {config.get(key)!r}"
                )
        expected_phase = "joint" if row.row_id != "S3" else phase.phase_id
        if phase.phase_id == "isolation_probe":
            expected_phase = "isolation_probe"
        if config.get("stage_b_v22_score_ownership") != row.score_ownership:
            raise RuntimeError(
                f"{row.row_id}/{phase.phase_id} score ownership mismatch"
            )
        if config.get("stage_b_v22_train_phase") != expected_phase:
            raise RuntimeError(
                f"{row.row_id}/{phase.phase_id} train phase mismatch: "
                f"{config.get('stage_b_v22_train_phase')!r}"
            )
        if config.get("stage_b_v22_objective_fidelity") != row.objective_fidelity and not (
            phase.phase_id == "isolation_probe"
            and config.get("stage_b_v22_objective_fidelity")
            == "common_objective_ownership_ablation_probe_only"
        ):
            raise RuntimeError(
                f"{row.row_id}/{phase.phase_id} objective-fidelity label mismatch"
            )
    return config


def _validate_dataset(
    row: MatrixRow, runtime: Runtime
) -> tuple[dict[str, Any], list[Path]]:
    dataset = _row_dataset(row)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    train = payload.get("train")
    if not isinstance(train, list) or payload.get("val") != []:
        raise RuntimeError(f"{row.row_id} dataset must contain train list and val=[]")
    expected_sources = 3 if row.row_id == "D0" else 4
    if len(train) != expected_sources:
        raise RuntimeError(
            f"{row.row_id} expected {expected_sources} train sources, got {len(train)}"
        )
    weights = [float(source.get("mix_weight", 1.0)) for source in train]
    expected_weights = [1.0, 1.0, 1.0] + ([] if row.row_id == "D0" else [3.0])
    if weights != expected_weights:
        raise RuntimeError(
            f"{row.row_id} mix weights must be {expected_weights}, got {weights}"
        )

    tn_contract: dict[str, Any] | None = None
    if row.row_id != "D0":
        tn = train[-1]
        if tn.get("source") != "sam3_tn_pair":
            raise RuntimeError(f"{row.row_id} last source must be sam3_tn_pair")
        if row.table == "B":
            expected_audit, expected_audit_sha, _ = _table_b_audit_contract(row)
            expected_tn = {
                "require_global_tn_verified": False,
                "require_single_edit_token_provenance": False,
                "paper_table_b_id": row.row_id,
                "paper_tn_scope": row.tn_scope,
            }
            for key, value in expected_tn.items():
                if tn.get(key) != value:
                    raise RuntimeError(
                        f"{row.row_id} TN dataset binding mismatch for {key}: "
                        f"expected {value!r}, got {tn.get(key)!r}"
                    )
            audit_raw = tn.get("paper_contract_audit")
            if not isinstance(audit_raw, str):
                raise RuntimeError(f"{row.row_id} lacks paper_contract_audit")
            audit_path = _expand_dataset_path(
                audit_raw, dataset=dataset, runtime=runtime
            )
            if audit_path.resolve(strict=True) != expected_audit.resolve(strict=True):
                raise RuntimeError(f"{row.row_id} points to the wrong Table-B audit")
            actual_audit_sha = token_launcher.HashCache().digest(audit_path)
            if actual_audit_sha != expected_audit_sha:
                raise RuntimeError(
                    "Table-B audit SHA-256 drifted; regenerate/review the sealed "
                    "data block before launching"
                )
            tn_contract = {
                **expected_tn,
                "paper_contract_audit": str(audit_path),
                "paper_contract_audit_sha256": actual_audit_sha,
                "tn_token_supervision_eligible": False,
            }
        else:
            expected_tn = {
                "require_global_tn_verified": False,
                "require_single_edit_token_provenance": True,
                "paper_table_b_id": "D3",
                "paper_tn_scope": "proposal_covered_verified",
            }
            for key, value in expected_tn.items():
                if tn.get(key) != value:
                    raise RuntimeError(
                        f"Table-D TN source requires {key}={value!r}, "
                        f"got {tn.get(key)!r}"
                    )
            audit_raw = tn.get("paper_contract_audit")
            if not isinstance(audit_raw, str):
                raise RuntimeError("Table-D TN source lacks paper_contract_audit")
            audit_path = _expand_dataset_path(
                audit_raw, dataset=dataset, runtime=runtime
            )
            if audit_path.resolve(strict=True) != TABLE_B_AUDIT.resolve(strict=True):
                raise RuntimeError("Table-D TN source points to the wrong scope audit")
            audit_sha = token_launcher.HashCache().digest(audit_path)
            if audit_sha != TABLE_B_AUDIT_SHA256:
                raise RuntimeError("Table-D TN scope audit SHA-256 drifted")
            tn_contract = {
                **expected_tn,
                "paper_contract_audit": str(audit_path),
                "paper_contract_audit_sha256": audit_sha,
            }

    source_files: set[Path] = set()
    source_paths: list[dict[str, Any]] = []
    for index, source in enumerate(train):
        for key in (
            "anno",
            "canonical_classes_json",
            "support_patch_tsv",
            "paper_contract_audit",
        ):
            raw = source.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            resolved = _expand_dataset_path(raw, dataset=dataset, runtime=runtime)
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"dataset source {index} field {key} is not a file: {resolved}"
                )
            source_paths.append(
                {
                    "dataset_index": index,
                    "field": key,
                    "declared": raw,
                    "resolved": str(resolved.resolve(strict=True)),
                }
            )
            source_files.add(resolved.resolve(strict=True))
    contract = {
        "train_source_count": len(train),
        "positive_source_count": 3,
        "tn_source_count": 0 if row.row_id == "D0" else 1,
        "mix_weights": weights,
        "expected_tn_draw_fraction": 0.0 if row.row_id == "D0" else 0.5,
        "tn_source": tn_contract,
        "source_paths": source_paths,
    }
    if row.table == "B":
        contract["token_fairness"] = {
            "positive_token_term_fixed": True,
            "tn_edit_token_supervision_disabled_for_all_table_b_rows": True,
            "reason": "Table B isolates paired TN/confidence data quality; Table C tests edit-token labels",
        }
    return contract, sorted(source_files, key=lambda value: str(value))


def _relevant_repository_sources() -> list[Path]:
    names = (
        "main.py",
        "engine.py",
        "datasets/patch_episode.py",
        "models/GroundingDINO/groundingdino.py",
        "models/GroundingDINO/stage_b_fixed_text_scorer.py",
        "models/GroundingDINO/stage_b_fixed_text_criterion.py",
        "util/stage_b_task_gradients.py",
        "util/stage_b_table_b_contract.py",
        "docs/paper_cvpr_ablation_protocol.md",
        "tools/run_stageb_paper_ablation_matrices.py",
        "tools/run_stageb_token_ablation_matrix.py",
    )
    return [(REPO_ROOT / name).resolve(strict=True) for name in names]


def _checkpoint_interval(runtime: Runtime, phase: Phase) -> int:
    return min(runtime.iter_checkpoint_interval, phase.updates)


def _build_command(
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    phase: Phase,
    output_dir: Path,
    *,
    rank_checkpoint: Path | None,
) -> list[str]:
    dataset = _row_dataset(row)
    config = (REPO_ROOT / phase.config).resolve(strict=True)
    if phase.pretrain_source == "rank_phase_checkpoint":
        if rank_checkpoint is None:
            rank_checkpoint = (
                output_directory(runtime, row, seed)
                / "rank/checkpoint_iter.pth"
            ).resolve(strict=False)
        pretrain = rank_checkpoint
    else:
        pretrain = runtime.stage_a_init
    command = [
        str(runtime.python),
        str((REPO_ROOT / "main.py").resolve(strict=True)),
        "-c",
        str(config),
        "--datasets",
        str(dataset),
        "--output_dir",
        str(output_dir),
        "--pretrain_model_path",
        str(pretrain),
        "--seed",
        str(seed),
        "--num_workers",
        str(runtime.num_workers),
        "--prefetch_factor",
        str(runtime.prefetch_factor),
        "--mp_sharing_strategy",
        runtime.mp_sharing_strategy,
        "--min_nofile",
        str(runtime.min_nofile),
        "--max_train_iters",
        str(phase.updates),
        "--iter_checkpoint_interval",
        str(_checkpoint_interval(runtime, phase)),
        "--note",
        f"paper_cvpr_v1_{row.row_id}_seed{seed}_{phase.phase_id}",
        "--amp",
        "--save_log",
        "--options",
        f"batch_size={runtime.batch_size}",
        (
            "stage_b_v22_gradient_diagnostic_interval="
            f"{phase.diagnostic_interval}"
        ),
        "skip_eval=True",
    ]
    if phase.scorer_warmstart:
        command.append(
            f"stage_b_v15_scorer_init_checkpoint={runtime.scorer_warmstart}"
        )
    if "--resume" in command:
        raise AssertionError("paper launcher must never resume optimizer state")
    return command


def _file_record(
    path: Path, cache: token_launcher.HashCache, *, role: str
) -> dict[str, Any]:
    return {"role": role, **token_launcher._file_record(path, cache)}


def _phase_manifest(
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    phase: Phase,
    output_dir: Path,
    cache: token_launcher.HashCache,
    *,
    rank_checkpoint: Path | None,
) -> dict[str, Any]:
    config_path = (REPO_ROOT / phase.config).resolve(strict=True)
    _validate_phase_config(row, phase)
    dataset_contract, dataset_sources = _validate_dataset(row, runtime)
    stage_a = _file_record(runtime.stage_a_init, cache, role="stage_a_initializer")
    scorer = _file_record(
        runtime.scorer_warmstart, cache, role="scorer_warmstart"
    )
    if runtime.stage_a_init == DEFAULT_STAGE_A_INIT.resolve(strict=False):
        if stage_a["sha256"] != DEFAULT_STAGE_A_SHA256:
            raise RuntimeError("default Stage-A checkpoint SHA-256 does not match protocol")
    if runtime.scorer_warmstart == DEFAULT_SCORER_WARMSTART.resolve(strict=False):
        if scorer["sha256"] != DEFAULT_SCORER_SHA256:
            raise RuntimeError("default scorer checkpoint SHA-256 does not match protocol")

    input_records = [stage_a, scorer]
    input_records.append(
        _file_record(_row_dataset(row), cache, role="dataset_manifest")
    )
    input_records.extend(
        _file_record(path, cache, role="config_dependency")
        for path in token_launcher._config_dependencies(config_path)
    )
    input_records.extend(
        _file_record(path, cache, role="dataset_source")
        for path in dataset_sources
    )
    input_records.extend(
        _file_record(path, cache, role="repository_source")
        for path in _relevant_repository_sources()
    )
    generated_dependency: dict[str, Any] | None = None
    if phase.pretrain_source == "rank_phase_checkpoint":
        expected_rank = (
            rank_checkpoint
            if rank_checkpoint is not None
            else output_directory(runtime, row, seed)
            / "rank/checkpoint_iter.pth"
        ).resolve(strict=False)
        if rank_checkpoint is not None:
            rank_record = _file_record(
                rank_checkpoint,
                cache,
                role="rank_phase_model_state_pretrain",
            )
            input_records.append(rank_record)
            generated_dependency = {
                "status": "materialized_and_hashed",
                "producer_phase": "rank",
                "path": str(expected_rank),
                "sha256": rank_record["sha256"],
                "size_bytes": rank_record["size_bytes"],
                "must_be_loaded_as": "pretrain_model_path_model_state_only",
                "optimizer_resume_forbidden": True,
            }
        else:
            generated_dependency = {
                "status": "deferred_until_rank_postflight",
                "producer_phase": "rank",
                "path": str(expected_rank),
                "must_be_loaded_as": "pretrain_model_path_model_state_only",
                "optimizer_resume_forbidden": True,
            }

    command = _build_command(
        runtime,
        row,
        seed,
        phase,
        output_dir,
        rank_checkpoint=rank_checkpoint,
    )
    return {
        "schema": "pivot.stageb.paper_ablation_phase_launch/v1",
        "status": "planned",
        "created_at_utc": _utc_now(),
        "run_id": f"{row.row_id}:{seed}",
        "row": asdict(row),
        "seed": seed,
        "phase": asdict(phase),
        "output_dir": str(output_dir.resolve(strict=False)),
        "command": command,
        "command_shell": shlex.join(command),
        "runtime": {
            "python": str(runtime.python),
            "batch_size": runtime.batch_size,
            "phase_train_iters": phase.updates,
            "total_paper_train_iters": runtime.total_train_iters,
            "iter_checkpoint_interval": _checkpoint_interval(runtime, phase),
            "num_workers": runtime.num_workers,
            "prefetch_factor": runtime.prefetch_factor,
            "omp_num_threads": runtime.omp_num_threads,
            "min_nofile": runtime.min_nofile,
            "cuda_visible_devices": runtime.cuda_visible_devices,
            "mp_sharing_strategy": runtime.mp_sharing_strategy,
            "amp": True,
        },
        "fixed_contract": {
            "architecture_ancestry": "v19_base_plus_gate_plus_acc50_hardneg",
            "candidate_topk": 50,
            "positive_iou_threshold": 0.5,
            "negative_iou_threshold": 0.499,
            "token_objective": "edit_bce",
            "predicate_pair_rank_weight": 1.0,
            "dataset": dataset_contract,
            "optimizer_resume": False,
            "gradient_diagnostic_interval": phase.diagnostic_interval,
        },
        "generated_dependency": generated_dependency,
        "inputs": {
            "records": input_records,
            "stage_a_expected_sha256": DEFAULT_STAGE_A_SHA256,
            "scorer_expected_sha256": DEFAULT_SCORER_SHA256,
        },
    }


def build_manifest(
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    cache: token_launcher.HashCache,
) -> dict[str, Any]:
    run_root = output_directory(runtime, row, seed).resolve(strict=False)
    phases = _phases(runtime, row)
    phase_manifests = [
        _phase_manifest(
            runtime,
            row,
            seed,
            phase,
            _phase_output(run_root, row, phase),
            cache,
            rank_checkpoint=None,
        )
        for phase in phases
    ]
    budget_updates = sum(
        phase.updates for phase in phases if phase.contributes_to_budget
    )
    if budget_updates != runtime.total_train_iters:
        raise AssertionError(
            f"{row.row_id} contributes {budget_updates} updates, expected "
            f"{runtime.total_train_iters}"
        )
    return {
        "schema": "pivot.stageb.paper_ablation_run_launch/v1",
        "status": "planned",
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "run_id": f"{row.row_id}:{seed}",
        "row": asdict(row),
        "seed": seed,
        "training_seeds_contract": list(SEEDS),
        "output_dir": str(run_root),
        "output_dir_fresh_at_plan": not run_root.exists(),
        "equal_budget_contract": {
            "batch_size": runtime.batch_size,
            "optimizer_updates": runtime.total_train_iters,
            "s3_probe_updates_excluded": 1 if row.row_id == "S3" else 0,
            "contributing_phase_updates": {
                phase.phase_id: phase.updates
                for phase in phases
                if phase.contributes_to_budget
            },
        },
        "phases": phase_manifests,
    }


def _iter_input_records(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield from manifest["inputs"]["records"]


def _verify_file_identities(manifest: Mapping[str, Any]) -> None:
    for record in _iter_input_records(manifest):
        path = Path(str(record["path"]))
        stat = path.stat()
        if (
            int(stat.st_size) != int(record["size_bytes"])
            or int(stat.st_mtime_ns) != int(record["mtime_ns"])
        ):
            raise RuntimeError(f"input changed after manifest hashing: {path}")


def _rehash_inputs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every unique input digest after training, bypassing hash caches."""

    expected_by_path: dict[Path, str] = {}
    roles_by_path: dict[Path, set[str]] = {}
    for record in _iter_input_records(manifest):
        path = Path(str(record["path"])).resolve(strict=True)
        expected = str(record["sha256"])
        previous = expected_by_path.setdefault(path, expected)
        if previous != expected:
            raise RuntimeError(f"manifest has conflicting hashes for input {path}")
        roles_by_path.setdefault(path, set()).add(str(record["role"]))
    fresh_cache = token_launcher.HashCache()
    records = []
    for path in sorted(expected_by_path, key=lambda value: str(value)):
        observed = fresh_cache.digest(path)
        stat = path.stat()
        expected = expected_by_path[path]
        records.append(
            {
                "path": str(path),
                "roles": sorted(roles_by_path[path]),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "observed_size_bytes": int(stat.st_size),
                "observed_mtime_ns": int(stat.st_mtime_ns),
                "passed": observed == expected,
            }
        )
    failed = [record["path"] for record in records if not record["passed"]]
    if failed:
        raise RuntimeError(f"post-run input SHA-256 drift: {failed}")
    return {
        "status": "passed",
        "algorithm": "sha256",
        "verified_at_utc": _utc_now(),
        "unique_input_count": len(records),
        "records": records,
    }


_TORCH_GPU_ENV_SCRIPT = r"""
import json
import torch

payload = {
    "torch_version": str(torch.__version__),
    "torch_cuda_build_version": str(torch.version.cuda),
    "cuda_available": bool(torch.cuda.is_available()),
    "cudnn_version": torch.backends.cudnn.version(),
    "device_count": int(torch.cuda.device_count()),
    "devices": [],
}
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    payload["devices"].append({
        "logical_index": index,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    })
print(json.dumps(payload, sort_keys=True))
"""


def _capture_gpu_environment(runtime: Runtime, output_dir: Path) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi is required for paper-run GPU telemetry")
    query = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        env=_subprocess_environment(runtime),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if query.returncode != 0 or not query.stdout.strip():
        raise RuntimeError(
            "nvidia-smi identity query failed: " + query.stderr.strip()[-2000:]
        )
    banner = subprocess.run(
        [nvidia_smi],
        cwd=REPO_ROOT,
        env=_subprocess_environment(runtime),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    cuda_match = re.search(r"CUDA Version:\s*([^|\s]+)", banner.stdout)
    devices = []
    for line in query.stdout.splitlines():
        fields = [value.strip() for value in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unparseable nvidia-smi identity row: {line!r}")
        devices.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "total_memory_mib": float(fields[4]),
            }
        )
    torch_query = subprocess.run(
        [str(runtime.python), "-c", _TORCH_GPU_ENV_SCRIPT],
        cwd=REPO_ROOT,
        env=_subprocess_environment(runtime),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if torch_query.returncode != 0:
        raise RuntimeError(
            "PyTorch CUDA environment query failed: "
            + torch_query.stderr.strip()[-2000:]
        )
    try:
        torch_environment = json.loads(torch_query.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PyTorch CUDA environment query returned invalid JSON") from exc
    if torch_environment.get("cuda_available") is not True:
        raise RuntimeError("paper training requires torch.cuda.is_available()=True")
    payload = {
        "schema": "pivot.gpu_environment/v1",
        "captured_at_utc": _utc_now(),
        "cuda_visible_devices": runtime.cuda_visible_devices,
        "nvidia_smi_path": nvidia_smi,
        "nvidia_smi_cuda_compatibility_version": (
            cuda_match.group(1) if cuda_match else None
        ),
        "nvidia_devices": devices,
        "torch_runtime": torch_environment,
        "amp_requested": True,
    }
    _write_json_atomic(output_dir / "gpu_environment.json", payload)
    return payload


def _summarize_nvidia_csv(path: Path) -> dict[str, Any]:
    by_uuid: dict[str, dict[str, Any]] = {}
    sample_count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if not line.strip() or line.startswith("timestamp,"):
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 9:
            raise RuntimeError(
                f"unparseable GPU telemetry line {line_number}: {line!r}"
            )
        timestamp, index, uuid, name, driver, total, used, free, utilization = fields
        try:
            total_mib = float(total)
            used_mib = float(used)
            free_mib = float(free)
            utilization_percent = float(utilization)
        except ValueError as exc:
            raise RuntimeError(
                f"non-numeric GPU telemetry line {line_number}: {line!r}"
            ) from exc
        sample_count += 1
        state = by_uuid.setdefault(
            uuid,
            {
                "physical_index": int(index),
                "uuid": uuid,
                "name": name,
                "driver_version": driver,
                "total_memory_mib": total_mib,
                "sample_count": 0,
                "peak_used_memory_mib": used_mib,
                "min_free_memory_mib": free_mib,
                "peak_utilization_percent": utilization_percent,
                "first_timestamp": timestamp,
                "last_timestamp": timestamp,
            },
        )
        if state["total_memory_mib"] != total_mib:
            raise RuntimeError(f"GPU total memory changed for {uuid}")
        state["sample_count"] += 1
        state["peak_used_memory_mib"] = max(
            state["peak_used_memory_mib"], used_mib
        )
        state["min_free_memory_mib"] = min(
            state["min_free_memory_mib"], free_mib
        )
        state["peak_utilization_percent"] = max(
            state["peak_utilization_percent"], utilization_percent
        )
        state["last_timestamp"] = timestamp
    if sample_count <= 0:
        raise RuntimeError("GPU telemetry contains no samples")
    return {
        "schema": "pivot.gpu_telemetry_summary/v1",
        "sample_rows": sample_count,
        "devices": sorted(by_uuid.values(), key=lambda value: value["physical_index"]),
    }


def _validate_gpu_telemetry_contract(
    gpu_environment: Mapping[str, Any],
    gpu_summary: Mapping[str, Any],
) -> None:
    """Bind the sampled devices to the identities captured before training."""

    if gpu_environment.get("schema") != "pivot.gpu_environment/v1":
        raise RuntimeError("GPU environment artifact has the wrong schema")
    if gpu_summary.get("schema") != "pivot.gpu_telemetry_summary/v1":
        raise RuntimeError("GPU telemetry summary has the wrong schema")
    if gpu_environment.get("torch_runtime", {}).get("cuda_available") is not True:
        raise RuntimeError("GPU environment audit did not confirm PyTorch CUDA")
    if int(gpu_summary.get("sample_rows", 0)) <= 0:
        raise RuntimeError("GPU telemetry summary contains no samples")

    captured = gpu_environment.get("nvidia_devices")
    sampled = gpu_summary.get("devices")
    if not isinstance(captured, list) or not captured:
        raise RuntimeError("GPU environment artifact contains no NVIDIA devices")
    if not isinstance(sampled, list) or not sampled:
        raise RuntimeError("GPU telemetry summary contains no devices")
    captured_by_uuid = {str(device.get("uuid")): device for device in captured}
    sampled_by_uuid = {str(device.get("uuid")): device for device in sampled}
    if set(captured_by_uuid) != set(sampled_by_uuid):
        raise RuntimeError(
            "GPU telemetry UUID set differs from pre-training identity capture: "
            f"captured={sorted(captured_by_uuid)}, sampled={sorted(sampled_by_uuid)}"
        )
    for uuid, identity in captured_by_uuid.items():
        sample = sampled_by_uuid[uuid]
        for key in ("name", "driver_version"):
            if str(sample.get(key)) != str(identity.get(key)):
                raise RuntimeError(
                    f"GPU telemetry identity mismatch for {uuid}/{key}: "
                    f"{sample.get(key)!r} != {identity.get(key)!r}"
                )
        expected_total = float(identity.get("total_memory_mib"))
        sampled_total = float(sample.get("total_memory_mib"))
        if abs(expected_total - sampled_total) > 1.0:
            raise RuntimeError(
                f"GPU telemetry memory mismatch for {uuid}: "
                f"{sampled_total} != {expected_total} MiB"
            )


class _GpuTelemetrySampler:
    HEADER = (
        "timestamp,index,uuid,name,driver_version,total_memory_mib,"
        "used_memory_mib,free_memory_mib,utilization_percent\n"
    )

    def __init__(self, runtime: Runtime, output_dir: Path) -> None:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            raise RuntimeError("nvidia-smi is required for GPU telemetry")
        self.path = output_dir / "gpu_telemetry.csv"
        self.summary_path = output_dir / "gpu_telemetry_summary.json"
        self._handle = self.path.open("w", encoding="utf-8", buffering=1)
        self._handle.write(self.HEADER)
        self._process = subprocess.Popen(
            [
                nvidia_smi,
                "--query-gpu=timestamp,index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                "1000",
            ],
            cwd=REPO_ROOT,
            env=_subprocess_environment(runtime),
            stdout=self._handle,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def stop(self) -> dict[str, Any]:
        if self._process.poll() is None:
            self._process.terminate()
        try:
            _, stderr = self._process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            _, stderr = self._process.communicate(timeout=15)
        self._handle.close()
        if self._process.returncode not in (0, -15):
            raise RuntimeError(
                "nvidia-smi telemetry sampler failed: " + str(stderr).strip()[-2000:]
            )
        summary = _summarize_nvidia_csv(self.path)
        summary["captured_at_utc"] = _utc_now()
        summary["sampling_interval_ms"] = 1000
        _write_json_atomic(self.summary_path, summary)
        return summary


_NUMBER_PATTERN = r"[-+0-9.eE]+"
_AMP_SKIP_RE = re.compile(
    rf"amp_step_skipped:\s*({_NUMBER_PATTERN})"
    rf"(?:\s*\(({_NUMBER_PATTERN})\))?"
)
_AMP_SCALE_RE = re.compile(
    rf"amp_scale:\s*({_NUMBER_PATTERN})"
    rf"(?:\s*\(({_NUMBER_PATTERN})\))?"
)
_LOSS_RE = re.compile(rf"(?:^|\s)loss:\s*({_NUMBER_PATTERN})")
_MAX_MEMORY_RE = re.compile(r"max mem:\s*([-+0-9.eE]+)")


def _flatten_meter_values(
    matches: Iterable[tuple[str, str]],
) -> list[float]:
    return [
        float(value)
        for match in matches
        for value in match
        if value != ""
    ]


def _training_numerical_status(info_log: Path, console_log: Path) -> dict[str, Any]:
    text = (
        info_log.read_text(encoding="utf-8", errors="replace")
        + "\n"
        + console_log.read_text(encoding="utf-8", errors="replace")
    )
    lower = text.lower()
    failure_markers = (
        "loss is nan",
        "loss is inf",
        "non-finite gradients",
        "floatingpointerror",
    )
    observed_failures = [marker for marker in failure_markers if marker in lower]
    amp_skip_meters = _AMP_SKIP_RE.findall(text)
    amp_skips = _flatten_meter_values(amp_skip_meters)
    amp_scales = _flatten_meter_values(_AMP_SCALE_RE.findall(text))
    losses = [float(value) for value in _LOSS_RE.findall(text)]
    max_memory = [float(value) for value in _MAX_MEMORY_RE.findall(text)]
    finite_losses = all(math.isfinite(value) for value in losses)
    finite_positive_scales = bool(amp_scales) and all(
        math.isfinite(value) and value > 0.0 for value in amp_scales
    )
    finite_nonnegative_memory = bool(max_memory) and all(
        math.isfinite(value) and value >= 0.0 for value in max_memory
    )
    max_amp_skip = max(amp_skips, default=None)
    passed = (
        bool(losses)
        and bool(amp_skip_meters)
        and not observed_failures
        and finite_losses
        and finite_positive_scales
        and finite_nonnegative_memory
        and max_amp_skip == 0.0
    )
    if not passed:
        raise RuntimeError(
            "training numerical/AMP audit failed: "
            f"markers={observed_failures}, finite_losses={finite_losses}, "
            f"finite_positive_amp_scales={finite_positive_scales}, "
            f"finite_nonnegative_cuda_memory={finite_nonnegative_memory}, "
            f"max_amp_step_skipped={max_amp_skip}"
        )
    return {
        "status": "passed",
        "finite_loss_observations": len(losses),
        "loss_values_all_finite": finite_losses,
        "amp_enabled": True,
        "amp_skip_observations": len(amp_skip_meters),
        "amp_meter_observations": len(amp_skip_meters),
        "amp_skip_values_audited": len(amp_skips),
        "max_amp_step_skipped": max_amp_skip,
        "amp_scale_values_audited": len(amp_scales),
        "min_amp_scale": min(amp_scales),
        "max_amp_scale": max(amp_scales),
        "torch_cuda_max_memory_allocated_mib_from_log": (
            max(max_memory) if max_memory else None
        ),
        "torch_cuda_max_memory_reserved": {
            "available": False,
            "reason": "current training checkpoint/log contract does not emit max_memory_reserved",
        },
    }


_SAFE_CHECKPOINT_INSPECT_SCRIPT = r"""
import json
import sys

import numpy as np
import torch

path = sys.argv[1]
numpy_core = getattr(np, "_core", np.core)
safe_globals = [
    numpy_core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    type(np.dtype(np.uint32)),
]
with torch.serialization.safe_globals(safe_globals):
    payload = torch.load(path, map_location="cpu", weights_only=True)
if not isinstance(payload, dict):
    raise TypeError("checkpoint payload must be a dict")
args = payload.get("args")
if hasattr(args, "__dict__"):
    args = vars(args)
if not isinstance(args, dict):
    raise TypeError("checkpoint args must be a dict")
wanted = [
    "seed", "batch_size", "max_train_iters", "iter_checkpoint_interval",
    "config_file", "datasets", "output_dir", "pretrain_model_path",
    "resume",
    "stage_b_v15_scorer_init_checkpoint", "stage_b_v15_scorer_init_audit",
    "stage_b_v20_acc50_aligned_hard_negatives", "stage_b_v11_candidate_topk",
    "stage_b_v11_positive_iou_threshold", "stage_b_v11_negative_iou_threshold",
    "stage_b_v21_token_objective", "stage_b_v21_token_weight",
    "stage_b_v21_token_positive_weight", "stage_b_v21_token_shared_weight",
    "stage_b_v21_token_edit_weight", "stage_b_v21_token_focal_alpha",
    "stage_b_v21_token_focal_gamma", "stage_b_v11_predicate_tn_rank_weight",
    "stage_b_v21_allow_legacy_token_diff_fallback",
    "stage_b_v19_allow_scope_labeled_tn_ablation",
    "stage_b_v23_ablation_table", "stage_b_v23_table_id",
    "stage_b_v23_objective_contract", "stage_b_v23_tn_token_provenance_contract",
    "stage_b_v19_table_b_id", "stage_b_v19_table_b_scope_allowlist",
    "stage_b_v19_table_b_audit", "stage_b_v19_table_b_audit_sha256",
    "stage_b_v19_table_b_allow_single_edit_token_provenance",
    "stage_b_v15_tail_queue_positive_trust_weight",
    "stage_b_v22_table_id",
    "stage_b_v22_score_ownership", "stage_b_v22_train_phase",
    "stage_b_v22_objective_fidelity", "stage_b_v22_gradient_diagnostic_interval",
    "stage_b_v22_phase_index", "stage_b_v22_phase_count",
    "stage_b_v22_requires_rank_phase_checkpoint", "stage_b_v22_probe_only",
    "only_train_keywords", "only_train_exclude_keywords", "skip_eval", "amp",
]
result = {
    "top_level_keys": sorted(payload),
    "has_complete_training_state": all(
        key in payload
        for key in ("model", "criterion", "optimizer", "lr_scheduler", "scaler")
    ),
    "epoch": payload.get("epoch"),
    "iteration": payload.get("iteration"),
    "epoch_finished": payload.get("epoch_finished"),
    "checkpoint_reason": payload.get("checkpoint_reason"),
    "checkpoint_cuda_memory": {
        key: payload.get(key)
        for key in (
            "torch_cuda_max_memory_allocated_bytes",
            "torch_cuda_max_memory_reserved_bytes",
        )
    },
    "args": {key: args.get(key) for key in wanted},
}
print(json.dumps(result, sort_keys=True))
"""


def _subprocess_environment(runtime: Runtime) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = runtime.cuda_visible_devices
    environment["DATA_ROOT"] = str(runtime.data_root)
    environment["OMP_NUM_THREADS"] = str(runtime.omp_num_threads)
    environment["TORCH_MP_SHARING_STRATEGY"] = runtime.mp_sharing_strategy
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not old_pythonpath
        else str(REPO_ROOT) + os.pathsep + old_pythonpath
    )
    return environment


def _inspect_checkpoint_safely(runtime: Runtime, checkpoint: Path) -> dict[str, Any]:
    environment = _subprocess_environment(runtime)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        completed = subprocess.run(
            [
                str(runtime.python),
                "-c",
                _SAFE_CHECKPOINT_INSPECT_SCRIPT,
                str(checkpoint),
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("safe checkpoint metadata inspection timed out") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "safe weights-only checkpoint metadata inspection failed: "
            + completed.stderr.strip()[-4000:]
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("safe checkpoint inspector returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("safe checkpoint metadata must be a mapping")
    return payload


def _resolved_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"checkpoint path metadata is invalid: {value!r}")
    return Path(value).expanduser().resolve(strict=False)


def _expected_table_id(row: MatrixRow, phase: Phase) -> str | None:
    if row.table == "B":
        return None
    if row.row_id != "S3":
        return row.row_id
    return {
        "isolation_probe": "S3-isolation-probe",
        "rank": "S3-rank",
        "confidence": "S3-confidence",
    }[phase.phase_id]


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    phase: Phase,
    output_dir: Path,
    pretrain_path: Path,
    scorer_audit: Mapping[str, Any] | None,
) -> None:
    if metadata.get("has_complete_training_state") is not True:
        raise RuntimeError("iteration checkpoint lacks complete training state")
    if int(metadata.get("iteration", -1)) != phase.updates:
        raise RuntimeError(
            f"checkpoint iteration mismatch: expected {phase.updates}, "
            f"got {metadata.get('iteration')}"
        )
    if metadata.get("checkpoint_reason") != "max_train_iters":
        raise RuntimeError("iteration checkpoint reason must be max_train_iters")
    if metadata.get("epoch_finished") is not False:
        raise RuntimeError("max_train_iters checkpoint must be a mid-epoch state")
    args = metadata.get("args")
    if not isinstance(args, Mapping):
        raise RuntimeError("iteration checkpoint args metadata is missing")
    phase_config = runpy.run_path(
        str((REPO_ROOT / phase.config).resolve(strict=True))
    )
    expected_scalars = {
        "seed": seed,
        "batch_size": runtime.batch_size,
        "max_train_iters": phase.updates,
        "iter_checkpoint_interval": _checkpoint_interval(runtime, phase),
        "stage_b_v20_acc50_aligned_hard_negatives": True,
        "stage_b_v11_candidate_topk": 50,
        "stage_b_v11_positive_iou_threshold": 0.5,
        "stage_b_v11_negative_iou_threshold": 0.499,
        "stage_b_v21_token_objective": "edit_bce",
        "stage_b_v21_token_weight": 1.0,
        "stage_b_v21_token_positive_weight": 1.0,
        "stage_b_v21_token_shared_weight": 0.25,
        "stage_b_v21_token_edit_weight": 1.0,
        "stage_b_v21_token_focal_alpha": 0.25,
        "stage_b_v21_token_focal_gamma": 2.0,
        "stage_b_v11_predicate_tn_rank_weight": 1.0,
        "stage_b_v21_allow_legacy_token_diff_fallback": False,
        "stage_b_v22_gradient_diagnostic_interval": phase.diagnostic_interval,
        "stage_b_v15_tail_queue_positive_trust_weight": phase_config.get(
            "stage_b_v15_tail_queue_positive_trust_weight"
        ),
        "only_train_keywords": phase_config.get("only_train_keywords"),
        "only_train_exclude_keywords": phase_config.get(
            "only_train_exclude_keywords"
        ),
        "skip_eval": True,
        "amp": True,
    }
    if row.table == "B":
        expected_audit, expected_audit_sha, provenance_contract = (
            _table_b_audit_contract(row)
        )
        expected_scalars.update(
            {
                "stage_b_v19_allow_scope_labeled_tn_ablation": row.row_id != "D0",
                "stage_b_v23_ablation_table": "B",
                "stage_b_v23_table_id": row.row_id,
                "stage_b_v23_objective_contract": (
                    "v19_base_plus_gate_acc50_hardneg_v21_l4"
                ),
                "stage_b_v23_tn_token_provenance_contract": (
                    provenance_contract
                ),
                "stage_b_v19_table_b_id": row.row_id,
                "stage_b_v19_table_b_scope_allowlist": (
                    [] if row.row_id == "D0" else [str(row.tn_scope)]
                ),
                "stage_b_v19_table_b_audit": (
                    str(expected_audit.relative_to(REPO_ROOT))
                ),
                "stage_b_v19_table_b_audit_sha256": expected_audit_sha,
            }
        )
    else:
        expected_scalars.update(
            {
                "stage_b_v19_allow_scope_labeled_tn_ablation": True,
                "stage_b_v19_table_b_id": "D3",
                "stage_b_v19_table_b_scope_allowlist": [
                    "proposal_covered_verified"
                ],
                "stage_b_v19_table_b_audit": (
                    "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
                ),
                "stage_b_v19_table_b_audit_sha256": TABLE_B_AUDIT_SHA256,
                "stage_b_v19_table_b_allow_single_edit_token_provenance": True,
                "stage_b_v22_table_id": _expected_table_id(row, phase),
                "stage_b_v22_score_ownership": row.score_ownership,
                "stage_b_v22_train_phase": (
                    "joint" if row.row_id != "S3" else phase.phase_id
                ),
                "stage_b_v22_objective_fidelity": phase_config.get(
                    "stage_b_v22_objective_fidelity"
                ),
                "stage_b_v22_phase_index": phase_config.get(
                    "stage_b_v22_phase_index"
                ),
                "stage_b_v22_phase_count": phase_config.get(
                    "stage_b_v22_phase_count"
                ),
                "stage_b_v22_requires_rank_phase_checkpoint": phase_config.get(
                    "stage_b_v22_requires_rank_phase_checkpoint"
                ),
                "stage_b_v22_probe_only": phase_config.get(
                    "stage_b_v22_probe_only"
                ),
            }
        )
    for key, expected in expected_scalars.items():
        if args.get(key) != expected:
            raise RuntimeError(
                f"checkpoint args mismatch for {key}: expected {expected!r}, "
                f"got {args.get(key)!r}"
            )
    if args.get("resume") not in (None, ""):
        raise RuntimeError(
            "paper phase unexpectedly resumed optimizer state: "
            f"{args.get('resume')!r}"
        )
    expected_paths = {
        "config_file": (REPO_ROOT / phase.config).resolve(strict=True),
        "datasets": _row_dataset(row),
        "output_dir": output_dir,
        "pretrain_model_path": pretrain_path,
    }
    for key, expected in expected_paths.items():
        if _resolved_path(args.get(key)) != expected.resolve(strict=False):
            raise RuntimeError(
                f"checkpoint args path mismatch for {key}: expected {expected}, "
                f"got {args.get(key)!r}"
            )
    if phase.scorer_warmstart:
        if _resolved_path(args.get("stage_b_v15_scorer_init_checkpoint")) != (
            runtime.scorer_warmstart.resolve(strict=False)
        ):
            raise RuntimeError("checkpoint scorer warm-start path mismatch")
        embedded = args.get("stage_b_v15_scorer_init_audit")
        if not isinstance(embedded, Mapping) or scorer_audit is None:
            raise RuntimeError("checkpoint lacks scorer warm-start audit")
        if dict(embedded) != dict(scorer_audit):
            raise RuntimeError("checkpoint scorer audit differs from persisted audit")
    else:
        if args.get("stage_b_v15_scorer_init_checkpoint") not in (None, ""):
            raise RuntimeError("S3 confidence phase reapplied scorer warm-start")
        if args.get("stage_b_v15_scorer_init_audit") is not None:
            raise RuntimeError("S3 confidence phase embedded a new scorer-init audit")


def _perform_postflight(
    manifest: Mapping[str, Any],
    *,
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    phase: Phase,
    cache: token_launcher.HashCache,
    rank_checkpoint: Path | None,
) -> dict[str, Any]:
    output_dir = Path(str(manifest["output_dir"]))
    required = {
        "checkpoint": output_dir / "checkpoint_iter.pth",
        "native_info_log": output_dir / "info.txt",
        "train_console_log": output_dir / "train_console.log",
        "launch_manifest": output_dir / "launch_manifest.json",
        "gpu_environment": output_dir / "gpu_environment.json",
        "gpu_telemetry": output_dir / "gpu_telemetry.csv",
        "gpu_telemetry_summary": output_dir / "gpu_telemetry_summary.json",
    }
    if phase.scorer_warmstart:
        required["scorer_init_audit"] = (
            output_dir / "stage_b_v15_scorer_init_audit.json"
        )
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"postflight is missing required artifacts: {missing}")
    for name in ("native_info_log", "train_console_log"):
        if required[name].stat().st_size <= 0:
            raise RuntimeError(f"postflight artifact is empty: {required[name]}")

    # Perform and persist the full digest audit before inspecting derived
    # outputs. If a later check fails, the immutable-input evidence survives as
    # its own artifact rather than existing only in an incomplete postflight.
    input_rehash = _rehash_inputs(manifest)
    input_rehash_path = output_dir / "input_rehash.json"
    _write_json_atomic(input_rehash_path, input_rehash)
    required["input_rehash"] = input_rehash_path
    scorer_audit: Mapping[str, Any] | None = None
    if phase.scorer_warmstart:
        scorer_audit = json.loads(
            required["scorer_init_audit"].read_text(encoding="utf-8")
        )
        expected_audit = {
            "schema": "stage_b_v15_scorer_init/v1",
            "status": "applied",
            "source_sha256": next(
                record["sha256"]
                for record in manifest["inputs"]["records"]
                if record["role"] == "scorer_warmstart"
            ),
            "loaded_num_layers": 3,
        }
        for key, expected in expected_audit.items():
            if scorer_audit.get(key) != expected:
                raise RuntimeError(
                    f"scorer initialization audit mismatch for {key}: "
                    f"expected {expected!r}, got {scorer_audit.get(key)!r}"
                )
        if _resolved_path(scorer_audit.get("resolved_source_path")) != (
            runtime.scorer_warmstart.resolve(strict=False)
        ):
            raise RuntimeError("scorer initialization audit source path mismatch")
    else:
        unexpected = output_dir / "stage_b_v15_scorer_init_audit.json"
        if unexpected.exists():
            raise RuntimeError(
                "S3 confidence phase must inherit rank model state without "
                "reapplying scorer initialization"
            )

    metadata = _inspect_checkpoint_safely(runtime, required["checkpoint"])
    pretrain = (
        rank_checkpoint
        if phase.pretrain_source == "rank_phase_checkpoint"
        else runtime.stage_a_init
    )
    if pretrain is None:
        raise RuntimeError("rank checkpoint missing for S3 confidence postflight")
    _validate_checkpoint_metadata(
        metadata,
        runtime=runtime,
        row=row,
        seed=seed,
        phase=phase,
        output_dir=output_dir,
        pretrain_path=pretrain,
        scorer_audit=scorer_audit,
    )
    if phase.diagnostic_interval > 0:
        logs = (
            required["native_info_log"].read_text(encoding="utf-8", errors="replace")
            + "\n"
            + required["train_console_log"].read_text(
                encoding="utf-8", errors="replace"
            )
        )
        expected_fragment = (
            "stage_b_v22_grad_cosine"
            if row.row_id in {"S0", "S1"}
            else "stage_b_v22_branch_isolation_pass"
        )
        if expected_fragment not in logs:
            raise RuntimeError(
                f"configured gradient diagnostic did not appear in logs: "
                f"{expected_fragment}"
            )
    gpu_environment = json.loads(
        required["gpu_environment"].read_text(encoding="utf-8")
    )
    gpu_summary = json.loads(
        required["gpu_telemetry_summary"].read_text(encoding="utf-8")
    )
    _validate_gpu_telemetry_contract(gpu_environment, gpu_summary)
    numerical_status = _training_numerical_status(
        required["native_info_log"], required["train_console_log"]
    )
    checkpoint_memory = metadata.get("checkpoint_cuda_memory", {})
    checkpoint_memory_available = isinstance(checkpoint_memory, Mapping) and any(
        value is not None for value in checkpoint_memory.values()
    )
    pretrain_role = (
        "rank_phase_model_state_pretrain"
        if phase.pretrain_source == "rank_phase_checkpoint"
        else "stage_a_initializer"
    )
    pretrain_records = [
        record
        for record in manifest["inputs"]["records"]
        if record.get("role") == pretrain_role
    ]
    if len(pretrain_records) != 1:
        raise RuntimeError(
            f"expected one {pretrain_role} manifest record, got "
            f"{len(pretrain_records)}"
        )
    pretrain_record = pretrain_records[0]
    if _resolved_path(pretrain_record.get("path")) != pretrain.resolve(strict=True):
        raise RuntimeError("pretrain lineage record path mismatch")
    return {
        "schema": "pivot.stageb.paper_ablation_phase_postflight/v1",
        "status": "passed",
        "validated_at_utc": _utc_now(),
        "run_id": manifest["run_id"],
        "phase_id": phase.phase_id,
        "checkpoint_metadata": metadata,
        "input_rehash": input_rehash,
        "gpu_environment": gpu_environment,
        "gpu_telemetry_summary": gpu_summary,
        "numerical_status": numerical_status,
        "checkpoint_cuda_memory": (
            {
                "available": True,
                "values": dict(checkpoint_memory),
            }
            if checkpoint_memory_available
            else {
                "available": False,
                "values": dict(checkpoint_memory) if isinstance(checkpoint_memory, Mapping) else {},
                "reason": "current checkpoint schema does not emit CUDA max allocated/reserved counters",
            }
        ),
        "artifacts": {
            name: token_launcher._file_record(path, cache)
            for name, path in required.items()
            if name != "launch_manifest"
        },
        "model_state_ancestry": {
            "pretrain_path": str(pretrain.resolve(strict=False)),
            "pretrain_sha256": pretrain_record["sha256"],
            "pretrain_manifest_role": pretrain_role,
            "pretrain_mode": "model_state_only_no_optimizer_resume",
            "checkpoint_resume_argument": None,
            "scorer_warmstart_applied": phase.scorer_warmstart,
            "generated_dependency": manifest.get("generated_dependency"),
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    token_launcher._write_json_atomic(path, payload)


def _orchestration_status_path() -> Path | None:
    raw = os.environ.get("PIVOT_ORCHESTRATION_STATUS", "").strip()
    return Path(raw).expanduser().resolve(strict=False) if raw else None


def _update_orchestration_status(
    path: Path | None,
    *,
    status: str,
    **fields: Any,
) -> None:
    if path is None:
        return
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            payload.update(existing)
    payload.update(fields)
    payload["schema"] = "pivot.stageb.paper_ablation_orchestration_status/v1"
    payload["status"] = status
    payload["updated_at_utc"] = _utc_now()
    payload["pid"] = os.getpid()
    _write_json_atomic(path, payload)


_ORCHESTRATION_NONTERMINAL_STATUSES = frozenset(
    {"prepared", "launched", "starting", "preflight_passed", "running"}
)
_ORCHESTRATION_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "spawn_failed", "hard_terminated_unknown"}
)


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must contain a JSON object: {path}")
    return payload


def _read_process_identity(pid: int) -> dict[str, Any]:
    """Read enough Linux process identity to detect later PID reuse."""

    proc_dir = Path("/proc") / str(pid)
    try:
        stat_text = (proc_dir / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return {
            "available": False,
            "pid": pid,
            "reason": "proc_entry_missing",
        }
    except OSError as exc:
        return {
            "available": False,
            "pid": pid,
            "reason": f"proc_stat_unreadable:{type(exc).__name__}",
        }
    closing_paren = stat_text.rfind(")")
    if closing_paren < 0:
        return {
            "available": False,
            "pid": pid,
            "reason": "proc_stat_malformed",
        }
    fields = stat_text[closing_paren + 2 :].split()
    if len(fields) <= 19:
        return {
            "available": False,
            "pid": pid,
            "reason": "proc_stat_too_short",
        }
    try:
        start_time_ticks = int(fields[19])
    except ValueError:
        return {
            "available": False,
            "pid": pid,
            "reason": "proc_start_time_malformed",
        }
    try:
        command = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ")
        command_text = command.decode("utf-8", errors="replace").strip()
    except OSError:
        command_text = ""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        boot_id = ""
    return {
        "available": True,
        "pid": pid,
        "state": fields[0],
        "start_time_ticks": start_time_ticks,
        "boot_id": boot_id or None,
        "command": command_text,
    }


def _probe_pid_liveness(
    pid: int | None,
    *,
    expected_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checked_at = _utc_now()
    if pid is None or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return {
            "checked_at_utc": checked_at,
            "pid": pid,
            "probe": "os.kill(pid, 0) plus /proc identity",
            "state": "unknown_no_valid_pid",
            "running": None,
        }
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {
            "checked_at_utc": checked_at,
            "pid": pid,
            "probe": "os.kill(pid, 0) plus /proc identity",
            "state": "not_found",
            "running": False,
        }
    except PermissionError:
        kill_state = "permission_denied_process_exists"
    except OSError as exc:
        return {
            "checked_at_utc": checked_at,
            "pid": pid,
            "probe": "os.kill(pid, 0) plus /proc identity",
            "state": "probe_error",
            "running": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        kill_state = "process_exists"

    observed = _read_process_identity(pid)
    result: dict[str, Any] = {
        "checked_at_utc": checked_at,
        "pid": pid,
        "probe": "os.kill(pid, 0) plus /proc identity",
        "state": "alive",
        "running": True,
        "kill_probe": kill_state,
        "observed_identity": observed,
    }
    if observed.get("available") and observed.get("state") == "Z":
        result.update(state="zombie", running=False)
        return result
    if expected_identity and expected_identity.get("available"):
        result["expected_identity"] = dict(expected_identity)
        if not observed.get("available"):
            result.update(state="identity_unavailable", running=None)
            return result
        expected_start = expected_identity.get("start_time_ticks")
        observed_start = observed.get("start_time_ticks")
        expected_boot = expected_identity.get("boot_id")
        observed_boot = observed.get("boot_id")
        start_mismatch = expected_start != observed_start
        boot_mismatch = bool(expected_boot and observed_boot and expected_boot != observed_boot)
        if start_mismatch or boot_mismatch:
            result.update(state="pid_reused", running=False)
            return result
        result["identity_match"] = True
    else:
        result["identity_match"] = None
        result["identity_note"] = (
            "launch predates process-identity capture; a live PID is retained "
            "as active conservatively"
        )
    return result


def _tail_file_evidence(
    path: Path,
    *,
    max_bytes: int = 16384,
    max_lines: int = 12,
    max_line_chars: int = 4096,
) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not record["exists"]:
        return record
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
    except OSError as exc:
        record["read_error"] = f"{type(exc).__name__}: {exc}"
        return record
    text_tail = data.decode("utf-8", errors="replace")
    raw_lines = text_tail.splitlines()[-max_lines:]
    truncated_line_count = sum(len(line) > max_line_chars for line in raw_lines)
    tail_lines = [
        line if len(line) <= max_line_chars else line[-max_line_chars:]
        for line in raw_lines
    ]
    record.update(
        {
            "size_bytes": size,
            "read_tail_bytes": len(data),
            "byte_tail_truncated": size > len(data),
            "ends_with_newline": data.endswith(b"\n"),
            "tail_lines": tail_lines,
            "tail_line_truncated_count": truncated_line_count,
            "tail_line_policy": (
                f"lines longer than {max_line_chars} characters retain only "
                "their suffix"
            ),
        }
    )
    return record


def _path_list(value: Any) -> list[Path]:
    if not isinstance(value, list):
        return []
    paths: list[Path] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            paths.append(Path(item).expanduser().resolve(strict=False))
    return paths


def _artifact_status_record(path: Path, *, kind: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "path": str(path),
        "exists": path.is_file(),
    }
    if not record["exists"]:
        return record
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"
        return record
    if not isinstance(payload, dict):
        record["parse_error"] = "top-level JSON is not an object"
        return record
    for key in (
        "status",
        "run_id",
        "phase_id",
        "failure_phase",
        "failure_error",
        "postflight_error",
        "error",
        "returncode",
        "finished_at_utc",
    ):
        if key in payload:
            record[key] = payload[key]
    return record


def _collect_artifact_evidence(run_roots: Sequence[Path]) -> dict[str, Any]:
    sequences: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    postflights: list[dict[str, Any]] = []
    for root in run_roots:
        sequences.append(
            _artifact_status_record(
                root / "sequence_manifest.json", kind="sequence_manifest"
            )
        )
        if root.is_dir():
            phases.extend(
                _artifact_status_record(path, kind="phase_launch_manifest")
                for path in sorted(root.rglob("launch_manifest.json"))
            )
            postflights.extend(
                _artifact_status_record(path, kind="phase_postflight")
                for path in sorted(root.rglob("postflight.json"))
            )
    failed_records = [
        record
        for record in (*sequences, *phases)
        if record.get("status") == "failed"
    ]
    all_sequences_completed = bool(run_roots) and all(
        record.get("status") == "completed" for record in sequences
    )
    if failed_records:
        classification = "failed"
        reason = "explicit_failed_run_or_phase_manifest"
    elif all_sequences_completed:
        classification = "completed"
        reason = "all_expected_sequence_manifests_explicitly_completed"
    else:
        classification = None
        reason = "no_explicit_terminal_artifact"
    return {
        "classification": classification,
        "reason": reason,
        "sequence_manifests": sequences,
        "phase_launch_manifests": phases,
        "phase_postflights": postflights,
    }


def _collect_tail_evidence(
    job_dir: Path,
    launch: Mapping[str, Any],
    run_roots: Sequence[Path],
) -> dict[str, Any]:
    raw_log = launch.get("orchestrator_log")
    orchestrator_log = (
        Path(raw_log).expanduser().resolve(strict=False)
        if isinstance(raw_log, str) and raw_log.strip()
        else job_dir / "orchestrator.log"
    )
    training_logs: list[Path] = []
    telemetry_files: list[Path] = []
    for root in run_roots:
        if root.is_dir():
            training_logs.extend(sorted(root.rglob("train_console.log")))
            telemetry_files.extend(sorted(root.rglob("gpu_telemetry.csv")))
    return {
        "orchestrator_log": _tail_file_evidence(orchestrator_log),
        "training_logs": [
            _tail_file_evidence(path) for path in dict.fromkeys(training_logs)
        ],
        "gpu_telemetry": [
            _tail_file_evidence(path) for path in dict.fromkeys(telemetry_files)
        ],
        "interpretation_policy": (
            "tails are preserved as observations only; text is never used to "
            "infer OOM or another termination cause"
        ),
    }


def _detached_job_observation(
    job_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    job_dir = job_dir.expanduser().resolve(strict=True)
    if not job_dir.is_dir():
        raise NotADirectoryError(f"detached job path is not a directory: {job_dir}")
    launch = _read_json_object(
        job_dir / "launch.json", description="detached launch record"
    )
    status_path = job_dir / "status.json"
    status_payload = _read_json_object(
        status_path, description="detached orchestration status"
    )
    persisted_status = status_payload.get("status")
    if not isinstance(persisted_status, str):
        raise RuntimeError(f"orchestration status has no string status: {status_path}")
    run_roots = _path_list(
        status_payload.get("expected_run_roots")
        or launch.get("expected_run_roots")
    )
    child_pid = launch.get("child_pid")
    pid_source = "launch.child_pid"
    if isinstance(child_pid, bool) or not isinstance(child_pid, int):
        if persisted_status != "prepared" and isinstance(status_payload.get("pid"), int):
            child_pid = status_payload["pid"]
            pid_source = "status.pid"
        else:
            child_pid = None
            pid_source = "unavailable"
    liveness = _probe_pid_liveness(
        child_pid,
        expected_identity=(
            launch.get("child_process_identity")
            if isinstance(launch.get("child_process_identity"), Mapping)
            else None
        ),
    )
    liveness["pid_source"] = pid_source
    artifacts = _collect_artifact_evidence(run_roots)

    observed_status = persisted_status
    reason = "persisted_status_retained"
    reconciliation_required = False
    if persisted_status in _ORCHESTRATION_TERMINAL_STATUSES:
        reason = "persisted_status_is_explicitly_terminal"
    elif persisted_status not in _ORCHESTRATION_NONTERMINAL_STATUSES:
        reason = "persisted_status_is_unrecognized_and_is_retained_conservatively"
    elif launch.get("status") == "spawn_failed":
        observed_status = "spawn_failed"
        reason = "launch_record_explicitly_reports_spawn_failed"
        reconciliation_required = observed_status != persisted_status
    elif liveness.get("running") is True:
        reason = "orchestrator_process_is_alive"
    elif liveness.get("running") is None:
        reason = "orchestrator_liveness_is_unresolved"
    else:
        terminal_artifact = artifacts.get("classification")
        if terminal_artifact in {"completed", "failed"}:
            observed_status = str(terminal_artifact)
            reason = str(artifacts["reason"])
        else:
            observed_status = "hard_terminated_unknown"
            reason = (
                "orchestrator_process_is_not_running_and_no_explicit_terminal_"
                "status_or_artifact_exists"
            )
        reconciliation_required = observed_status != persisted_status

    tails = _collect_tail_evidence(job_dir, launch, run_roots)
    observation: dict[str, Any] = {
        "schema": "pivot.stageb.paper_ablation_detached_observation/v1",
        "observed_at_utc": _utc_now(),
        "job_dir": str(job_dir),
        "persisted_status": persisted_status,
        "observed_status": observed_status,
        "reason": reason,
        "reconciliation_required": reconciliation_required,
        "pid_liveness": liveness,
        "artifact_evidence": artifacts,
        "evidence_tails": tails,
        "termination_cause": (
            "unknown" if observed_status == "hard_terminated_unknown" else None
        ),
        "oom_classification": (
            "not_established"
            if observed_status == "hard_terminated_unknown"
            else "not_applicable"
        ),
        "cause_inference_policy": (
            "terminal class uses process liveness and explicit JSON status only; "
            "log and telemetry text cannot establish OOM"
        ),
    }
    return observation, status_payload


def _inspect_or_reconcile_detached_job(
    job_dir: Path,
    *,
    mutate: bool,
) -> dict[str, Any]:
    observation, _ = _detached_job_observation(job_dir)
    observation["mutated"] = False
    if not mutate or not observation["reconciliation_required"]:
        return observation

    # Re-read every source immediately before the atomic update. A live child
    # that wrote a terminal state during inspection always wins.
    observation, status_payload = _detached_job_observation(job_dir)
    observation["mutated"] = False
    if (
        not observation["reconciliation_required"]
        or observation["pid_liveness"].get("running") is not False
    ):
        return observation
    previous_status = observation["persisted_status"]
    target_status = observation["observed_status"]
    reconciliation = {
        "schema": "pivot.stageb.paper_ablation_detached_reconciliation/v1",
        "reconciled_at_utc": _utc_now(),
        "previous_status": previous_status,
        "status": target_status,
        "reason": observation["reason"],
        "pid_liveness": observation["pid_liveness"],
        "artifact_evidence": observation["artifact_evidence"],
        "evidence_tails": observation["evidence_tails"],
        "termination_cause": observation["termination_cause"],
        "oom_classification": observation["oom_classification"],
        "cause_inference_policy": observation["cause_inference_policy"],
        "reconciler_pid": os.getpid(),
    }
    updated_status = dict(status_payload)
    updated_status["status"] = target_status
    updated_status["updated_at_utc"] = reconciliation["reconciled_at_utc"]
    updated_status["reconciled_from_status"] = previous_status
    updated_status["reconciliation"] = reconciliation
    _write_json_atomic(
        Path(observation["job_dir"]) / "status.json", updated_status
    )
    observation["persisted_status"] = target_status
    observation["previous_persisted_status"] = previous_status
    observation["reconciliation_required"] = False
    observation["mutated"] = True
    observation["reconciliation"] = reconciliation
    return observation


def _parse_run_id(value: str) -> tuple[MatrixRow, int]:
    try:
        row_value, seed_value = value.split(":", 1)
        row = ROW_BY_ID_CASEFOLD[row_value.casefold()]
        seed = int(seed_value)
    except (KeyError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"run id must be ROW:SEED with ROW in {tuple(ROW_BY_ID)} and "
            f"SEED in {SEEDS}; got {value!r}"
        ) from exc
    if seed not in SEEDS:
        raise argparse.ArgumentTypeError(
            f"training seed must be one of {SEEDS}, got {seed}"
        )
    return row, seed


def _filtered_rows(table: str) -> tuple[MatrixRow, ...]:
    if table == "all":
        return ROWS
    wanted = table.upper()
    return tuple(row for row in ROWS if row.table == wanted)


def _all_runs(table: str = "all") -> list[tuple[MatrixRow, int]]:
    return [(row, seed) for row in _filtered_rows(table) for seed in SEEDS]


def _selected_runs(args: argparse.Namespace) -> list[tuple[MatrixRow, int]]:
    if args.run_id:
        selected = list(dict.fromkeys(args.run_id))
        outside = [row.row_id for row, _ in selected if args.table != "all" and row.table != args.table.upper()]
        if outside:
            raise ValueError(
                f"selected rows do not belong to --table {args.table}: {outside}"
            )
        return selected
    if args.mode == "dry-run" or args.all:
        return _all_runs(args.table)
    raise ValueError("run requires at least one --run-id ROW:SEED or --all")


def _add_selection_arguments(parser: argparse.ArgumentParser, *, run: bool) -> None:
    parser.add_argument(
        "--table",
        choices=("all", "B", "D", "b", "d"),
        default="all",
        help="restrict selection to Table B or Table D",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        type=_parse_run_id,
        help="select one ROW:SEED; repeat to select multiple runs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every selected table row" if run else "show all rows (default)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    list_parser = subparsers.add_parser("list", help="list rows, seeds, and run ids")
    list_parser.add_argument(
        "--table", choices=("all", "B", "D", "b", "d"), default="all"
    )
    list_parser.add_argument("--json", action="store_true", help="emit JSON")
    dry_parser = subparsers.add_parser(
        "dry-run", help="hash inputs and print commands without creating run outputs"
    )
    _add_selection_arguments(dry_parser, run=False)
    dry_parser.add_argument("--manifest", type=Path)
    dry_parser.add_argument("--manifest-dir", type=Path)
    run_parser = subparsers.add_parser(
        "run", help="launch explicitly selected fresh training runs"
    )
    _add_selection_arguments(run_parser, run=True)
    detach_parser = subparsers.add_parser(
        "detach",
        help=(
            "preflight then launch an independent background orchestrator with "
            "persistent log/status artifacts"
        ),
    )
    _add_selection_arguments(detach_parser, run=True)
    detach_parser.add_argument(
        "--orchestration-root",
        type=Path,
        default=None,
        help=(
            "control-artifact root (default: PIVOT_ORCHESTRATION_ROOT or "
            "outputs/paper_cvpr_v1/orchestration/paper_ablation_matrices)"
        ),
    )
    status_parser = subparsers.add_parser(
        "status",
        help="inspect one detached job without modifying its persisted status",
    )
    status_parser.add_argument(
        "job_dir",
        type=Path,
        help="job directory containing launch.json and status.json",
    )
    reconcile_parser = subparsers.add_parser(
        "reconcile",
        help=(
            "atomically repair a stale detached status using PID liveness and "
            "explicit terminal artifacts"
        ),
    )
    reconcile_parser.add_argument(
        "job_dir",
        type=Path,
        help="job directory containing launch.json and status.json",
    )
    return parser


def _list_rows(table: str, as_json: bool) -> int:
    rows = _filtered_rows(table)
    payload = {
        "rows": [asdict(row) for row in rows],
        "seeds": list(SEEDS),
        "run_ids": [f"{row.row_id}:{seed}" for row in rows for seed in SEEDS],
        "s3_phases": [
            "isolation_probe (excluded from paper update budget)",
            "rank (half budget)",
            "confidence (half budget; model-state pretrain, no optimizer resume)",
        ],
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in rows:
            detail = row.tn_scope or row.score_ownership
            print(
                f"{row.row_id}: table={row.table}, contract={detail}, "
                f"config={row.config}, dataset={row.dataset}"
            )
        print("seeds: " + ",".join(str(seed) for seed in SEEDS))
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    selections = _selected_runs(args)
    if args.manifest is not None and len(selections) != 1:
        raise ValueError("--manifest requires exactly one --run-id")
    if args.manifest is not None and args.manifest_dir is not None:
        raise ValueError("use only one of --manifest and --manifest-dir")
    runtime = runtime_from_environment()
    cache = token_launcher.HashCache()
    for row, seed in selections:
        manifest = build_manifest(runtime, row, seed, cache)
        for phase_manifest in manifest["phases"]:
            print(
                f"[{manifest['run_id']}/{phase_manifest['phase']['phase_id']}] "
                f"{phase_manifest['command_shell']}"
            )
        if args.manifest is not None:
            _write_json_atomic(args.manifest.resolve(strict=False), manifest)
        elif args.manifest_dir is not None:
            target = (
                args.manifest_dir.resolve(strict=False)
                / row.row_id
                / f"seed{seed}.launch.json"
            )
            _write_json_atomic(target, manifest)
    return 0


def _detach(args: argparse.Namespace) -> int:
    """Preflight synchronously, then hand the matrix to a new OS session."""

    selections = _selected_runs(args)
    runtime = runtime_from_environment()
    run_roots = [output_directory(runtime, row, seed) for row, seed in selections]
    conflicts = [path for path in run_roots if path.exists()]
    if conflicts:
        rendered = "\n".join(f"  {path}" for path in conflicts)
        raise FileExistsError(
            "every selected run root must be fresh; existing paths:\n" + rendered
        )

    # This repeats in the child immediately before execution. Doing it here is
    # intentional: a trust/data/hash failure must not create a detached job.
    cache = token_launcher.HashCache()
    planned = [build_manifest(runtime, row, seed, cache) for row, seed in selections]
    root_value = args.orchestration_root
    if root_value is None:
        root_value = Path(
            os.environ.get(
                "PIVOT_ORCHESTRATION_ROOT", str(DEFAULT_ORCHESTRATION_ROOT)
            )
        )
    orchestration_root = root_value.expanduser().resolve(strict=False)
    job_name = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-pid{os.getpid()}"
    )
    job_dir = orchestration_root / job_name
    job_dir.mkdir(parents=True, exist_ok=False)
    plans_dir = job_dir / "plans"
    for (row, seed), manifest in zip(selections, planned):
        _write_json_atomic(plans_dir / row.row_id / f"seed{seed}.json", manifest)

    child_command = [
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        "run",
    ]
    for row, seed in selections:
        child_command.extend(("--run-id", f"{row.row_id}:{seed}"))
    log_path = job_dir / "orchestrator.log"
    status_path = job_dir / "status.json"
    launch_path = job_dir / "launch.json"
    launch: dict[str, Any] = {
        "schema": "pivot.stageb.paper_ablation_detached_launch/v1",
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "job_dir": str(job_dir),
        "run_ids": [f"{row.row_id}:{seed}" for row, seed in selections],
        "expected_run_roots": [str(path) for path in run_roots],
        "command": child_command,
        "command_shell": shlex.join(child_command),
        "orchestrator_log": str(log_path),
        "orchestrator_status": str(status_path),
        "plans_dir": str(plans_dir),
        "runtime": {
            "python": str(runtime.python),
            "batch_size": runtime.batch_size,
            "total_train_iters": runtime.total_train_iters,
            "cuda_visible_devices": runtime.cuda_visible_devices,
            "tn_output_root": str(runtime.tn_output_root),
            "score_output_root": str(runtime.score_output_root),
        },
    }
    _write_json_atomic(launch_path, launch)
    _update_orchestration_status(
        status_path,
        status="prepared",
        job_dir=str(job_dir),
        run_ids=launch["run_ids"],
        expected_run_roots=launch["expected_run_roots"],
    )
    environment = dict(os.environ)
    environment["PIVOT_ORCHESTRATION_STATUS"] = str(status_path)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                child_command,
                cwd=REPO_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException as exc:
        launch["status"] = "spawn_failed"
        launch["spawn_error"] = f"{type(exc).__name__}: {exc}"
        launch["finished_at_utc"] = _utc_now()
        _write_json_atomic(launch_path, launch)
        _update_orchestration_status(
            status_path,
            status="spawn_failed",
            error=launch["spawn_error"],
        )
        raise
    launch["status"] = "launched"
    launch["launched_at_utc"] = _utc_now()
    launch["child_pid"] = int(process.pid)
    launch["child_process_identity"] = _read_process_identity(int(process.pid))
    launch["child_start_new_session"] = True
    launch["stdin"] = "DEVNULL"
    launch["stdout_stderr"] = str(log_path)
    _write_json_atomic(launch_path, launch)
    print(
        json.dumps(
            {
                "status": "launched",
                "pid": int(process.pid),
                "job_dir": str(job_dir),
                "status_file": str(status_path),
                "log_file": str(log_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _run_phase(
    *,
    runtime: Runtime,
    row: MatrixRow,
    seed: int,
    phase: Phase,
    output_dir: Path,
    cache: token_launcher.HashCache,
    rank_checkpoint: Path | None,
) -> tuple[dict[str, Any], Path]:
    manifest = _phase_manifest(
        runtime,
        row,
        seed,
        phase,
        output_dir,
        cache,
        rank_checkpoint=rank_checkpoint,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "launch_manifest.json"
    manifest["status"] = "running"
    manifest["started_at_utc"] = _utc_now()
    _write_json_atomic(manifest_path, manifest)
    _verify_file_identities(manifest)
    print(
        f"[{manifest['run_id']}/{phase.phase_id}] {manifest['command_shell']}",
        flush=True,
    )
    try:
        gpu_environment = _capture_gpu_environment(runtime, output_dir)
        sampler = _GpuTelemetrySampler(runtime, output_dir)
        try:
            returncode = token_launcher._stream_subprocess(
                manifest["command"],
                runtime=runtime,
                console_log=output_dir / "train_console.log",
            )
        finally:
            gpu_summary = sampler.stop()
        manifest["gpu_environment"] = gpu_environment
        manifest["gpu_telemetry_summary"] = gpu_summary
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure_phase"] = "gpu_telemetry_or_training_process"
        manifest["failure_error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at_utc"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        raise
    manifest["returncode"] = int(returncode)
    manifest["finished_at_utc"] = _utc_now()
    if returncode != 0:
        manifest["status"] = "failed"
        manifest["failure_phase"] = "training_process"
        _write_json_atomic(manifest_path, manifest)
        raise RuntimeError(
            f"{row.row_id}:{seed}/{phase.phase_id} training exited {returncode}"
        )
    try:
        postflight = _perform_postflight(
            manifest,
            runtime=runtime,
            row=row,
            seed=seed,
            phase=phase,
            cache=cache,
            rank_checkpoint=rank_checkpoint,
        )
        postflight_path = output_dir / "postflight.json"
        _write_json_atomic(postflight_path, postflight)
        manifest["postflight"] = postflight
        manifest["postflight_artifact"] = token_launcher._file_record(
            postflight_path, cache
        )
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_phase"] = "postflight"
        manifest["postflight_error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(manifest_path, manifest)
        raise
    _write_json_atomic(manifest_path, manifest)
    return manifest, output_dir / "checkpoint_iter.pth"


def _run_body(
    args: argparse.Namespace,
    *,
    orchestration_status: Path | None,
) -> int:
    selections = _selected_runs(args)
    runtime = runtime_from_environment()
    run_roots = [output_directory(runtime, row, seed) for row, seed in selections]
    conflicts = [path for path in run_roots if path.exists()]
    if conflicts:
        rendered = "\n".join(f"  {path}" for path in conflicts)
        raise FileExistsError(
            "every selected run root must be fresh; existing paths:\n" + rendered
        )
    cache = token_launcher.HashCache()
    # Hash and validate every selected run before creating the first directory.
    planned = [build_manifest(runtime, row, seed, cache) for row, seed in selections]
    _update_orchestration_status(
        orchestration_status,
        status="preflight_passed",
        run_ids=[f"{row.row_id}:{seed}" for row, seed in selections],
        expected_run_roots=[str(path) for path in run_roots],
        completed_run_ids=[],
    )
    completed_run_ids: list[str] = []
    for (row, seed), sequence in zip(selections, planned):
        run_id = f"{row.row_id}:{seed}"
        run_root = output_directory(runtime, row, seed)
        if row.row_id == "S3":
            run_root.mkdir(parents=True, exist_ok=False)
            sequence_path = run_root / "sequence_manifest.json"
        else:
            sequence_path = run_root / "sequence_manifest.json"
        sequence["status"] = "running"
        sequence["started_at_utc"] = _utc_now()
        if row.row_id == "S3":
            _write_json_atomic(sequence_path, sequence)
        rank_checkpoint: Path | None = None
        completed_phases: list[dict[str, Any]] = []
        _update_orchestration_status(
            orchestration_status,
            status="running",
            current_run_id=run_id,
            current_phase_id=None,
            completed_run_ids=completed_run_ids,
        )
        try:
            for phase in _phases(runtime, row):
                _update_orchestration_status(
                    orchestration_status,
                    status="running",
                    current_run_id=run_id,
                    current_phase_id=phase.phase_id,
                    completed_run_ids=completed_run_ids,
                )
                output_dir = _phase_output(run_root, row, phase)
                phase_manifest, checkpoint = _run_phase(
                    runtime=runtime,
                    row=row,
                    seed=seed,
                    phase=phase,
                    output_dir=output_dir,
                    cache=cache,
                    rank_checkpoint=(
                        rank_checkpoint
                        if phase.pretrain_source == "rank_phase_checkpoint"
                        else None
                    ),
                )
                completed_phases.append(
                    {
                        "phase_id": phase.phase_id,
                        "status": phase_manifest["status"],
                        "output_dir": str(output_dir),
                        "checkpoint": token_launcher._file_record(checkpoint, cache),
                    }
                )
                if phase.phase_id == "rank":
                    rank_checkpoint = checkpoint.resolve(strict=True)
        except Exception as exc:
            sequence["status"] = "failed"
            sequence["finished_at_utc"] = _utc_now()
            sequence["completed_phases"] = completed_phases
            sequence["error"] = f"{type(exc).__name__}: {exc}"
            _write_json_atomic(sequence_path, sequence)
            _update_orchestration_status(
                orchestration_status,
                status="failed",
                current_run_id=run_id,
                current_phase_id=(
                    _phases(runtime, row)[len(completed_phases)].phase_id
                    if len(completed_phases) < len(_phases(runtime, row))
                    else None
                ),
                completed_run_ids=completed_run_ids,
                error=sequence["error"],
            )
            print(f"[{row.row_id}:{seed}] failed: {exc}", file=sys.stderr)
            return 1
        sequence["status"] = "completed"
        sequence["finished_at_utc"] = _utc_now()
        sequence["completed_phases"] = completed_phases
        _write_json_atomic(sequence_path, sequence)
        completed_run_ids.append(run_id)
        _update_orchestration_status(
            orchestration_status,
            status="running",
            current_run_id=None,
            current_phase_id=None,
            completed_run_ids=completed_run_ids,
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    status_path = _orchestration_status_path()
    try:
        selections = _selected_runs(args)
        _update_orchestration_status(
            status_path,
            status="starting",
            run_ids=[f"{row.row_id}:{seed}" for row, seed in selections],
            started_at_utc=_utc_now(),
        )
        result = _run_body(args, orchestration_status=status_path)
    except BaseException as exc:
        _update_orchestration_status(
            status_path,
            status="failed",
            finished_at_utc=_utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    if result == 0:
        _update_orchestration_status(
            status_path,
            status="completed",
            finished_at_utc=_utc_now(),
            current_run_id=None,
            current_phase_id=None,
        )
    else:
        _update_orchestration_status(
            status_path,
            status="failed",
            finished_at_utc=_utc_now(),
            returncode=int(result),
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "list":
            return _list_rows(args.table, args.json)
        if args.mode == "dry-run":
            return _dry_run(args)
        if args.mode == "run":
            return _run(args)
        if args.mode == "detach":
            return _detach(args)
        if args.mode == "status":
            print(
                json.dumps(
                    _inspect_or_reconcile_detached_job(
                        args.job_dir, mutate=False
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.mode == "reconcile":
            print(
                json.dumps(
                    _inspect_or_reconcile_detached_job(
                        args.job_dir, mutate=True
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        parser.error(f"unknown mode: {args.mode}")
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
