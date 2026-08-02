#!/usr/bin/env python3
"""Run the sealed Stage-B paper evaluation protocol.

Each invocation evaluates exactly one checkpoint.  PIVOT checkpoints are
resolved from a completed paper-training ``sequence_manifest.json``; the
final confidence phase is selected for S3 unless ``--training-phase rank``
explicitly requests its diagnostic-only predecessor.  A historical
pure-GDINO baseline is accepted only through an explicit config/checkpoint
pair.

The final-release protocol has two evaluator processes:

* REF8 and strict2031 in one model load;
* strict1607 with ``--skip_ref`` in a second model load.

``run`` is fail-closed: the output root must not exist, every material input
is SHA-256 sealed before launch, and completion requires full per-example
records bound to the locked manifests.  ``dry-run`` performs the same source
and input checks without creating files or starting an evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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


def _artifact_repository_root() -> Path:
    outputs_entry = REPO_ROOT / "outputs"
    if outputs_entry.exists():
        resolved_outputs = outputs_entry.resolve(strict=True)
        if not resolved_outputs.is_dir():
            raise RuntimeError("execution repository outputs entry is invalid")
        candidates = [REPO_ROOT]
        if outputs_entry.is_symlink():
            target = Path(os.readlink(outputs_entry))
            if not target.is_absolute():
                target = outputs_entry.parent / target
            if target.name == "outputs":
                candidates.append(target.parent)
        candidates.append(resolved_outputs.parent)
        for root in dict.fromkeys(candidates):
            if (
                (root / "config").is_dir()
                and (root / "data").is_dir()
                and (root / "outputs").resolve(strict=True)
                == resolved_outputs
            ):
                return root.resolve(strict=True)
        raise RuntimeError(
            "execution repository outputs target has no artifact repository"
        )
    # A durable source archive has no writable artifact link.  Dependency
    # inventory can still import it; materialized execution trees always have
    # the checked outputs link before they resolve formal training artifacts.
    return REPO_ROOT


ARTIFACT_REPOSITORY_ROOT = _artifact_repository_root()
ARTIFACT_OUTPUTS_ROOT = (ARTIFACT_REPOSITORY_ROOT / "outputs").resolve(
    strict=False
)


SCHEMA = "pivot.stageb.paper_evaluation_launch/v1"
POSTFLIGHT_SCHEMA = "pivot.stageb.paper_evaluation_postflight/v1"
INPUT_REHASH_SCHEMA = "pivot.stageb.paper_evaluation_input_rehash/v1"
EVAL_SEED = 42
TRAINING_SEQUENCE_SCHEMA = "pivot.stageb.paper_ablation_run_launch/v1"
TRAINING_PHASE_SCHEMA = "pivot.stageb.paper_ablation_phase_launch/v1"
TOKEN_TRAINING_SEQUENCE_SCHEMA = "pivot.stageb.token_ablation_sequence/v1"
TOKEN_TRAINING_PHASE_SCHEMA = "pivot.stageb.token_ablation_launch/v2"
FINAL_PROFILE = "final"
SCREEN_PROFILE = "screen_validation"
MATRIX_PROFILE = "matrix_validation"
VALIDATION_PROFILES = (SCREEN_PROFILE, MATRIX_PROFILE)
EVALUATION_PROFILES = (*VALIDATION_PROFILES, FINAL_PROFILE)
FORMAL_TRAIN_BATCH_SIZE = 40
FORMAL_TRAIN_UPDATES = 1000
DEFAULT_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
DEFAULT_DATA_ROOT = Path("/media/haoyi/T9/data")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/evaluations"
EVALUATOR = REPO_ROOT / "tools/eval_text_groundingdino_refcoco_tn.py"

from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLITS,
)
from tools import stageb_evaluation_source_contracts as source_contracts  # noqa: E402

SCREEN_REF_SPLITS = (
    "refcoco_val",
    "refcocop_val",
    "refcocog_val",
)

STRICT_ROOT = (
    ARTIFACT_REPOSITORY_ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
)
STRICT_SPECS: Mapping[str, Mapping[str, Any]] = {
    "strict2031": {
        "path": STRICT_ROOT / "eval_manifest.jsonl",
        "rows": 2031,
        "sha256": "0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918",
        "source_counts": {"refcoco+_unc": 1249, "refcocog_umd": 782},
    },
    "strict1607": {
        "path": STRICT_ROOT / "semantic_stageb_union_image_disjoint_manifest.jsonl",
        "rows": 1607,
        "sha256": "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25",
        "source_counts": {"refcoco+_unc": 965, "refcocog_umd": 642},
    },
}

DATA_INPUT_RELATIVE_PATHS = (
    "canonical_classes_with_aliases.json",
    "COCO/refcoco/instances.json",
    "COCO/refcoco/refs(unc).p",
    "COCO/refcoco+/instances.json",
    "COCO/refcoco+/refs(unc).p",
    "COCO/refcocog/instances.json",
    "COCO/refcocog/refs(umd).p",
    "refcoco_text_pairs/refcoco_unc_pairs.jsonl",
    "data_proc/refcoco_text_pairs/refcoco+_unc_pairs.jsonl",
    "data_proc/refcoco_text_pairs/refcocog_google_pairs.jsonl",
    "patch_episode_prebuilt/refcocoplus_stageb_phrase_v1.jsonl",
    "patch_episode_prebuilt/refcocog_stageb_phrase_v1.jsonl",
    "patches_quality_emb/emb_index_from_quality.tsv",
)

EVAL_COMMON_CODE_ENTRIES = (
    "tools/eval_text_groundingdino_refcoco_tn.py",
    "tools/eval_refcoco_stageb.py",
    "tools/eval_stageb_tn_val.py",
    "tools/stageb_eval_records.py",
    "tools/compare_stageb_fpr95_records.py",
    "tools/stageb_screen_calibration.py",
    "tools/stageb_ref_split_contract.py",
    "tools/run_stageb_paper_evaluations.py",
)
EVAL_COMMON_CODE_INCLUDE = (
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "models/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/transformer.py",
    "models/GroundingDINO/stage_b_fixed_text_scorer.py",
    "models/GroundingDINO/stage_b_gdino_score_adapter.py",
)
EVAL_COMMON_CODE_PRUNED_EDGES = (
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/run_stageb_paper_ablation_matrices.py",
    ),
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/stageb_headline_release_contract.py",
    ),
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/run_stageb_token_ablation_matrix.py",
    ),
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/run_stageb_serial_matrix_queue.py",
    ),
    (
        "tools/run_stageb_paper_evaluations.py",
        "tools/stageb_evaluation_source_contracts.py",
    ),
)
SOURCE_PROVENANCE_ENTRIES: Mapping[str, tuple[str, ...]] = {
    source_contracts.SOURCE_FAMILY_TOKEN: (
        "tools/run_stageb_token_ablation_matrix.py",
        "tools/run_stageb_serial_matrix_queue.py",
        "tools/stageb_evaluation_source_contracts.py",
    ),
    source_contracts.SOURCE_FAMILY_PAPER: (
        "tools/run_stageb_paper_ablation_matrices.py",
        "tools/run_stageb_serial_matrix_queue.py",
        "tools/stageb_evaluation_source_contracts.py",
    ),
    source_contracts.SOURCE_FAMILY_HISTORICAL_BASELINE: (
        "tools/stageb_evaluation_source_contracts.py",
    ),
}
SOURCE_PROVENANCE_PRUNED_EDGES: Mapping[
    str, tuple[tuple[str, str], ...]
] = {
    source_contracts.SOURCE_FAMILY_TOKEN: (
        (
            "tools/run_stageb_token_ablation_matrix.py",
            "tools/run_stageb_paper_ablation_matrices.py",
        ),
    ),
    source_contracts.SOURCE_FAMILY_PAPER: (),
    source_contracts.SOURCE_FAMILY_HISTORICAL_BASELINE: (),
}


class PaperEvaluationError(RuntimeError):
    """Raised when a formal paper-evaluation contract cannot be proven."""


@dataclass(frozen=True)
class Runtime:
    python: Path
    data_root: Path
    device: str
    batch_size: int
    num_workers: int
    amp: bool
    log_every: int


@dataclass(frozen=True)
class EvaluationSource:
    kind: str
    evaluation_id: str
    config: Path
    checkpoint: Path
    checkpoint_sha256: str
    training_run_id: str | None = None
    training_seed: int | None = None
    training_run_root: Path | None = None
    sequence_manifest: Path | None = None
    training_phase: str = "final"
    diagnostic_only: bool = False
    final_phase_id: str | None = None
    final_phase_manifest: Path | None = None
    training_postflight: Path | None = None
    selected_phase_id: str | None = None
    selected_phase_manifest: Path | None = None
    selected_training_postflight: Path | None = None
    training_queue_manifest: Path | None = None
    training_queue_detached_launch: Path | None = None
    training_queue_detached_status: Path | None = None
    training_queue_id: str | None = None
    training_queue_plan_sha256: str | None = None
    artifact_repository_root: Path | None = None
    training_data: tuple[Path, ...] = ()
    formal_contract_id: str | None = None
    matrix_validation_only: bool = False


@dataclass(frozen=True)
class TrainingQueueAttestation:
    manifest: Path
    detached_launch: Path
    detached_status: Path
    queue_id: str
    plan_sha256: str
    repository_root: Path
    artifact_outputs_root: Path


class HashCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}

    def digest(self, path: Path) -> str:
        path = path.resolve(strict=True)
        stat = path.stat()
        key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._cache[key] = value
        return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperEvaluationError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PaperEvaluationError(f"{label} must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve_file(
    value: str | Path, *, base: Path = ARTIFACT_REPOSITORY_ROOT
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise PaperEvaluationError(f"expected a file: {resolved}")
    return resolved


def _resolve_directory(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise PaperEvaluationError(f"expected a directory: {path}")
    return path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _checkpoint_run_id(checkpoint: Path) -> str:
    parent = checkpoint.parent.name
    stem = checkpoint.stem
    return _safe_name(f"{parent}_{stem}") if parent else _safe_name(stem)


def _file_record(
    path: Path,
    cache: HashCache,
    *,
    roles: Iterable[str] = (),
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise PaperEvaluationError(f"input is not a regular file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "roles": sorted(set(str(role) for role in roles)),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": cache.digest(path),
    }


def _verify_declared_file(
    record: Mapping[str, Any],
    *,
    label: str,
    cache: HashCache,
    require_hash: bool = True,
) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PaperEvaluationError(f"{label} has no path")
    path = Path(raw_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise PaperEvaluationError(f"{label} is not a file: {path}")
    stat = path.stat()
    for key, observed in (
        ("size_bytes", int(stat.st_size)),
        ("mtime_ns", int(stat.st_mtime_ns)),
    ):
        try:
            expected = int(record.get(key))
        except (TypeError, ValueError) as exc:
            raise PaperEvaluationError(f"{label} has invalid {key}") from exc
        if expected != observed:
            raise PaperEvaluationError(
                f"{label} {key} mismatch: expected {expected}, found {observed}"
            )
    expected_sha = str(record.get("sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise PaperEvaluationError(f"{label} has invalid sha256")
    if require_hash:
        observed_sha = cache.digest(path)
        if observed_sha != expected_sha:
            raise PaperEvaluationError(
                f"{label} SHA-256 mismatch: expected {expected_sha}, found {observed_sha}"
            )
    return path


def _resolve_manifest_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PaperEvaluationError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ARTIFACT_REPOSITORY_ROOT / path
    return path.resolve(strict=True)


def _attested_artifact_repository_root(
    plan: Mapping[str, Any], *, label: str
) -> tuple[Path, Path]:
    raw_root = plan.get("repository_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise PaperEvaluationError(f"{label} repository root is invalid")
    try:
        root = Path(raw_root).expanduser().resolve(strict=True)
        outputs = (root / "outputs").resolve(strict=True)
    except OSError as exc:
        raise PaperEvaluationError(f"{label} repository root mismatch") from exc
    if not root.is_dir() or not outputs.is_dir() or outputs != ARTIFACT_OUTPUTS_ROOT:
        raise PaperEvaluationError(f"{label} repository root mismatch")
    return root, outputs


def _expected_phase_ids(row_id: str) -> tuple[str, ...]:
    contract = source_contracts.formal_paper_run_contract(row_id)
    if contract is not None:
        return contract.phase_ids
    return (
        ("isolation_probe", "rank", "confidence")
        if row_id == "S3"
        else ("joint",)
    )


def _validate_paper_formal_run_contract(
    *,
    sequence: Mapping[str, Any],
    run_root: Path,
    row: Mapping[str, Any],
    row_id: str,
    seed: int,
) -> source_contracts.FormalPaperRunContract | None:
    formal_contract = source_contracts.formal_paper_run_contract(row_id)
    if formal_contract is not None:
        expected_fields = formal_contract.expected_row()
        if any(row.get(key) != value for key, value in expected_fields.items()):
            raise PaperEvaluationError(
                f"{row_id} differs from its fixed S2F architecture/objective contract"
            )
        if seed not in formal_contract.seeds or sequence.get(
            "training_seeds_contract"
        ) != list(formal_contract.seeds):
            raise PaperEvaluationError(f"{row_id} training seed contract mismatch")
        expected_root = formal_contract.canonical_training_root(seed)
        if run_root.resolve() != expected_root:
            raise PaperEvaluationError(
                f"{row_id} training root is not canonical: expected {expected_root}"
            )
        if sequence.get("equal_budget_contract") != formal_contract.expected_budget():
            raise PaperEvaluationError(
                f"{row_id} is not the sealed B40/U23532 successful-update "
                "batch-slot-matched main run"
            )
        return formal_contract

    from tools import run_stageb_paper_ablation_matrices as launcher

    canonical = launcher.ROW_BY_ID.get(row_id)
    if canonical is None or dict(row) != asdict(canonical):
        raise PaperEvaluationError("paper training row is not the canonical matrix row")
    if seed not in launcher.SEEDS or sequence.get("training_seeds_contract") != list(
        launcher.SEEDS
    ):
        raise PaperEvaluationError("paper training seed contract mismatch")
    expected_root_base = (
        launcher.DEFAULT_TN_OUTPUT_ROOT
        if canonical.table == "B"
        else launcher.DEFAULT_SCORE_OUTPUT_ROOT
    )
    expected_root = (expected_root_base / row_id / f"seed{seed}").resolve()
    if run_root.resolve() != expected_root:
        raise PaperEvaluationError(
            f"paper training root is not canonical: expected {expected_root}"
        )
    expected_contributing = (
        {"rank": 500, "confidence": 500}
        if row_id == "S3"
        else {"joint": FORMAL_TRAIN_UPDATES}
    )
    expected_budget = {
        "batch_size": FORMAL_TRAIN_BATCH_SIZE,
        "optimizer_updates": FORMAL_TRAIN_UPDATES,
        "s3_probe_updates_excluded": 1 if row_id == "S3" else 0,
        "contributing_phase_updates": expected_contributing,
    }
    if sequence.get("equal_budget_contract") != expected_budget:
        raise PaperEvaluationError(
            "paper training is not the sealed batch-40/1000-update formal run"
        )
    return None


def _token_queue_attestation(
    queue_dir: Path,
    *,
    run_root: Path,
    row_id: str,
    seed: int,
) -> TrainingQueueAttestation:
    from tools import run_stageb_serial_matrix_queue as queue_runner

    try:
        queue_dir = queue_dir.expanduser().resolve(strict=True)
        queue = queue_runner.load_queue(queue_dir)
        verification = queue_runner.verify_queue(queue_dir)
    except (
        queue_runner.QueueContractError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        raise PaperEvaluationError(
            f"training queue attestation failed: {exc}"
        ) from exc

    if queue.get("status") != "completed" or verification.get("status") != "passed":
        raise PaperEvaluationError("training queue is not completed and verified")
    plan = queue.get("plan")
    if not isinstance(plan, Mapping):
        raise PaperEvaluationError("training queue has no immutable plan")
    queue_id = plan.get("queue_id")
    plan_sha256 = queue.get("plan_sha256")
    if (
        not isinstance(queue_id, str)
        or not queue_id
        or verification.get("queue_id") != queue_id
        or not isinstance(plan_sha256, str)
        or verification.get("plan_sha256") != plan_sha256
    ):
        raise PaperEvaluationError("training queue identity or plan SHA-256 mismatch")
    repository_root, artifact_outputs_root = _attested_artifact_repository_root(
        plan, label="training queue"
    )

    environment = plan.get("runtime_environment")
    raw_output_root = (
        environment.get("PIVOT_TOKEN_OUTPUT_ROOT")
        if isinstance(environment, Mapping)
        else None
    )
    if not isinstance(raw_output_root, str) or not raw_output_root.strip():
        raise PaperEvaluationError(
            "training queue does not bind PIVOT_TOKEN_OUTPUT_ROOT"
        )
    output_root = Path(raw_output_root).expanduser()
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    expected_run_root = (output_root / row_id / f"seed{seed}").resolve(
        strict=False
    )
    if run_root.resolve() != expected_run_root:
        raise PaperEvaluationError(
            "token training root differs from the verified queue output root"
        )

    run_id = f"{row_id}:{seed}"
    planned_matches = [
        item
        for item in plan.get("items", [])
        if isinstance(item, Mapping)
        and item.get("run_id") == run_id
        and item.get("runner") == "token"
    ]
    item_matches = [
        item
        for item in queue.get("items", [])
        if isinstance(item, Mapping)
        and item.get("run_id") == run_id
        and item.get("runner") == "token"
    ]
    verified_matches = [
        item
        for item in verification.get("verified_items", [])
        if isinstance(item, Mapping)
        and item.get("run_id") == run_id
        and item.get("runner") == "token"
    ]
    if not (
        len(planned_matches) == len(item_matches) == len(verified_matches) == 1
    ):
        raise PaperEvaluationError(
            "training queue does not uniquely attest the token run"
        )
    item = item_matches[0]
    verified = verified_matches[0]
    if item.get("status") != "completed":
        raise PaperEvaluationError("training queue item is not completed")
    if Path(str(item.get("output_root", ""))).resolve(strict=False) != run_root:
        raise PaperEvaluationError("training queue item output root mismatch")
    if Path(str(verified.get("output_root", ""))).resolve(strict=True) != run_root:
        raise PaperEvaluationError("verified queue output root mismatch")

    job_dir = Path(str(verified.get("job_dir", ""))).resolve(strict=True)
    manifest = (queue_dir / "queue.json").resolve(strict=True)
    detached_launch = (job_dir / "launch.json").resolve(strict=True)
    detached_status = (job_dir / "status.json").resolve(strict=True)
    return TrainingQueueAttestation(
        manifest=manifest,
        detached_launch=detached_launch,
        detached_status=detached_status,
        queue_id=queue_id,
        plan_sha256=plan_sha256,
        repository_root=repository_root,
        artifact_outputs_root=artifact_outputs_root,
    )


def _paper_queue_attestation(
    queue_dir: Path,
    *,
    run_root: Path,
    row_id: str,
    seed: int,
) -> TrainingQueueAttestation:
    """Bind one paper-matrix run to a completed verified serial queue."""

    from tools import run_stageb_serial_matrix_queue as queue_runner

    try:
        queue_dir = queue_dir.expanduser().resolve(strict=True)
        queue = queue_runner.load_queue(queue_dir)
        verification = queue_runner.verify_queue(queue_dir)
    except (
        queue_runner.QueueContractError,
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        raise PaperEvaluationError(
            f"paper training queue attestation failed: {exc}"
        ) from exc
    plan = queue.get("plan")
    if (
        queue.get("status") != "completed"
        or verification.get("status") != "passed"
        or not isinstance(plan, Mapping)
    ):
        raise PaperEvaluationError(
            "paper training queue is not completed and verified"
        )
    queue_id = plan.get("queue_id")
    plan_sha256 = queue.get("plan_sha256")
    if (
        not isinstance(queue_id, str)
        or not queue_id
        or verification.get("queue_id") != queue_id
        or not isinstance(plan_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None
        or verification.get("plan_sha256") != plan_sha256
    ):
        raise PaperEvaluationError("paper training queue identity/plan drifted")
    repository_root, artifact_outputs_root = _attested_artifact_repository_root(
        plan, label="paper training queue"
    )
    run_id = f"{row_id}:{seed}"
    selectors = (
        (plan.get("items"), "planned"),
        (queue.get("items"), "persisted"),
        (verification.get("verified_items"), "verified"),
    )
    matches: dict[str, Mapping[str, Any]] = {}
    for values, label in selectors:
        selected = [
            value
            for value in values or []
            if isinstance(value, Mapping)
            and value.get("run_id") == run_id
            and value.get("runner") == "paper"
        ]
        if len(selected) != 1:
            raise PaperEvaluationError(
                f"paper training queue does not uniquely attest {run_id} in {label}"
            )
        matches[label] = selected[0]
    if matches["persisted"].get("status") != "completed":
        raise PaperEvaluationError("paper training queue item is not completed")
    for label in ("persisted", "verified"):
        try:
            output = Path(str(matches[label].get("output_root", ""))).resolve(
                strict=True
            )
        except (FileNotFoundError, OSError) as exc:
            raise PaperEvaluationError(
                f"paper queue {label} output root is invalid"
            ) from exc
        if output != run_root.resolve(strict=True):
            raise PaperEvaluationError(
                f"paper queue {label} output root differs from formal run"
            )
    job_dir = Path(str(matches["verified"].get("job_dir", ""))).resolve(
        strict=True
    )
    return TrainingQueueAttestation(
        manifest=(queue_dir / "queue.json").resolve(strict=True),
        detached_launch=(job_dir / "launch.json").resolve(strict=True),
        detached_status=(job_dir / "status.json").resolve(strict=True),
        queue_id=queue_id,
        plan_sha256=plan_sha256,
        repository_root=repository_root,
        artifact_outputs_root=artifact_outputs_root,
    )


def _validate_token_formal_run_contract(
    *,
    sequence: Mapping[str, Any],
    run_root: Path,
    row: Mapping[str, Any],
    row_id: str,
    seed: int,
    training_queue_dir: Path | None = None,
) -> TrainingQueueAttestation | None:
    from tools import run_stageb_token_ablation_matrix as launcher

    canonical = launcher.ROW_BY_ID.get(row_id)
    if canonical is None or dict(row) != asdict(canonical):
        raise PaperEvaluationError("token training row is not the canonical matrix row")
    if seed not in launcher.SEEDS or sequence.get("training_seeds_contract") != list(
        launcher.SEEDS
    ):
        raise PaperEvaluationError("token training seed contract mismatch")
    expected_root = (
        launcher.DEFAULT_OUTPUT_ROOT / row_id / f"seed{seed}"
    ).resolve()
    if run_root.resolve() != expected_root and training_queue_dir is None:
        raise PaperEvaluationError(
            "token training root is not canonical and no verified "
            f"--training-queue-dir was provided: expected {expected_root}"
        )
    expected_budget = {
        "batch_size": FORMAL_TRAIN_BATCH_SIZE,
        "optimizer_updates": FORMAL_TRAIN_UPDATES,
        "contributing_phase_updates": {"joint": FORMAL_TRAIN_UPDATES},
    }
    if sequence.get("equal_budget_contract") != expected_budget:
        raise PaperEvaluationError(
            "token training is not the sealed batch-40/1000-update formal run"
        )
    if training_queue_dir is None:
        return None
    return _token_queue_attestation(
        training_queue_dir,
        run_root=run_root,
        row_id=row_id,
        seed=seed,
    )


def _record_roles(record: Mapping[str, Any]) -> set[str]:
    roles: set[str] = set()
    role = record.get("role")
    if isinstance(role, str) and role:
        roles.add(role)
    raw_roles = record.get("roles")
    if isinstance(raw_roles, list):
        roles.update(str(value) for value in raw_roles if str(value))
    return roles


def _verified_training_data(
    records: Iterable[Mapping[str, Any]],
    *,
    cache: HashCache,
) -> tuple[Path, ...]:
    selected: list[Path] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        if not _record_roles(record).intersection(
            {"dataset_manifest", "dataset_source", "training_data"}
        ):
            continue
        selected.append(
            _verify_declared_file(
                record,
                label=f"training data input {index}",
                cache=cache,
            )
        )
    return tuple(dict.fromkeys(selected))


def _resolve_paper_source(
    run_root: Path,
    cache: HashCache,
    *,
    training_phase: str = "final",
    training_queue_dir: Path | None = None,
    allow_nonformal_fixture: bool = False,
) -> EvaluationSource:
    if training_phase not in {"final", "rank"}:
        raise PaperEvaluationError(
            f"unsupported paper training phase: {training_phase!r}"
        )
    if training_phase == "rank":
        final_source = _resolve_paper_source(
            run_root,
            cache,
            training_phase="final",
            training_queue_dir=training_queue_dir,
            allow_nonformal_fixture=allow_nonformal_fixture,
        )
        return _resolve_s3_rank_diagnostic_source(
            final_source,
            cache,
            allow_nonformal_fixture=allow_nonformal_fixture,
        )

    run_root = _resolve_directory(run_root)
    sequence_path = (run_root / "sequence_manifest.json").resolve(strict=True)
    sequence = _read_json(sequence_path, label="training sequence manifest")
    if sequence.get("schema") != TRAINING_SEQUENCE_SCHEMA:
        raise PaperEvaluationError("paper training sequence manifest schema mismatch")
    if sequence.get("status") != "completed":
        raise PaperEvaluationError("training sequence is not completed")
    if _resolve_manifest_path(sequence.get("output_dir"), label="sequence output_dir") != run_root:
        raise PaperEvaluationError("training sequence output_dir does not equal the run root")

    run_id = sequence.get("run_id")
    row = sequence.get("row")
    if not isinstance(run_id, str) or not run_id or not isinstance(row, Mapping):
        raise PaperEvaluationError("training sequence has no run_id/row contract")
    row_id = str(row.get("row_id", ""))
    if not row_id or not run_id.startswith(f"{row_id}:"):
        raise PaperEvaluationError("training sequence row_id/run_id mismatch")
    try:
        run_seed = int(sequence.get("seed"))
        run_id_seed = int(run_id.rsplit(":", 1)[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise PaperEvaluationError("training sequence seed is invalid") from exc
    if run_seed != run_id_seed:
        raise PaperEvaluationError("training sequence seed/run_id mismatch")
    formal_contract = source_contracts.formal_paper_run_contract(row_id)
    if not allow_nonformal_fixture:
        validated_contract = _validate_paper_formal_run_contract(
            sequence=sequence,
            run_root=run_root,
            row=row,
            row_id=row_id,
            seed=run_seed,
        )
        if validated_contract is not formal_contract:
            raise PaperEvaluationError("paper formal contract registry drifted")
        if formal_contract is not None and training_queue_dir is None:
            raise PaperEvaluationError(
                f"{formal_contract.id} requires its completed exact training queue"
            )
    queue_attestation = (
        _paper_queue_attestation(
            training_queue_dir,
            run_root=run_root,
            row_id=row_id,
            seed=run_seed,
        )
        if training_queue_dir is not None
        else None
    )
    artifact_repository_root = (
        queue_attestation.repository_root
        if queue_attestation is not None
        else ARTIFACT_REPOSITORY_ROOT
    )

    expected_ids = _expected_phase_ids(row_id)
    planned = sequence.get("phases")
    completed = sequence.get("completed_phases")
    if not isinstance(planned, list) or not isinstance(completed, list):
        raise PaperEvaluationError("training sequence phase lists are missing")
    planned_ids = tuple(
        str(value.get("phase", {}).get("phase_id", ""))
        if isinstance(value, Mapping) and isinstance(value.get("phase"), Mapping)
        else ""
        for value in planned
    )
    completed_ids = tuple(
        str(value.get("phase_id", "")) if isinstance(value, Mapping) else ""
        for value in completed
    )
    if planned_ids != expected_ids or completed_ids != expected_ids:
        raise PaperEvaluationError(
            "training sequence phase order mismatch: "
            f"expected {expected_ids}, planned {planned_ids}, completed {completed_ids}"
        )
    for phase_id, entry in zip(expected_ids, completed):
        if not isinstance(entry, Mapping) or entry.get("status") != "completed":
            raise PaperEvaluationError(f"training phase {phase_id} is not completed")
        expected_output = run_root if row_id != "S3" else run_root / phase_id
        if _resolve_manifest_path(
            entry.get("output_dir"), label=f"completed phase {phase_id} output_dir"
        ) != expected_output.resolve(strict=True):
            raise PaperEvaluationError(f"training phase {phase_id} output_dir mismatch")

    final_phase_id = expected_ids[-1]
    final_entry = completed[-1]
    final_output = run_root if row_id != "S3" else run_root / final_phase_id
    expected_checkpoint = (final_output / "checkpoint_iter.pth").resolve(strict=True)
    checkpoint_record = final_entry.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise PaperEvaluationError("final training phase has no checkpoint record")
    declared_checkpoint = _verify_declared_file(
        checkpoint_record,
        label="final training checkpoint",
        cache=cache,
    )
    if declared_checkpoint != expected_checkpoint:
        raise PaperEvaluationError("final training checkpoint path is not canonical")

    phase_manifest_path = (final_output / "launch_manifest.json").resolve(strict=True)
    phase_manifest = _read_json(phase_manifest_path, label="final phase launch manifest")
    if phase_manifest.get("schema") != TRAINING_PHASE_SCHEMA:
        raise PaperEvaluationError("final phase launch manifest schema mismatch")
    if phase_manifest.get("status") != "completed" or phase_manifest.get("returncode") != 0:
        raise PaperEvaluationError("final phase launch did not complete successfully")
    if phase_manifest.get("run_id") != run_id:
        raise PaperEvaluationError("final phase launch run_id mismatch")
    phase = phase_manifest.get("phase")
    if not isinstance(phase, Mapping) or phase.get("phase_id") != final_phase_id:
        raise PaperEvaluationError("final phase launch phase_id mismatch")
    if _resolve_manifest_path(
        phase_manifest.get("output_dir"), label="final phase launch output_dir"
    ) != final_output.resolve(strict=True):
        raise PaperEvaluationError("final phase launch output_dir mismatch")

    planned_final = planned[-1]
    planned_phase = planned_final.get("phase") if isinstance(planned_final, Mapping) else None
    if not isinstance(planned_phase, Mapping) or planned_phase.get("config") != phase.get("config"):
        raise PaperEvaluationError("planned and completed final-phase config differ")
    config = _resolve_file(
        str(phase.get("config", "")), base=artifact_repository_root
    )

    inputs = phase_manifest.get("inputs")
    input_records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(input_records, list):
        raise PaperEvaluationError("final phase launch has no input records")
    config_records = [
        record
        for record in input_records
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).expanduser().resolve(strict=False) == config
        and record.get("role") == "config_dependency"
    ]
    if len(config_records) != 1:
        raise PaperEvaluationError("final training config is not uniquely SHA-bound")
    _verify_declared_file(
        config_records[0], label="final training config", cache=cache
    )
    training_data = _verified_training_data(input_records, cache=cache)

    postflight = phase_manifest.get("postflight")
    if not isinstance(postflight, Mapping) or postflight.get("status") != "passed":
        raise PaperEvaluationError("final training phase has no passed postflight")
    postflight_artifact = phase_manifest.get("postflight_artifact")
    if not isinstance(postflight_artifact, Mapping):
        raise PaperEvaluationError("final training postflight artifact is missing")
    postflight_path = _verify_declared_file(
        postflight_artifact,
        label="final training postflight",
        cache=cache,
    )
    if postflight_path != (final_output / "postflight.json").resolve(strict=True):
        raise PaperEvaluationError("final training postflight path is not canonical")
    postflight_payload = _read_json(postflight_path, label="final training postflight")
    if postflight_payload != postflight:
        raise PaperEvaluationError("embedded and persisted training postflight differ")
    artifacts = postflight.get("artifacts")
    postflight_checkpoint = artifacts.get("checkpoint") if isinstance(artifacts, Mapping) else None
    if not isinstance(postflight_checkpoint, Mapping) or dict(postflight_checkpoint) != dict(checkpoint_record):
        raise PaperEvaluationError("sequence and postflight checkpoint records differ")
    metadata = postflight.get("checkpoint_metadata")
    args = metadata.get("args") if isinstance(metadata, Mapping) else None
    if not isinstance(args, Mapping):
        raise PaperEvaluationError("training checkpoint metadata is missing")
    if _resolve_manifest_path(args.get("config_file"), label="checkpoint config_file") != config:
        raise PaperEvaluationError("checkpoint metadata config does not match final phase")
    if _resolve_manifest_path(args.get("output_dir"), label="checkpoint output_dir") != final_output.resolve(strict=True):
        raise PaperEvaluationError("checkpoint metadata output_dir mismatch")
    expected_final_updates = (
        formal_contract.final_phase_updates
        if formal_contract is not None
        else (500 if row_id == "S3" else FORMAL_TRAIN_UPDATES)
    )
    expected_checkpoint_interval = (
        formal_contract.iter_checkpoint_interval
        if formal_contract is not None
        else expected_final_updates
    )
    expected_batch_size = (
        formal_contract.batch_size
        if formal_contract is not None
        else FORMAL_TRAIN_BATCH_SIZE
    )
    if not allow_nonformal_fixture:
        expected_checkpoint_scalars = {
            "seed": run_seed,
            "batch_size": expected_batch_size,
            "max_train_iters": expected_final_updates,
            "iter_checkpoint_interval": expected_checkpoint_interval,
        }
        if any(args.get(key) != value for key, value in expected_checkpoint_scalars.items()):
            raise PaperEvaluationError(
                "paper checkpoint metadata is not the formal phase budget"
            )
    input_rehash = postflight.get("input_rehash")
    if not isinstance(input_rehash, Mapping) or input_rehash.get("status") != "passed":
        raise PaperEvaluationError("final training input rehash did not pass")

    return EvaluationSource(
        kind="pivot_paper_training_run",
        evaluation_id=run_id.replace(":", "_seed"),
        config=config,
        checkpoint=declared_checkpoint,
        checkpoint_sha256=str(checkpoint_record["sha256"]),
        training_run_id=run_id,
        training_seed=run_seed,
        training_run_root=run_root,
        sequence_manifest=sequence_path,
        training_phase="final",
        diagnostic_only=False,
        final_phase_id=final_phase_id,
        final_phase_manifest=phase_manifest_path,
        training_postflight=postflight_path,
        selected_phase_id=final_phase_id,
        selected_phase_manifest=phase_manifest_path,
        selected_training_postflight=postflight_path,
        training_queue_manifest=(
            queue_attestation.manifest if queue_attestation else None
        ),
        training_queue_detached_launch=(
            queue_attestation.detached_launch if queue_attestation else None
        ),
        training_queue_detached_status=(
            queue_attestation.detached_status if queue_attestation else None
        ),
        training_queue_id=(
            queue_attestation.queue_id if queue_attestation else None
        ),
        training_queue_plan_sha256=(
            queue_attestation.plan_sha256 if queue_attestation else None
        ),
        artifact_repository_root=artifact_repository_root,
        training_data=training_data,
        formal_contract_id=(
            formal_contract.id if formal_contract is not None else None
        ),
        matrix_validation_only=(
            formal_contract.matrix_validation_only
            if formal_contract is not None
            else False
        ),
    )


def _resolve_s3_rank_diagnostic_source(
    final_source: EvaluationSource,
    cache: HashCache,
    *,
    allow_nonformal_fixture: bool,
) -> EvaluationSource:
    """Resolve S3's completed rank phase without presenting it as the final row."""
    run_root = final_source.training_run_root
    sequence_path = final_source.sequence_manifest
    if run_root is None or sequence_path is None:
        raise PaperEvaluationError("rank diagnostics require a PIVOT training run")
    sequence = _read_json(sequence_path, label="S3 training sequence manifest")
    row = sequence.get("row")
    if (
        not isinstance(row, Mapping)
        or row.get("row_id") != "S3"
        or final_source.training_run_id != f"S3:{final_source.training_seed}"
        or final_source.final_phase_id != "confidence"
    ):
        raise PaperEvaluationError(
            "--training-phase rank is restricted to a completed formal S3 run"
        )
    planned = sequence.get("phases")
    completed = sequence.get("completed_phases")
    if not isinstance(planned, list) or not isinstance(completed, list):
        raise PaperEvaluationError("S3 rank phase lists are missing")
    rank_index = _expected_phase_ids("S3").index("rank")
    planned_rank = planned[rank_index]
    completed_rank = completed[rank_index]
    if not isinstance(planned_rank, Mapping) or not isinstance(
        completed_rank, Mapping
    ):
        raise PaperEvaluationError("S3 rank phase entries are invalid")
    if completed_rank.get("status") != "completed":
        raise PaperEvaluationError("S3 rank phase is not completed")

    rank_output = (run_root / "rank").resolve(strict=True)
    if _resolve_manifest_path(
        completed_rank.get("output_dir"), label="S3 rank completed output_dir"
    ) != rank_output:
        raise PaperEvaluationError("S3 rank phase output_dir mismatch")
    checkpoint_record = completed_rank.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise PaperEvaluationError("S3 rank phase has no checkpoint record")
    checkpoint = _verify_declared_file(
        checkpoint_record,
        label="S3 rank checkpoint",
        cache=cache,
    )
    if checkpoint != (rank_output / "checkpoint_iter.pth").resolve(strict=True):
        raise PaperEvaluationError("S3 rank checkpoint path is not canonical")

    launch_path = (rank_output / "launch_manifest.json").resolve(strict=True)
    launch = _read_json(launch_path, label="S3 rank phase launch manifest")
    if launch.get("schema") != TRAINING_PHASE_SCHEMA:
        raise PaperEvaluationError("S3 rank phase launch schema mismatch")
    if launch.get("status") != "completed" or launch.get("returncode") != 0:
        raise PaperEvaluationError("S3 rank phase launch did not complete")
    if launch.get("run_id") != final_source.training_run_id:
        raise PaperEvaluationError("S3 rank phase run_id mismatch")
    phase = launch.get("phase")
    planned_phase = planned_rank.get("phase")
    if (
        not isinstance(phase, Mapping)
        or phase.get("phase_id") != "rank"
        or not isinstance(planned_phase, Mapping)
        or planned_phase.get("phase_id") != "rank"
        or planned_phase.get("config") != phase.get("config")
    ):
        raise PaperEvaluationError("S3 planned and completed rank phases differ")
    if _resolve_manifest_path(
        launch.get("output_dir"), label="S3 rank launch output_dir"
    ) != rank_output:
        raise PaperEvaluationError("S3 rank launch output_dir mismatch")
    config = _resolve_file(
        str(phase.get("config", "")),
        base=(
            final_source.artifact_repository_root
            or ARTIFACT_REPOSITORY_ROOT
        ),
    )

    inputs = launch.get("inputs")
    input_records = inputs.get("records") if isinstance(inputs, Mapping) else None
    if not isinstance(input_records, list):
        raise PaperEvaluationError("S3 rank launch has no input records")
    config_records = [
        record
        for record in input_records
        if isinstance(record, Mapping)
        and Path(str(record.get("path", ""))).expanduser().resolve(strict=False)
        == config
        and record.get("role") == "config_dependency"
    ]
    if len(config_records) != 1:
        raise PaperEvaluationError("S3 rank config is not uniquely SHA-bound")
    _verify_declared_file(config_records[0], label="S3 rank config", cache=cache)
    training_data = _verified_training_data(input_records, cache=cache)

    postflight = launch.get("postflight")
    if not isinstance(postflight, Mapping) or postflight.get("status") != "passed":
        raise PaperEvaluationError("S3 rank phase has no passed postflight")
    postflight_artifact = launch.get("postflight_artifact")
    if not isinstance(postflight_artifact, Mapping):
        raise PaperEvaluationError("S3 rank postflight artifact is missing")
    postflight_path = _verify_declared_file(
        postflight_artifact,
        label="S3 rank postflight",
        cache=cache,
    )
    if postflight_path != (rank_output / "postflight.json").resolve(strict=True):
        raise PaperEvaluationError("S3 rank postflight path is not canonical")
    persisted = _read_json(postflight_path, label="S3 rank postflight")
    if persisted != postflight:
        raise PaperEvaluationError("embedded and persisted S3 rank postflight differ")
    artifacts = postflight.get("artifacts")
    postflight_checkpoint = (
        artifacts.get("checkpoint") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(postflight_checkpoint, Mapping) or dict(
        postflight_checkpoint
    ) != dict(checkpoint_record):
        raise PaperEvaluationError("S3 rank checkpoint records differ")
    metadata = postflight.get("checkpoint_metadata")
    args = metadata.get("args") if isinstance(metadata, Mapping) else None
    if not isinstance(args, Mapping):
        raise PaperEvaluationError("S3 rank checkpoint metadata is missing")
    if _resolve_manifest_path(
        args.get("config_file"), label="S3 rank checkpoint config_file"
    ) != config:
        raise PaperEvaluationError("S3 rank checkpoint config mismatch")
    if _resolve_manifest_path(
        args.get("output_dir"), label="S3 rank checkpoint output_dir"
    ) != rank_output:
        raise PaperEvaluationError("S3 rank checkpoint output_dir mismatch")
    if not allow_nonformal_fixture:
        expected_scalars = {
            "seed": final_source.training_seed,
            "batch_size": FORMAL_TRAIN_BATCH_SIZE,
            "max_train_iters": 500,
            "iter_checkpoint_interval": 500,
        }
        if any(args.get(key) != value for key, value in expected_scalars.items()):
            raise PaperEvaluationError(
                "S3 rank checkpoint is not the formal batch/update budget"
            )
    input_rehash = postflight.get("input_rehash")
    if not isinstance(input_rehash, Mapping) or input_rehash.get("status") != "passed":
        raise PaperEvaluationError("S3 rank input rehash did not pass")

    return EvaluationSource(
        kind="pivot_paper_training_run_rank_diagnostic",
        evaluation_id=f"{final_source.evaluation_id}_rank_diagnostic",
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=str(checkpoint_record["sha256"]),
        training_run_id=final_source.training_run_id,
        training_seed=final_source.training_seed,
        training_run_root=run_root,
        sequence_manifest=sequence_path,
        training_phase="rank",
        diagnostic_only=True,
        final_phase_id=final_source.final_phase_id,
        final_phase_manifest=final_source.final_phase_manifest,
        training_postflight=final_source.training_postflight,
        selected_phase_id="rank",
        selected_phase_manifest=launch_path,
        selected_training_postflight=postflight_path,
        training_queue_manifest=final_source.training_queue_manifest,
        training_queue_detached_launch=final_source.training_queue_detached_launch,
        training_queue_detached_status=final_source.training_queue_detached_status,
        training_queue_id=final_source.training_queue_id,
        training_queue_plan_sha256=final_source.training_queue_plan_sha256,
        artifact_repository_root=final_source.artifact_repository_root,
        training_data=training_data,
        formal_contract_id=final_source.formal_contract_id,
        matrix_validation_only=final_source.matrix_validation_only,
    )


def _token_input_records(
    inputs: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    for field, value in inputs.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                records.append((str(field), item))
    return records


def _resolve_token_source(
    run_root: Path,
    cache: HashCache,
    *,
    training_queue_dir: Path | None = None,
    allow_nonformal_fixture: bool = False,
) -> EvaluationSource:
    run_root = _resolve_directory(run_root)
    sequence_path = (run_root / "sequence_manifest.json").resolve(strict=True)
    sequence = _read_json(sequence_path, label="token training sequence manifest")
    if sequence.get("schema") != TOKEN_TRAINING_SEQUENCE_SCHEMA:
        raise PaperEvaluationError("token training sequence manifest schema mismatch")
    if sequence.get("status") != "completed":
        raise PaperEvaluationError("token training sequence is not completed")
    if _resolve_manifest_path(
        sequence.get("output_dir"), label="token sequence output_dir"
    ) != run_root:
        raise PaperEvaluationError("token training sequence output_dir mismatch")

    run_id = sequence.get("run_id")
    row = sequence.get("row")
    if not isinstance(run_id, str) or not isinstance(row, Mapping):
        raise PaperEvaluationError("token training sequence has no run_id/row")
    row_id = str(row.get("row_id", ""))
    if re.fullmatch(r"L(?:10|[0-9])", row_id) is None or not run_id.startswith(
        f"{row_id}:"
    ):
        raise PaperEvaluationError("token training row_id/run_id mismatch")
    try:
        seed = int(sequence.get("seed"))
        run_seed = int(run_id.rsplit(":", 1)[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise PaperEvaluationError("token training seed is invalid") from exc
    if seed != run_seed:
        raise PaperEvaluationError("token training seed/run_id mismatch")
    queue_attestation = None
    if not allow_nonformal_fixture:
        queue_attestation = _validate_token_formal_run_contract(
            sequence=sequence,
            run_root=run_root,
            row=row,
            row_id=row_id,
            seed=seed,
            training_queue_dir=training_queue_dir,
        )
    elif training_queue_dir is not None:
        raise PaperEvaluationError(
            "training queue attestation is unavailable for nonformal fixtures"
        )
    artifact_repository_root = (
        queue_attestation.repository_root
        if queue_attestation is not None
        else ARTIFACT_REPOSITORY_ROOT
    )

    planned = sequence.get("phases")
    completed = sequence.get("completed_phases")
    if not (
        isinstance(planned, list)
        and len(planned) == 1
        and isinstance(planned[0], Mapping)
        and planned[0].get("phase_id") == "joint"
        and planned[0].get("contributes_to_budget") is True
        and _required_int(
            planned[0].get("optimizer_updates"),
            label="token joint optimizer_updates",
        )
        > 0
        and _resolve_manifest_path(
            planned[0].get("output_dir"), label="token planned output_dir"
        )
        == run_root
    ):
        raise PaperEvaluationError("token training planned joint phase is invalid")
    if not (
        isinstance(completed, list)
        and len(completed) == 1
        and isinstance(completed[0], Mapping)
        and completed[0].get("phase_id") == "joint"
        and completed[0].get("status") == "completed"
        and _resolve_manifest_path(
            completed[0].get("output_dir"), label="token completed output_dir"
        )
        == run_root
    ):
        raise PaperEvaluationError("token training completed joint phase is invalid")

    checkpoint_record = completed[0].get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise PaperEvaluationError("token training sequence has no checkpoint record")
    checkpoint = _verify_declared_file(
        checkpoint_record,
        label="token final checkpoint",
        cache=cache,
    )
    if checkpoint != (run_root / "checkpoint_iter.pth").resolve(strict=True):
        raise PaperEvaluationError("token final checkpoint path is not canonical")

    launch_path = (run_root / "launch_manifest.json").resolve(strict=True)
    launch = _read_json(launch_path, label="token training launch manifest")
    if launch.get("schema") != TOKEN_TRAINING_PHASE_SCHEMA:
        raise PaperEvaluationError("token training launch schema mismatch")
    if launch.get("status") != "completed" or launch.get("returncode") != 0:
        raise PaperEvaluationError("token training launch did not complete")
    if launch.get("run_id") != run_id or launch.get("seed") != seed:
        raise PaperEvaluationError("token training launch run identity mismatch")
    if launch.get("row") != row:
        raise PaperEvaluationError("token sequence and launch row contracts differ")
    if _resolve_manifest_path(
        launch.get("output_dir"), label="token launch output_dir"
    ) != run_root:
        raise PaperEvaluationError("token training launch output_dir mismatch")

    config = _resolve_file(
        str(row.get("config", "")), base=artifact_repository_root
    )
    inputs = launch.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PaperEvaluationError("token training launch has no input contract")
    nested_records = _token_input_records(inputs)
    config_records = [
        record
        for field, record in nested_records
        if field == "config_dependencies"
        and Path(str(record.get("path", ""))).expanduser().resolve(strict=False)
        == config
    ]
    if len(config_records) != 1:
        raise PaperEvaluationError("token final config is not uniquely SHA-bound")
    _verify_declared_file(
        config_records[0], label="token final config", cache=cache
    )
    data_records: list[Mapping[str, Any]] = []
    for field, record in nested_records:
        if field not in {"dataset_manifest", "dataset_source_files"}:
            continue
        tagged = dict(record)
        tagged["role"] = (
            "dataset_manifest" if field == "dataset_manifest" else "dataset_source"
        )
        data_records.append(tagged)
    training_data = _verified_training_data(data_records, cache=cache)
    if not training_data:
        raise PaperEvaluationError("token training launch has no bound training data")

    postflight_artifact = launch.get("postflight_artifact")
    if not isinstance(postflight_artifact, Mapping):
        raise PaperEvaluationError("token training postflight artifact is missing")
    postflight_path = _verify_declared_file(
        postflight_artifact,
        label="token training postflight",
        cache=cache,
    )
    if postflight_path != (run_root / "postflight.json").resolve(strict=True):
        raise PaperEvaluationError("token training postflight path is not canonical")
    sequence_postflight = completed[0].get("postflight")
    if not isinstance(sequence_postflight, Mapping):
        raise PaperEvaluationError("token sequence does not bind its postflight")
    sequence_postflight_path = _verify_declared_file(
        sequence_postflight,
        label="token sequence postflight",
        cache=cache,
    )
    if sequence_postflight_path != postflight_path or dict(
        sequence_postflight
    ) != dict(postflight_artifact):
        raise PaperEvaluationError("token sequence and launch postflight differ")
    postflight = _read_json(postflight_path, label="token training postflight")
    if postflight.get("status") != "passed" or launch.get("postflight") != postflight:
        raise PaperEvaluationError("token training postflight did not pass exactly")
    artifacts = postflight.get("artifacts")
    postflight_checkpoint = (
        artifacts.get("checkpoint") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(postflight_checkpoint, Mapping) or dict(
        postflight_checkpoint
    ) != dict(checkpoint_record):
        raise PaperEvaluationError("token sequence and postflight checkpoint differ")
    metadata = postflight.get("checkpoint_metadata")
    checkpoint_args = metadata.get("args") if isinstance(metadata, Mapping) else None
    if not isinstance(checkpoint_args, Mapping):
        raise PaperEvaluationError("token checkpoint metadata is missing")
    if _resolve_manifest_path(
        checkpoint_args.get("config_file"), label="token checkpoint config_file"
    ) != config:
        raise PaperEvaluationError("token checkpoint metadata config mismatch")
    if _resolve_manifest_path(
        checkpoint_args.get("output_dir"), label="token checkpoint output_dir"
    ) != run_root:
        raise PaperEvaluationError("token checkpoint metadata output_dir mismatch")
    if not allow_nonformal_fixture:
        expected_checkpoint_scalars = {
            "seed": seed,
            "batch_size": FORMAL_TRAIN_BATCH_SIZE,
            "max_train_iters": FORMAL_TRAIN_UPDATES,
            "iter_checkpoint_interval": FORMAL_TRAIN_UPDATES,
        }
        if any(
            checkpoint_args.get(key) != value
            for key, value in expected_checkpoint_scalars.items()
        ):
            raise PaperEvaluationError(
                "token checkpoint metadata is not the formal batch/update budget"
            )
    input_rehash = postflight.get("input_rehash")
    if not isinstance(input_rehash, Mapping) or input_rehash.get("status") != "passed":
        raise PaperEvaluationError("token training input rehash did not pass")

    return EvaluationSource(
        kind="pivot_token_ablation_training_run",
        evaluation_id=run_id.replace(":", "_seed"),
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=str(checkpoint_record["sha256"]),
        training_run_id=run_id,
        training_seed=seed,
        training_run_root=run_root,
        sequence_manifest=sequence_path,
        training_phase="final",
        diagnostic_only=False,
        final_phase_id="joint",
        final_phase_manifest=launch_path,
        training_postflight=postflight_path,
        selected_phase_id="joint",
        selected_phase_manifest=launch_path,
        selected_training_postflight=postflight_path,
        training_queue_manifest=(
            queue_attestation.manifest if queue_attestation else None
        ),
        training_queue_detached_launch=(
            queue_attestation.detached_launch if queue_attestation else None
        ),
        training_queue_detached_status=(
            queue_attestation.detached_status if queue_attestation else None
        ),
        training_queue_id=(
            queue_attestation.queue_id if queue_attestation else None
        ),
        training_queue_plan_sha256=(
            queue_attestation.plan_sha256 if queue_attestation else None
        ),
        artifact_repository_root=artifact_repository_root,
        training_data=training_data,
    )


def _resolve_pivot_source(
    run_root: Path,
    cache: HashCache,
    *,
    training_phase: str = "final",
    training_queue_dir: Path | None = None,
    allow_nonformal_fixture: bool = False,
) -> EvaluationSource:
    root = _resolve_directory(run_root)
    sequence_path = (root / "sequence_manifest.json").resolve(strict=True)
    sequence = _read_json(sequence_path, label="training sequence manifest")
    schema = sequence.get("schema")
    if schema == TRAINING_SEQUENCE_SCHEMA:
        return _resolve_paper_source(
            root,
            cache,
            training_phase=training_phase,
            training_queue_dir=training_queue_dir,
            allow_nonformal_fixture=allow_nonformal_fixture,
        )
    if schema == TOKEN_TRAINING_SEQUENCE_SCHEMA:
        if training_phase != "final":
            raise PaperEvaluationError(
                "--training-phase rank is restricted to completed formal S3 runs"
            )
        return _resolve_token_source(
            root,
            cache,
            training_queue_dir=training_queue_dir,
            allow_nonformal_fixture=allow_nonformal_fixture,
        )
    raise PaperEvaluationError(
        "unsupported training sequence schema: " + repr(schema)
    )


def _resolve_baseline_source(
    config: Path,
    checkpoint: Path,
    baseline_id: str,
    cache: HashCache,
) -> EvaluationSource:
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        raise PaperEvaluationError("--baseline-id must be non-empty")
    config = _resolve_file(config)
    checkpoint = _resolve_file(checkpoint)
    return EvaluationSource(
        kind="historical_pure_gdino_explicit",
        evaluation_id=_safe_name(baseline_id),
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=cache.digest(checkpoint),
    )


def _resolve_source(args: argparse.Namespace, cache: HashCache) -> EvaluationSource:
    training_phase = getattr(args, "training_phase", "final")
    has_training = args.training_run_root is not None
    has_baseline = args.baseline_config is not None or args.baseline_checkpoint is not None
    if has_training == has_baseline:
        raise PaperEvaluationError(
            "select exactly one source: --training-run-root, or both "
            "--baseline-config and --baseline-checkpoint"
        )
    if has_training:
        if args.baseline_config is not None or args.baseline_checkpoint is not None:
            raise PaperEvaluationError("PIVOT and baseline source arguments cannot be mixed")
        return _resolve_pivot_source(
            Path(args.training_run_root),
            cache,
            training_phase=training_phase,
            training_queue_dir=(
                Path(args.training_queue_dir)
                if args.training_queue_dir is not None
                else None
            ),
        )
    if args.training_queue_dir is not None:
        raise PaperEvaluationError(
            "--training-queue-dir requires --training-run-root"
        )
    if training_phase != "final":
        raise PaperEvaluationError(
            "--training-phase rank requires --training-run-root for completed formal S3"
        )
    if args.baseline_config is None or args.baseline_checkpoint is None:
        raise PaperEvaluationError(
            "historical baseline requires both --baseline-config and --baseline-checkpoint"
        )
    return _resolve_baseline_source(
        Path(args.baseline_config),
        Path(args.baseline_checkpoint),
        args.baseline_id,
        cache,
    )


def _strict_manifest_record(
    label: str, cache: HashCache
) -> dict[str, Any]:
    specification = STRICT_SPECS[label]
    path = _resolve_file(Path(specification["path"]))
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise PaperEvaluationError(f"{label} is not UTF-8 JSONL") from exc
    rows: list[Mapping[str, Any]] = []
    counts: dict[str, int] = {}
    sample_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PaperEvaluationError(f"{label}:{line_number}: invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise PaperEvaluationError(f"{label}:{line_number}: row must be an object")
        if row.get("manifest_schema") != "stageb_vlm_verified_strict_tn_v2":
            raise PaperEvaluationError(f"{label}:{line_number}: wrong manifest schema")
        if row.get("visual_verified_negative") is not True or row.get("coverage_pass") is not True:
            raise PaperEvaluationError(f"{label}:{line_number}: row is not verified")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in sample_ids:
            raise PaperEvaluationError(f"{label}:{line_number}: missing/duplicate sample_id")
        sample_ids.add(sample_id)
        source = str(row.get("tn_eval_pair_source", ""))
        counts[source] = counts.get(source, 0) + 1
        rows.append(row)
    if len(rows) != int(specification["rows"]):
        raise PaperEvaluationError(
            f"{label} row count mismatch: expected {specification['rows']}, found {len(rows)}"
        )
    if counts != dict(specification["source_counts"]):
        raise PaperEvaluationError(f"{label} source-count contract mismatch: {counts}")
    record = _file_record(path, cache, roles=(label, "locked_tn_source_manifest"))
    if record["sha256"] != specification["sha256"]:
        raise PaperEvaluationError(
            f"{label} SHA-256 mismatch: expected {specification['sha256']}, "
            f"found {record['sha256']}"
        )
    record["rows"] = len(rows)
    record["source_counts"] = counts
    return record


def _screen_calibration_contract(cache: HashCache) -> dict[str, Any]:
    from tools.stageb_screen_calibration import (
        DEFAULT_AUDIT,
        DEFAULT_SOURCE,
        ScreenCalibrationError,
        validate_locked_source,
    )

    try:
        contract = validate_locked_source(DEFAULT_SOURCE, DEFAULT_AUDIT)
    except (OSError, ScreenCalibrationError) as exc:
        raise PaperEvaluationError(
            f"screen calibration contract failed: {exc}"
        ) from exc
    source = _file_record(
        Path(contract["source_manifest"]["path"]),
        cache,
        roles=("screen_calibration_source",),
    )
    source["rows"] = int(contract["source_manifest"]["rows"])
    audit = _file_record(
        Path(contract["source_audit"]["path"]),
        cache,
        roles=("screen_calibration_audit",),
    )
    if source["sha256"] != contract["source_manifest"]["sha256"]:
        raise PaperEvaluationError("screen calibration source changed during preflight")
    if audit["sha256"] != contract["source_audit"]["sha256"]:
        raise PaperEvaluationError("screen calibration audit changed during preflight")
    return {
        **contract,
        "source_manifest": source,
        "source_audit": audit,
    }


def evaluation_common_code_paths() -> list[Path]:
    """Return the source-family-invariant evaluator runtime closure."""

    from tools.stageb_profile_dependency_audit import (
        ProfileDependencyAuditError,
        recursive_local_python_dependencies,
    )

    try:
        return recursive_local_python_dependencies(
            EVAL_COMMON_CODE_ENTRIES,
            repository_root=REPO_ROOT,
            include_paths=EVAL_COMMON_CODE_INCLUDE,
            pruned_edges=EVAL_COMMON_CODE_PRUNED_EDGES,
        )
    except ProfileDependencyAuditError as exc:
        raise PaperEvaluationError(
            f"evaluation dependency profile failed: {exc}"
        ) from exc


def _evaluation_code_paths() -> list[Path]:
    """Compatibility alias for common-code fingerprint consumers."""

    return evaluation_common_code_paths()


def evaluation_source_family(source: EvaluationSource) -> str:
    try:
        return source_contracts.source_family_for_kind(source.kind)
    except source_contracts.EvaluationSourceContractError as exc:
        raise PaperEvaluationError(str(exc)) from exc


def evaluation_source_provenance_paths(source_family: str) -> list[Path]:
    """Return validators and launchers specific to exactly one source family."""

    from tools.stageb_profile_dependency_audit import (
        ProfileDependencyAuditError,
        recursive_local_python_dependencies,
    )

    if source_family not in source_contracts.SOURCE_FAMILIES:
        raise PaperEvaluationError(
            f"unsupported evaluation source family: {source_family!r}"
        )
    try:
        return recursive_local_python_dependencies(
            SOURCE_PROVENANCE_ENTRIES[source_family],
            repository_root=REPO_ROOT,
            pruned_edges=SOURCE_PROVENANCE_PRUNED_EDGES[source_family],
        )
    except ProfileDependencyAuditError as exc:
        raise PaperEvaluationError(
            f"source provenance dependency profile failed: {exc}"
        ) from exc


def _evaluation_source_provenance_paths(
    source: EvaluationSource | str,
) -> list[Path]:
    family = source if isinstance(source, str) else evaluation_source_family(source)
    paths = evaluation_source_provenance_paths(family)
    if isinstance(source, str) or source.formal_contract_id is None:
        return paths
    contract = source_contracts.formal_paper_run_contract(
        source.formal_contract_id
    )
    if contract is None:
        raise PaperEvaluationError(
            f"unknown formal paper contract: {source.formal_contract_id!r}"
        )
    runner = (REPO_ROOT / contract.runner).resolve(strict=True)
    if not runner.is_file():
        raise PaperEvaluationError(
            f"formal source runner is not a file: {runner}"
        )
    return sorted(set(paths).union({runner}), key=lambda path: str(path))


def _config_paths(
    config: Path, *, repository_root: Path = ARTIFACT_REPOSITORY_ROOT
) -> list[Path]:
    from tools.stageb_dependency_audit import (
        DependencyAuditError,
        local_python_dependency_paths,
    )

    config = config.resolve(strict=True)
    repository_root = repository_root.resolve(strict=True)
    try:
        config.relative_to(repository_root)
        dependency_root = repository_root
    except ValueError:
        # Historical pure-GDINO configs may live in the predecessor checkout.
        # Bind their full local import chain to that checkout, without allowing
        # the dependency walker to escape into an unrelated filesystem tree.
        dependency_root = config.parent
        for ancestor in config.parents:
            if (ancestor / "main.py").is_file() and (
                (ancestor / "config").is_dir() or (ancestor / "configs").is_dir()
            ):
                dependency_root = ancestor
                break
    try:
        paths = local_python_dependency_paths([config], root=dependency_root)
    except DependencyAuditError as exc:
        raise PaperEvaluationError(f"config dependency audit failed: {exc}") from exc
    if not paths or config.resolve() not in {path.resolve() for path in paths}:
        raise PaperEvaluationError("config dependency audit omitted the selected config")
    return paths


def _data_input_paths(data_root: Path) -> list[Path]:
    paths = []
    for relative in DATA_INPUT_RELATIVE_PATHS:
        path = (data_root / relative).resolve(strict=True)
        if not path.is_file():
            raise PaperEvaluationError(f"evaluation data input is missing: {path}")
        paths.append(path)
    return paths


def _merge_input_records(
    entries: Iterable[tuple[Path, str]], cache: HashCache
) -> list[dict[str, Any]]:
    roles: dict[Path, set[str]] = {}
    for path, role in entries:
        path = path.resolve(strict=True)
        roles.setdefault(path, set()).add(role)
    return [
        _file_record(path, cache, roles=roles[path])
        for path in sorted(roles, key=lambda value: str(value))
    ]


def _runtime_from_args(args: argparse.Namespace) -> Runtime:
    python_raw = str(args.python)
    candidate = Path(python_raw).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        python = candidate.resolve(strict=True)
    else:
        executable = shutil.which(python_raw)
        if executable is None:
            raise PaperEvaluationError(f"Python executable not found: {python_raw}")
        python = Path(executable).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PaperEvaluationError(f"Python executable is not executable: {python}")
    data_root = _resolve_directory(args.data_root)
    if int(args.batch_size) <= 0 or int(args.num_workers) < 0 or int(args.log_every) <= 0:
        raise PaperEvaluationError("batch_size/log_every must be positive and num_workers nonnegative")
    return Runtime(
        python=python,
        data_root=data_root,
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=not bool(args.no_amp),
        log_every=int(args.log_every),
    )


def _common_eval_command(runtime: Runtime, source: EvaluationSource) -> list[str]:
    command = [
        str(runtime.python),
        str(EVALUATOR.resolve(strict=True)),
        "--config",
        str(source.config),
        "--ckpts",
        str(source.checkpoint),
        "--data_root",
        str(runtime.data_root),
        "--device",
        runtime.device,
        "--batch_size",
        str(runtime.batch_size),
        "--num_workers",
        str(runtime.num_workers),
        "--seed",
        str(EVAL_SEED),
        "--topk",
        "1",
        "--threshold_tprs",
        "0.75",
        "0.9",
        "0.95",
        "--score_thresholds",
        "0.5",
        "--max_ref_batches",
        "0",
        "--max_tn_batches",
        "0",
        "--log_every",
        str(runtime.log_every),
    ]
    if runtime.amp:
        command.append("--amp")
    return command


def _commands(
    runtime: Runtime,
    source: EvaluationSource,
    output_dir: Path,
    *,
    profile: str = FINAL_PROFILE,
    screen_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    common = _common_eval_command(runtime, source)
    if profile in VALIDATION_PROFILES:
        if screen_contract is None:
            screen_contract = _screen_calibration_contract(HashCache())
        source_record = screen_contract.get("source_manifest")
        audit_record = screen_contract.get("source_audit")
        if not isinstance(source_record, Mapping) or not isinstance(
            audit_record, Mapping
        ):
            raise PaperEvaluationError(
                "screen calibration command contract is incomplete"
            )

        screen = [
            *common,
            "--output_dir",
            str(output_dir / "validation_calibration"),
            "--ref_splits",
            *SCREEN_REF_SPLITS,
            "--tn_jsonl",
            str(Path(str(source_record.get("path", ""))).resolve(strict=True)),
            "--screen_calibration_manifest",
            "--screen_calibration_audit",
            str(Path(str(audit_record.get("path", ""))).resolve(strict=True)),
        ]
        return [
            {
                "phase_id": "validation_calibration",
                "command": screen,
                "command_shell": shlex.join(screen),
                "console_log": str(
                    output_dir / "validation_calibration_console.log"
                ),
            }
        ]
    if profile != FINAL_PROFILE:
        raise PaperEvaluationError(f"unsupported evaluation profile: {profile!r}")
    primary = [
        *common,
        "--output_dir",
        str(output_dir / "ref8_strict2031"),
        "--ref_splits",
        *REF_SPLITS,
        "--tn_jsonl",
        str(Path(STRICT_SPECS["strict2031"]["path"]).resolve(strict=True)),
        "--tn_splits",
        "refcocop_val",
        "refcocog_umd_val",
    ]
    supplemental = [
        *common,
        "--output_dir",
        str(output_dir / "strict1607"),
        "--skip_ref",
        "--tn_jsonl",
        str(Path(STRICT_SPECS["strict1607"]["path"]).resolve(strict=True)),
        "--tn_splits",
        "refcocop_val",
        "refcocog_umd_val",
    ]
    return [
        {
            "phase_id": "ref8_strict2031",
            "command": primary,
            "command_shell": shlex.join(primary),
            "console_log": str(output_dir / "ref8_strict2031_console.log"),
        },
        {
            "phase_id": "strict1607",
            "command": supplemental,
            "command_shell": shlex.join(supplemental),
            "console_log": str(output_dir / "strict1607_console.log"),
        },
    ]


def _matrix_training_queue_dir(source: EvaluationSource) -> Path | None:
    queue_values = (
        source.training_queue_manifest,
        source.training_queue_detached_launch,
        source.training_queue_detached_status,
        source.training_queue_id,
        source.training_queue_plan_sha256,
    )
    if not any(value is not None for value in queue_values):
        return None
    if not all(value is not None for value in queue_values):
        raise PaperEvaluationError(
            "matrix source has a partial training queue attestation"
        )
    manifest = source.training_queue_manifest
    assert manifest is not None
    manifest = manifest.expanduser().resolve(strict=True)
    if manifest.name != "queue.json":
        raise PaperEvaluationError(
            "matrix source training queue manifest is not canonical"
        )
    queue_id = source.training_queue_id
    plan_sha256 = source.training_queue_plan_sha256
    if (
        not isinstance(queue_id, str)
        or not queue_id
        or not isinstance(plan_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None
    ):
        raise PaperEvaluationError("matrix source training queue identity is invalid")
    return manifest.parent


def _revalidate_matrix_source(
    source: EvaluationSource, cache: HashCache
) -> None:
    """Reconstruct the formal source instead of trusting caller-supplied tags."""

    if source.training_run_root is None:
        raise PaperEvaluationError(
            "matrix source has no formal training run root to revalidate"
        )
    queue_dir = _matrix_training_queue_dir(source)
    try:
        observed = _resolve_pivot_source(
            source.training_run_root,
            cache,
            training_phase=source.training_phase,
            training_queue_dir=queue_dir,
        )
    except (PaperEvaluationError, FileNotFoundError, OSError, ValueError) as exc:
        raise PaperEvaluationError(
            f"matrix source formal provenance revalidation failed: {exc}"
        ) from exc
    if observed != source:
        fields = EvaluationSource.__dataclass_fields__
        drifted = [
            name
            for name in fields
            if getattr(observed, name) != getattr(source, name)
        ]
        raise PaperEvaluationError(
            "matrix source differs from re-resolved formal provenance: "
            + ", ".join(drifted)
        )


def build_plan(
    runtime: Runtime,
    source: EvaluationSource,
    output_dir: Path,
    cache: HashCache,
    *,
    profile: str = FINAL_PROFILE,
    final_gate: Path | None = None,
    matrix_queue_spec: Path | None = None,
) -> dict[str, Any]:
    if profile not in EVALUATION_PROFILES:
        raise PaperEvaluationError(f"unsupported evaluation profile: {profile!r}")
    if matrix_queue_spec is not None and profile != MATRIX_PROFILE:
        raise PaperEvaluationError(
            "--matrix-queue-spec is restricted to matrix_validation"
        )
    declared_contract = (
        source_contracts.formal_paper_run_contract(source.formal_contract_id)
        if source.formal_contract_id is not None
        else None
    )
    if source.formal_contract_id is not None and declared_contract is None:
        raise PaperEvaluationError(
            f"unknown formal paper contract: {source.formal_contract_id!r}"
        )
    run_contract = None
    if (
        source.kind == "pivot_paper_training_run"
        and isinstance(source.training_run_id, str)
        and ":" in source.training_run_id
    ):
        run_contract = source_contracts.formal_paper_run_contract(
            source.training_run_id.split(":", 1)[0]
        )
    if declared_contract is not None and run_contract not in {
        None,
        declared_contract,
    }:
        raise PaperEvaluationError(
            "formal paper source identity differs from the registered contract"
        )
    formal_contract = declared_contract or run_contract
    if (
        formal_contract is not None
        and formal_contract.matrix_validation_only
        and profile != MATRIX_PROFILE
    ):
        raise PaperEvaluationError(
            "matrix-validation-only formal sources require the matrix_validation profile"
        )
    if source.matrix_validation_only and formal_contract is None:
        raise PaperEvaluationError(
            "matrix-validation-only source has no registered formal contract"
        )
    if formal_contract is not None:
        artifact_repository_root = (
            source.artifact_repository_root or ARTIFACT_REPOSITORY_ROOT
        )
        canonical_config = (
            artifact_repository_root / formal_contract.config
        ).resolve(strict=False)
        canonical_root = (
            formal_contract.canonical_training_root(int(source.training_seed))
            if source.training_seed in formal_contract.seeds
            else None
        )
        if (
            source.formal_contract_id != formal_contract.id
            or source.kind != "pivot_paper_training_run"
            or source.evaluation_id
            != f"{formal_contract.id}_seed{source.training_seed}"
            or source.training_run_id
            != f"{formal_contract.id}:{source.training_seed}"
            or source.training_run_root is None
            or Path(source.training_run_root).resolve(strict=False) != canonical_root
            or Path(source.config).resolve(strict=False) != canonical_config
            or source.training_phase != "final"
            or source.diagnostic_only
            or source.matrix_validation_only
            != formal_contract.matrix_validation_only
        ):
            raise PaperEvaluationError(
                "formal paper source identity differs from the registered contract"
            )
    rank_diagnostic_kind = (
        source.kind == "pivot_paper_training_run_rank_diagnostic"
    )
    if rank_diagnostic_kind != source.diagnostic_only:
        raise PaperEvaluationError(
            "rank diagnostic source kind and diagnostic_only flag disagree"
        )
    if source.training_phase == "rank" and not source.diagnostic_only:
        raise PaperEvaluationError(
            "rank phase cannot be presented as a main paper result source"
        )
    if source.diagnostic_only and (
        source.kind != "pivot_paper_training_run_rank_diagnostic"
        or source.training_phase != "rank"
        or source.selected_phase_id != "rank"
        or source.final_phase_id != "confidence"
        or profile != MATRIX_PROFILE
    ):
        raise PaperEvaluationError(
            "S3 rank diagnostics require the matrix_validation profile and "
            "explicit diagnostic source contract"
        )
    if profile == SCREEN_PROFILE and not (
        source.kind == "pivot_token_ablation_training_run"
        and source.training_seed == 17
        and source.training_run_id in {f"L{index}:17" for index in range(5)}
    ):
        raise PaperEvaluationError(
            "screen_validation is restricted to the predeclared L0-L4 seed-17 screen"
        )
    output_dir = output_dir.expanduser().resolve(strict=False)

    matrix_headline_baseline = (
        source.kind == "historical_pure_gdino_explicit"
        and output_dir
        == source_contracts.canonical_validation_root(
            "baseline", source_contracts.BASELINE_TRAIN_SEED
        ).resolve(strict=False)
    )
    if matrix_headline_baseline:
        try:
            source_contracts.validate_fixed_baseline_identity(
                evaluation_id=source.evaluation_id,
                config=source.config,
                checkpoint=source.checkpoint,
                checkpoint_sha256=source.checkpoint_sha256,
            )
            source_contracts.validate_canonical_runtime(
                {
                    "python": str(runtime.python),
                    "data_root": str(runtime.data_root),
                    "device": runtime.device,
                    "batch_size": runtime.batch_size,
                    "num_workers": runtime.num_workers,
                    "amp": runtime.amp,
                    "log_every": runtime.log_every,
                    "eval_seed": EVAL_SEED,
                    "max_ref_batches": 0,
                    "max_tn_batches": 0,
                }
            )
        except source_contracts.EvaluationSourceContractError as exc:
            raise PaperEvaluationError(
                f"headline validation baseline contract failed: {exc}"
            ) from exc
    matrix_main_source = (
        source.kind
        in {
            "pivot_token_ablation_training_run",
            "pivot_paper_training_run",
        }
        and source.training_phase == "final"
        and not source.diagnostic_only
    )
    matrix_rank_diagnostic = (
        source.kind == "pivot_paper_training_run_rank_diagnostic"
        and source.training_phase == "rank"
        and source.diagnostic_only
        and source.selected_phase_id == "rank"
        and source.final_phase_id == "confidence"
    )
    if profile == MATRIX_PROFILE and not (
        (
            (matrix_main_source or matrix_rank_diagnostic)
            and isinstance(source.training_run_id, str)
            and bool(source.training_run_id)
        )
        or matrix_headline_baseline
    ):
        raise PaperEvaluationError(
            "matrix_validation requires a completed formal PIVOT training run "
            "or the canonical fixed-b58 headline validation instance"
        )
    if profile == MATRIX_PROFILE and not matrix_headline_baseline:
        _revalidate_matrix_source(source, cache)
    if output_dir.exists():
        raise FileExistsError(f"evaluation output root must be fresh: {output_dir}")
    if profile != FINAL_PROFILE and final_gate is not None:
        raise PaperEvaluationError("validation profiles cannot bind a final gate")
    strict_records = (
        {label: _strict_manifest_record(label, cache) for label in STRICT_SPECS}
        if profile == FINAL_PROFILE
        else {}
    )
    screen_contract = (
        _screen_calibration_contract(cache)
        if profile in VALIDATION_PROFILES
        else None
    )
    entries: list[tuple[Path, str]] = [
        (source.config, "evaluation_config"),
        (source.checkpoint, "evaluation_checkpoint"),
    ]
    resolved_matrix_queue_spec = (
        matrix_queue_spec.expanduser().resolve(strict=True)
        if matrix_queue_spec is not None
        else None
    )
    if resolved_matrix_queue_spec is not None:
        entries.append((resolved_matrix_queue_spec, "matrix_validation_queue_spec"))
    entries.extend(
        (path, "config_dependency")
        for path in _config_paths(
            source.config,
            repository_root=(
                source.artifact_repository_root
                or ARTIFACT_REPOSITORY_ROOT
            ),
        )
    )
    entries.extend((path, "evaluation_code_dependency") for path in _evaluation_code_paths())
    entries.extend(
        (path, "source_provenance_dependency")
        for path in _evaluation_source_provenance_paths(source)
    )
    entries.extend((path, "evaluation_data_input") for path in _data_input_paths(runtime.data_root))
    for label, record in strict_records.items():
        entries.append((Path(record["path"]), label))
    if screen_contract is not None:
        calibration_prefix = "screen" if profile == SCREEN_PROFILE else "matrix"
        entries.extend(
            (
                (
                    Path(screen_contract["source_manifest"]["path"]),
                    f"{calibration_prefix}_calibration_source",
                ),
                (
                    Path(screen_contract["source_audit"]["path"]),
                    f"{calibration_prefix}_calibration_audit",
                ),
            )
        )
    entries.extend((path, "training_data") for path in source.training_data)
    for path, role in (
        (source.sequence_manifest, "training_sequence_manifest"),
        (source.final_phase_manifest, "training_final_phase_manifest"),
        (source.training_postflight, "training_final_phase_postflight"),
        (source.selected_phase_manifest, "training_selected_phase_manifest"),
        (
            source.selected_training_postflight,
            "training_selected_phase_postflight",
        ),
        (source.training_queue_manifest, "training_queue_manifest"),
        (
            source.training_queue_detached_launch,
            "training_queue_detached_launch",
        ),
        (
            source.training_queue_detached_status,
            "training_queue_detached_status",
        ),
    ):
        if path is not None:
            entries.append((path, role))
    input_records = _merge_input_records(entries, cache)
    checkpoint_records = [
        record
        for record in input_records
        if "evaluation_checkpoint" in record["roles"]
    ]
    if len(checkpoint_records) != 1 or checkpoint_records[0]["sha256"] != source.checkpoint_sha256:
        raise PaperEvaluationError("selected checkpoint changed during plan construction")
    plan = {
        "schema": SCHEMA,
        "status": "planned",
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "artifact_repository_root": str(
            source.artifact_repository_root or ARTIFACT_REPOSITORY_ROOT
        ),
        "artifact_outputs_root": str(ARTIFACT_OUTPUTS_ROOT),
        "evaluation_id": source.evaluation_id,
        "output_dir": str(output_dir),
        "output_dir_fresh_at_plan": True,
        "matrix_validation_queue_spec": (
            _file_record(
                resolved_matrix_queue_spec,
                cache,
                roles=("matrix_validation_queue_spec",),
            )
            if resolved_matrix_queue_spec is not None
            else None
        ),
        "source": {
            **asdict(source),
            "config": str(source.config),
            "checkpoint": str(source.checkpoint),
            "training_run_root": (
                str(source.training_run_root) if source.training_run_root else None
            ),
            "sequence_manifest": (
                str(source.sequence_manifest) if source.sequence_manifest else None
            ),
            "final_phase_manifest": (
                str(source.final_phase_manifest) if source.final_phase_manifest else None
            ),
            "training_postflight": (
                str(source.training_postflight) if source.training_postflight else None
            ),
            "selected_phase_manifest": (
                str(source.selected_phase_manifest)
                if source.selected_phase_manifest
                else None
            ),
            "selected_training_postflight": (
                str(source.selected_training_postflight)
                if source.selected_training_postflight
                else None
            ),
            "training_queue_manifest": (
                str(source.training_queue_manifest)
                if source.training_queue_manifest
                else None
            ),
            "training_queue_detached_launch": (
                str(source.training_queue_detached_launch)
                if source.training_queue_detached_launch
                else None
            ),
            "training_queue_detached_status": (
                str(source.training_queue_detached_status)
                if source.training_queue_detached_status
                else None
            ),
            "artifact_repository_root": (
                str(source.artifact_repository_root)
                if source.artifact_repository_root
                else None
            ),
            "training_data": [str(path) for path in source.training_data],
        },
        "runtime": {
            "python": str(runtime.python),
            "data_root": str(runtime.data_root),
            "device": runtime.device,
            "batch_size": runtime.batch_size,
            "num_workers": runtime.num_workers,
            "amp": runtime.amp,
            "log_every": runtime.log_every,
            "eval_seed": EVAL_SEED,
            "max_ref_batches": 0,
            "max_tn_batches": 0,
        },
        "protocol": {
            "profile": profile,
            "ref_splits": list(
                SCREEN_REF_SPLITS if profile in VALIDATION_PROFILES else REF_SPLITS
            ),
            "strict_manifests": strict_records,
            "screen_calibration": screen_contract,
            "processes": (
                ["validation_calibration"]
                if profile in VALIDATION_PROFILES
                else ["ref8_strict2031", "strict1607"]
            ),
            "strict1607_skip_ref": profile == FINAL_PROFILE,
            "per_example_records": True,
            "release_policy": (
                "development_selection_only_no_ref_test_or_strict_access"
                if profile == SCREEN_PROFILE
                else (
                    "ablation_matrix_validation_only_no_ref_test_or_strict_access"
                    if profile == MATRIX_PROFILE
                    else "single_final_release_ref8_strict2031_strict1607"
                )
            ),
        },
        "commands": _commands(
            runtime,
            source,
            output_dir,
            profile=profile,
            screen_contract=screen_contract,
        ),
        "inputs": {
            "algorithm": "sha256",
            "records": input_records,
        },
    }
    if profile == FINAL_PROFILE and final_gate is not None:
        from tools import stageb_headline_release_contract as headline_release

        try:
            binding = headline_release.bind_final_gate(plan, final_gate)
            release_paths = headline_release.release_input_paths(binding)
        except headline_release.HeadlineReleaseError as exc:
            raise PaperEvaluationError(f"headline final gate failed: {exc}") from exc
        release_entries = [
            (path, "headline_release_receipt") for path in release_paths
        ]
        plan["inputs"]["records"] = _merge_input_records(
            [
                (Path(record["path"]), role)
                for record in plan["inputs"]["records"]
                for role in record["roles"]
            ]
            + release_entries,
            cache,
        )
        plan["headline_release"] = binding
    return plan


def _verify_input_identities(plan: Mapping[str, Any]) -> None:
    records = plan.get("inputs", {}).get("records")
    if not isinstance(records, list) or not records:
        raise PaperEvaluationError("launch plan has no input records")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PaperEvaluationError(f"input record {index} is invalid")
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        stat = path.stat()
        if (
            int(record.get("size_bytes", -1)) != int(stat.st_size)
            or int(record.get("mtime_ns", -1)) != int(stat.st_mtime_ns)
        ):
            raise PaperEvaluationError(f"input changed after hashing: {path}")


def _rehash_inputs(plan: Mapping[str, Any]) -> dict[str, Any]:
    cache = HashCache()
    rows = []
    for record in plan["inputs"]["records"]:
        path = Path(record["path"]).resolve(strict=True)
        observed = cache.digest(path)
        expected = str(record["sha256"])
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "roles": list(record["roles"]),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "observed_size_bytes": int(stat.st_size),
                "observed_mtime_ns": int(stat.st_mtime_ns),
                "passed": (
                    observed == expected
                    and int(stat.st_size) == int(record["size_bytes"])
                    and int(stat.st_mtime_ns) == int(record["mtime_ns"])
                ),
            }
        )
    failed = [row["path"] for row in rows if not row["passed"]]
    if failed:
        raise PaperEvaluationError(f"post-run input drift: {failed}")
    return {
        "schema": INPUT_REHASH_SCHEMA,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "records": rows,
    }


def _subprocess_environment(runtime: Runtime) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PIVOT_ARTIFACT_REPOSITORY_ROOT", None)
    environment["DATA_ROOT"] = str(runtime.data_root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.setdefault("OMP_NUM_THREADS", "1")
    python_paths = [str(REPO_ROOT)]
    if ARTIFACT_REPOSITORY_ROOT != REPO_ROOT:
        python_paths.append(str(ARTIFACT_REPOSITORY_ROOT))
    python_paths.extend(
        path
        for path in environment.get("PYTHONPATH", "").split(os.pathsep)
        if path
    )
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    return environment


def _stream_command(command: Sequence[str], *, runtime: Runtime, log_path: Path) -> int:
    with log_path.open("xb") as raw_log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=_subprocess_environment(runtime),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            close_fds=True,
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.readline(), b""):
            raw_log.write(chunk)
            raw_log.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return int(process.wait())


def _required_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise PaperEvaluationError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PaperEvaluationError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PaperEvaluationError(f"{label} must be an exact integer")
    return result


def _summary_record_path(
    value: Any,
    *,
    summary_path: Path,
    section_dir: Path,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PaperEvaluationError("summary row has no records_jsonl")
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        REPO_ROOT / raw,
        summary_path.parent / raw,
        section_dir / raw,
    ]
    existing = {candidate.resolve(strict=True) for candidate in candidates if candidate.is_file()}
    if len(existing) != 1:
        raise PaperEvaluationError(
            f"records_jsonl must resolve to exactly one file, found {len(existing)}"
        )
    path = next(iter(existing))
    try:
        path.relative_to(section_dir.resolve(strict=True))
    except ValueError as exc:
        raise PaperEvaluationError("records_jsonl escapes its fresh output section") from exc
    return path


def _validate_summary_row_common(
    row: Mapping[str, Any],
    *,
    checkpoint: Path,
    run_id: str,
    seed: int,
) -> None:
    reported = row.get("checkpoint")
    if not isinstance(reported, str) or not reported.strip():
        raise PaperEvaluationError("summary row has no checkpoint")
    path = Path(reported).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.resolve() != checkpoint.resolve():
        raise PaperEvaluationError("summary row checkpoint mismatch")
    if row.get("run_id") != run_id:
        raise PaperEvaluationError("summary row run_id mismatch")
    if _required_int(row.get("seed"), label="summary seed") != seed:
        raise PaperEvaluationError("summary row seed mismatch")
    if _required_int(row.get("max_batches"), label="summary max_batches") != 0:
        raise PaperEvaluationError("formal evaluation requires max_batches=0")
    if _required_int(row.get("invalid_records", 0), label="invalid_records") != 0:
        raise PaperEvaluationError("formal evaluation contains invalid records")


def _load_summary(path: Path, *, label: str) -> Mapping[str, Any]:
    summary = _read_json(path, label=label)
    if set(summary) != {"refcoco", "tn"}:
        raise PaperEvaluationError(f"{label} must contain exactly refcoco and tn")
    if not isinstance(summary["refcoco"], list) or not isinstance(summary["tn"], list):
        raise PaperEvaluationError(f"{label} sections must be lists")
    return summary


def _verify_ref_rows(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    section_dir: Path,
    checkpoint: Path,
    run_id: str,
    cache: HashCache,
    expected_splits: Sequence[str] = REF_SPLITS,
) -> dict[str, Any]:
    from tools.stageb_eval_records import (  # lazy: list stays lightweight
        RefRecordContractError,
        load_formal_ref_records,
    )

    rows = summary["refcoco"]
    expected_splits = tuple(expected_splits)
    if len(rows) != len(expected_splits):
        raise PaperEvaluationError(
            "Ref summary must contain exactly "
            f"{len(expected_splits)} rows, found {len(rows)}"
        )
    by_split: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PaperEvaluationError("REF8 summary contains a non-object row")
        split = str(row.get("dataset", ""))
        if split not in expected_splits or split in by_split:
            raise PaperEvaluationError(f"Ref summary split set is invalid: {split!r}")
        by_split[split] = row
    if set(by_split) != set(expected_splits):
        raise PaperEvaluationError("Ref summary does not cover the fixed split set")

    artifacts: dict[str, Any] = {}
    for split_index, split in enumerate(expected_splits):
        row = by_split[split]
        _validate_summary_row_common(
            row,
            checkpoint=checkpoint,
            run_id=run_id,
            seed=EVAL_SEED + split_index * 100000,
        )
        expected = REF_SPLIT_CONTRACT[split]
        if _required_int(row.get("manifest_n"), label=f"{split}.manifest_n") != int(expected["rows"]):
            raise PaperEvaluationError(f"{split} manifest row-count contract mismatch")
        if str(row.get("manifest_sha256", "")).lower() != expected["sha256"]:
            raise PaperEvaluationError(f"{split} manifest SHA-256 contract mismatch")
        records_path = _summary_record_path(
            row.get("records_jsonl"),
            summary_path=summary_path,
            section_dir=section_dir,
        )
        try:
            loaded = load_formal_ref_records(
                str(records_path),
                base_dir=section_dir,
                label=f"formal REF8 {split}",
                split=split,
                summary_row=row,
                summary_path=summary_path,
                split_contract=REF_SPLIT_CONTRACT,
            )
        except (RefRecordContractError, OSError, ValueError) as exc:
            raise PaperEvaluationError(f"{split} record validation failed: {exc}") from exc
        artifacts[split] = {
            "summary_acc50": float(row["acc50"]),
            "manifest_n": loaded.manifest_n,
            "manifest_sha256": loaded.manifest_sha256,
            "records": _file_record(records_path, cache, roles=(f"ref:{split}",)),
        }
    return artifacts


def _verify_screen_tn_row(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    section_dir: Path,
    checkpoint: Path,
    run_id: str,
    cache: HashCache,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.compare_stageb_fpr95_records import (
        RecordComparisonError,
        exact_fpr95,
        load_manifest,
        load_tn_records,
    )
    from tools.stageb_screen_calibration import (
        DERIVATION_ALGORITHM,
        EVAL_SPLIT,
        SCHEMA as BINDING_SCHEMA,
        ScreenCalibrationError,
        load_binding,
        sha256_file,
    )

    rows = summary["tn"]
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise PaperEvaluationError(
            "screen calibration summary must contain exactly one TN row"
        )
    row = rows[0]
    _validate_summary_row_common(
        row,
        checkpoint=checkpoint,
        run_id=run_id,
        seed=EVAL_SEED,
    )
    expected_n = int(contract["source_manifest"]["rows"])
    for field in ("manifest_n", "num_pairs", "screen_calibration_source_n"):
        if _required_int(row.get(field), label=f"screen.{field}") != expected_n:
            raise PaperEvaluationError(f"screen calibration {field} mismatch")
    expected_scalars = {
        "screen_calibration_binding_schema": BINDING_SCHEMA,
        "screen_calibration_derivation_algorithm": DERIVATION_ALGORITHM,
        "screen_calibration_source_sha256": contract["source_manifest"]["sha256"],
        "screen_calibration_audit_sha256": contract["source_audit"]["sha256"],
        "screen_calibration_scope": "proposal_covered_verified",
        "screen_calibration_single_edit": True,
    }
    for field, expected in expected_scalars.items():
        if row.get(field) != expected:
            raise PaperEvaluationError(
                f"screen calibration summary {field} mismatch"
            )
    source_path = _resolve_manifest_path(
        row.get("screen_calibration_source_path"),
        label="screen_calibration_source_path",
    )
    audit_path = _resolve_manifest_path(
        row.get("screen_calibration_audit_path"),
        label="screen_calibration_audit_path",
    )
    if source_path != Path(contract["source_manifest"]["path"]).resolve(
        strict=True
    ) or audit_path != Path(contract["source_audit"]["path"]).resolve(strict=True):
        raise PaperEvaluationError("screen calibration source/audit path mismatch")
    derived_path = _resolve_manifest_path(
        row.get("screen_calibration_derived_path"),
        label="screen_calibration_derived_path",
    )
    binding_path = _resolve_manifest_path(
        row.get("screen_calibration_binding_path"),
        label="screen_calibration_binding_path",
    )
    for path, label in (
        (derived_path, "derived manifest"),
        (binding_path, "binding"),
    ):
        try:
            path.relative_to(section_dir.resolve(strict=True))
        except ValueError as exc:
            raise PaperEvaluationError(
                f"screen calibration {label} escapes its fresh output"
            ) from exc
    if sha256_file(binding_path) != row.get("screen_calibration_binding_sha256"):
        raise PaperEvaluationError("screen calibration binding SHA mismatch")
    try:
        binding = load_binding(binding_path, expected_derived=derived_path)
    except (ScreenCalibrationError, OSError, ValueError) as exc:
        raise PaperEvaluationError(
            f"screen calibration deterministic binding failed: {exc}"
        ) from exc
    if (
        dict(binding.source_manifest)
        != {
            key: contract["source_manifest"][key]
            for key in ("path", "sha256", "size_bytes", "rows")
        }
        or dict(binding.source_audit)
        != {
            key: contract["source_audit"][key]
            for key in ("path", "sha256", "size_bytes")
        }
        or binding.row_mapping_sha256
        != row.get("screen_calibration_row_mapping_sha256")
    ):
        raise PaperEvaluationError("screen calibration binding source contract mismatch")
    if binding.eval_split != EVAL_SPLIT:
        raise PaperEvaluationError("screen calibration binding split mismatch")
    if str(binding.derived_manifest["sha256"]) != str(
        row.get("manifest_sha256", "")
    ) or str(binding.derived_manifest["sha256"]) != str(
        row.get("screen_calibration_derived_sha256", "")
    ):
        raise PaperEvaluationError("screen calibration derived SHA mismatch")

    records_path = _summary_record_path(
        row.get("records_jsonl"),
        summary_path=summary_path,
        section_dir=section_dir,
    )
    try:
        manifest = load_manifest(derived_path)
        loaded = load_tn_records(
            records_path, manifest, label="screen calibration"
        )
    except (RecordComparisonError, OSError, ValueError) as exc:
        raise PaperEvaluationError(
            f"screen calibration record binding failed: {exc}"
        ) from exc
    if loaded.manifest_binding_mode != "legacy_direct_source_v1":
        raise PaperEvaluationError(
            "screen calibration records must directly bind the deterministic derived manifest"
        )
    if loaded.run_ids != (run_id,) or not bool(loaded.valid.all()):
        raise PaperEvaluationError("screen calibration records are invalid/mixed")
    if set(manifest.splits) != {EVAL_SPLIT}:
        raise PaperEvaluationError("screen calibration derived split drifted")
    measured = float(exact_fpr95(loaded.positive, loaded.negative)["fpr"])
    try:
        reported = float(row.get("fpr95tpr"))
    except (TypeError, ValueError) as exc:
        raise PaperEvaluationError("screen calibration FPR95 is invalid") from exc
    if not math.isfinite(reported) or not math.isclose(
        measured, reported, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PaperEvaluationError(
            f"screen calibration FPR95 {reported} != record replay {measured}"
        )
    return {
        "summary_fpr95": reported,
        "scope": "proposal_covered_verified",
        "single_edit": True,
        "manifest_n": len(loaded.rows),
        "source_manifest": dict(binding.source_manifest),
        "source_audit": dict(binding.source_audit),
        "derived_manifest": dict(binding.derived_manifest),
        "binding": _file_record(
            binding_path, cache, roles=("screen_calibration_binding",)
        ),
        "records": _file_record(
            records_path, cache, roles=("screen_calibration", "tn_records")
        ),
    }


def _verify_tn_row(
    summary: Mapping[str, Any],
    *,
    label: str,
    summary_path: Path,
    section_dir: Path,
    checkpoint: Path,
    run_id: str,
    cache: HashCache,
) -> dict[str, Any]:
    from tools.compare_stageb_fpr95_records import (
        RecordComparisonError,
        exact_fpr95,
        load_manifest,
        load_tn_records,
    )

    rows = summary["tn"]
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise PaperEvaluationError(f"{label} summary must contain exactly one TN row")
    row = rows[0]
    _validate_summary_row_common(
        row,
        checkpoint=checkpoint,
        run_id=run_id,
        seed=EVAL_SEED,
    )
    expected = STRICT_SPECS[label]
    if _required_int(row.get("manifest_n"), label=f"{label}.manifest_n") != int(expected["rows"]):
        raise PaperEvaluationError(f"{label} summary manifest_n mismatch")
    if _required_int(row.get("num_pairs"), label=f"{label}.num_pairs") != int(expected["rows"]):
        raise PaperEvaluationError(f"{label} summary num_pairs is not full-set")
    if str(row.get("source_manifest_sha256", "")).lower() != expected["sha256"]:
        raise PaperEvaluationError(f"{label} summary source-manifest SHA mismatch")
    if _required_int(row.get("source_manifest_n"), label=f"{label}.source_manifest_n") != int(expected["rows"]):
        raise PaperEvaluationError(f"{label} summary source-manifest N mismatch")
    source_path = _resolve_manifest_path(
        row.get("source_manifest_path"), label=f"{label}.source_manifest_path"
    )
    if source_path != Path(expected["path"]).resolve(strict=True):
        raise PaperEvaluationError(f"{label} summary source-manifest path mismatch")

    records_path = _summary_record_path(
        row.get("records_jsonl"),
        summary_path=summary_path,
        section_dir=section_dir,
    )
    try:
        manifest = load_manifest(Path(expected["path"]))
        loaded = load_tn_records(records_path, manifest, label=f"formal {label}")
    except (RecordComparisonError, OSError, ValueError) as exc:
        raise PaperEvaluationError(f"{label} record binding failed: {exc}") from exc
    if loaded.manifest_binding_mode != "source_to_derived_v1":
        raise PaperEvaluationError(f"{label} records lack the two-layer manifest binding")
    if loaded.run_ids != (run_id,):
        raise PaperEvaluationError(f"{label} record run_id mismatch: {loaded.run_ids}")
    if not bool(loaded.valid.all()):
        raise PaperEvaluationError(f"{label} formal records contain invalid rows")
    record_manifest_hashes = {
        str(record.get("manifest_sha256", "")).lower()
        for record in loaded.rows
    }
    if len(record_manifest_hashes) != 1 or str(
        row.get("manifest_sha256", "")
    ).lower() != next(iter(record_manifest_hashes)):
        raise PaperEvaluationError(
            f"{label} summary and records bind different derived manifests"
        )
    measured = float(exact_fpr95(loaded.positive, loaded.negative)["fpr"])
    try:
        reported = float(row.get("fpr95tpr"))
    except (TypeError, ValueError) as exc:
        raise PaperEvaluationError(f"{label} summary fpr95tpr is invalid") from exc
    if not math.isfinite(reported) or not math.isclose(
        measured, reported, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PaperEvaluationError(
            f"{label} summary FPR95 {reported} != record replay {measured}"
        )
    return {
        "summary_fpr95": reported,
        "manifest_binding_mode": loaded.manifest_binding_mode,
        "manifest_n": len(loaded.rows),
        "source_manifest_sha256": manifest.sha256,
        "derived_manifest_sha256": str(row.get("manifest_sha256", "")),
        "records": _file_record(records_path, cache, roles=(label, "tn_records")),
    }


def _postflight_screen(
    plan: Mapping[str, Any], input_rehash: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        planned_artifact_root = Path(
            str(plan.get("artifact_repository_root", ""))
        ).resolve(strict=True)
        planned_outputs_root = (planned_artifact_root / "outputs").resolve(
            strict=True
        )
        declared_outputs_root = Path(
            str(plan.get("artifact_outputs_root", ""))
        ).resolve(strict=True)
    except OSError as exc:
        raise PaperEvaluationError(
            "evaluation artifact-root binding drifted"
        ) from exc
    if (
        planned_outputs_root != ARTIFACT_OUTPUTS_ROOT
        or declared_outputs_root != ARTIFACT_OUTPUTS_ROOT
    ):
        raise PaperEvaluationError("evaluation artifact-root binding drifted")
    protocol = plan.get("protocol")
    profile = (
        protocol.get("profile") if isinstance(protocol, Mapping) else None
    )
    if profile not in VALIDATION_PROFILES:
        raise PaperEvaluationError(
            f"validation/calibration postflight received profile {profile!r}"
        )
    output_dir = Path(plan["output_dir"]).resolve(strict=True)
    checkpoint = Path(plan["source"]["checkpoint"]).resolve(strict=True)
    expected_checkpoint_sha = str(plan["source"]["checkpoint_sha256"])
    cache = HashCache()
    observed_checkpoint_sha = cache.digest(checkpoint)
    if observed_checkpoint_sha != expected_checkpoint_sha:
        raise PaperEvaluationError(
            "checkpoint changed between screen launch and postflight"
        )
    section_dir = (output_dir / "validation_calibration").resolve(strict=True)
    summary_path = (section_dir / "summary.json").resolve(strict=True)
    summary = _load_summary(
        summary_path, label="validation-only Ref+calibration summary"
    )
    run_id = _checkpoint_run_id(checkpoint)
    ref = _verify_ref_rows(
        summary,
        summary_path=summary_path,
        section_dir=section_dir,
        checkpoint=checkpoint,
        run_id=run_id,
        cache=cache,
        expected_splits=SCREEN_REF_SPLITS,
    )
    calibration_contract = (
        protocol.get("screen_calibration")
        if isinstance(protocol, Mapping)
        else None
    )
    if not isinstance(calibration_contract, Mapping):
        raise PaperEvaluationError("screen plan has no calibration contract")
    calibration = _verify_screen_tn_row(
        summary,
        summary_path=summary_path,
        section_dir=section_dir,
        checkpoint=checkpoint,
        run_id=run_id,
        cache=cache,
        contract=calibration_contract,
    )
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "validated_at_utc": _utc_now(),
        "evaluation_id": plan["evaluation_id"],
        "profile": profile,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": observed_checkpoint_sha,
            "run_id": run_id,
        },
        "fixed_runtime": {
            "eval_seed": EVAL_SEED,
            "max_ref_batches": 0,
            "max_tn_batches": 0,
        },
        "input_rehash": dict(input_rehash),
        "artifacts": {
            "summary": _file_record(
                summary_path,
                cache,
                roles=(
                    (
                        "screen_ref_validation"
                        if profile == SCREEN_PROFILE
                        else "matrix_ref_validation"
                    ),
                    (
                        "screen_calibration"
                        if profile == SCREEN_PROFILE
                        else "matrix_calibration"
                    ),
                ),
            ),
            "ref_validation": ref,
            (
                "screen_calibration"
                if profile == SCREEN_PROFILE
                else "matrix_calibration"
            ): calibration,
        },
        "contracts": {
            "ref_validation_split_set_exact": True,
            "ref_test_splits_not_run": True,
            "strict2031_not_run": True,
            "strict1607_not_run": True,
            "full_per_example_records": True,
            "zero_invalid_records": True,
            "calibration_source_to_derived_binding": True,
            "proposal_covered_scope_preserved": True,
            "single_edit_calibration_only": True,
            "checkpoint_consistent_across_all_rows": True,
        },
    }


def _postflight(plan: Mapping[str, Any], input_rehash: Mapping[str, Any]) -> dict[str, Any]:
    protocol = plan.get("protocol")
    profile = (
        protocol.get("profile", FINAL_PROFILE)
        if isinstance(protocol, Mapping)
        else FINAL_PROFILE
    )
    if profile in VALIDATION_PROFILES:
        return _postflight_screen(plan, input_rehash)
    if profile != FINAL_PROFILE:
        raise PaperEvaluationError(f"unsupported postflight profile: {profile!r}")
    output_dir = Path(plan["output_dir"]).resolve(strict=True)
    checkpoint = Path(plan["source"]["checkpoint"]).resolve(strict=True)
    expected_checkpoint_sha = str(plan["source"]["checkpoint_sha256"])
    cache = HashCache()
    observed_checkpoint_sha = cache.digest(checkpoint)
    if observed_checkpoint_sha != expected_checkpoint_sha:
        raise PaperEvaluationError("checkpoint changed between launch and postflight")
    run_id = _checkpoint_run_id(checkpoint)

    primary_dir = (output_dir / "ref8_strict2031").resolve(strict=True)
    supplemental_dir = (output_dir / "strict1607").resolve(strict=True)
    primary_summary_path = (primary_dir / "summary.json").resolve(strict=True)
    supplemental_summary_path = (supplemental_dir / "summary.json").resolve(strict=True)
    primary_summary = _load_summary(primary_summary_path, label="REF8+strict2031 summary")
    supplemental_summary = _load_summary(supplemental_summary_path, label="strict1607 summary")
    if supplemental_summary["refcoco"]:
        raise PaperEvaluationError("strict1607 --skip_ref output unexpectedly contains REF rows")

    ref = _verify_ref_rows(
        primary_summary,
        summary_path=primary_summary_path,
        section_dir=primary_dir,
        checkpoint=checkpoint,
        run_id=run_id,
        cache=cache,
    )
    strict2031 = _verify_tn_row(
        primary_summary,
        label="strict2031",
        summary_path=primary_summary_path,
        section_dir=primary_dir,
        checkpoint=checkpoint,
        run_id=run_id,
        cache=cache,
    )
    strict1607 = _verify_tn_row(
        supplemental_summary,
        label="strict1607",
        summary_path=supplemental_summary_path,
        section_dir=supplemental_dir,
        checkpoint=checkpoint,
        run_id=run_id,
        cache=cache,
    )
    from tools import stageb_headline_release_contract as headline_release

    try:
        release_evidence = headline_release.validate_completed_final_plan(
            plan,
            final_artifacts={
                "ref8": ref,
                "strict2031": strict2031,
                "strict1607": strict1607,
            },
        )
    except headline_release.HeadlineReleaseError as exc:
        raise PaperEvaluationError(
            f"headline final receipt replay failed: {exc}"
        ) from exc
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "validated_at_utc": _utc_now(),
        "evaluation_id": plan["evaluation_id"],
        "profile": FINAL_PROFILE,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": observed_checkpoint_sha,
            "run_id": run_id,
        },
        "fixed_runtime": {
            "eval_seed": EVAL_SEED,
            "max_ref_batches": 0,
            "max_tn_batches": 0,
        },
        "input_rehash": dict(input_rehash),
        "artifacts": {
            "primary_summary": _file_record(
                primary_summary_path, cache, roles=("ref8", "strict2031")
            ),
            "supplemental_summary": _file_record(
                supplemental_summary_path, cache, roles=("strict1607",)
            ),
            "ref8": ref,
            "strict2031": strict2031,
            "strict1607": strict1607,
        },
        "headline_release": release_evidence,
        "contracts": {
            "ref_split_set_exact": True,
            "full_per_example_records": True,
            "zero_invalid_records": True,
            "locked_manifest_binding": True,
            "checkpoint_consistent_across_all_rows": True,
            "strict1607_skip_ref_observed": True,
            "one_time_final_gate_verified": True,
            "fixed_s2f_selection_receipt_verified": True,
            "fixed_b58_identity_verified": True,
            "cross_model_runtime_code_data_parity_verified": True,
            "fresh_single_use_final_consumption_verified": True,
        },
    }


def _execute(plan: dict[str, Any], runtime: Runtime) -> int:
    output_dir = Path(plan["output_dir"])
    protocol = plan.get("protocol")
    profile = protocol.get("profile") if isinstance(protocol, Mapping) else None
    if profile == FINAL_PROFILE:
        from tools import stageb_headline_release_contract as headline_release

        try:
            consumption = headline_release.consume_final_instance(plan)
            plan["headline_release"]["final_consumption"] = consumption
            plan["inputs"]["records"] = _merge_input_records(
                [
                    (Path(record["path"]), role)
                    for record in plan["inputs"]["records"]
                    for role in record["roles"]
                ]
                + [
                    (
                        Path(consumption["path"]),
                        "headline_final_consumption_receipt",
                    )
                ],
                HashCache(),
            )
        except (headline_release.HeadlineReleaseError, OSError) as exc:
            print(
                f"[FAIL] headline final gate consumption failed: {exc}",
                file=sys.stderr,
            )
            return 1
    output_dir.mkdir(parents=True, exist_ok=False)
    launch_path = output_dir / "launch_manifest.json"
    plan["status"] = "running"
    plan["started_at_utc"] = _utc_now()
    plan["completed_phases"] = []
    _write_json_atomic(launch_path, plan)
    try:
        for command_spec in plan["commands"]:
            phase_id = command_spec["phase_id"]
            plan["current_phase"] = phase_id
            _write_json_atomic(launch_path, plan)
            _verify_input_identities(plan)
            returncode = _stream_command(
                command_spec["command"],
                runtime=runtime,
                log_path=Path(command_spec["console_log"]),
            )
            if returncode != 0:
                raise PaperEvaluationError(
                    f"evaluation phase {phase_id} exited with code {returncode}"
                )
            plan["completed_phases"].append(
                {
                    "phase_id": phase_id,
                    "status": "completed",
                    "returncode": 0,
                    "finished_at_utc": _utc_now(),
                }
            )
            _write_json_atomic(launch_path, plan)
        input_rehash = _rehash_inputs(plan)
        input_rehash_path = output_dir / "input_rehash.json"
        _write_json_atomic(input_rehash_path, input_rehash)
        postflight = _postflight(plan, input_rehash)
        postflight_path = output_dir / "postflight.json"
        _write_json_atomic(postflight_path, postflight)
        cache = HashCache()
        plan["input_rehash_artifact"] = _file_record(
            input_rehash_path, cache, roles=("input_rehash",)
        )
        plan["postflight_artifact"] = _file_record(
            postflight_path, cache, roles=("postflight",)
        )
        plan["postflight"] = postflight
        plan["status"] = "completed"
        plan["current_phase"] = None
        plan["finished_at_utc"] = _utc_now()
        _write_json_atomic(launch_path, plan)
        print(f"[OK] formal paper evaluation completed: {output_dir}")
        return 0
    except BaseException as exc:
        plan["status"] = "failed"
        plan["failure_phase"] = plan.get("current_phase") or "postflight"
        plan["error"] = f"{type(exc).__name__}: {exc}"
        plan["finished_at_utc"] = _utc_now()
        _write_json_atomic(launch_path, plan)
        print(f"[FAIL] {plan['error']}", file=sys.stderr)
        return 1


def _list_payload() -> dict[str, Any]:
    from tools.run_stageb_paper_ablation_matrices import (
        ROWS as PAPER_ROWS,
        SEEDS as PAPER_SEEDS,
    )
    from tools.run_stageb_token_ablation_matrix import (
        ROWS as TOKEN_ROWS,
        SEEDS as TOKEN_SEEDS,
    )

    if tuple(PAPER_SEEDS) != tuple(TOKEN_SEEDS):
        raise PaperEvaluationError("paper and token training seed contracts differ")
    all_rows = tuple(TOKEN_ROWS) + tuple(PAPER_ROWS)
    formal_contracts = tuple(source_contracts.FORMAL_PAPER_RUN_CONTRACTS.values())
    formal_run_ids = [
        run_id
        for contract in formal_contracts
        for run_id in contract.dedicated_queue_run_ids
    ]

    return {
        "schema": "pivot.stageb.paper_evaluation_catalog/v1",
        "eval_seed": EVAL_SEED,
        "ref_splits": list(REF_SPLITS),
        "screen_ref_splits": list(SCREEN_REF_SPLITS),
        "matrix_ref_splits": list(SCREEN_REF_SPLITS),
        "strict_manifests": {
            label: {
                "path": str(Path(specification["path"]).resolve(strict=False)),
                "rows": specification["rows"],
                "sha256": specification["sha256"],
            }
            for label, specification in STRICT_SPECS.items()
        },
        "token_training_rows": [row.row_id for row in TOKEN_ROWS],
        "paper_training_rows": [row.row_id for row in PAPER_ROWS],
        "all_training_rows": [
            *(row.row_id for row in all_rows),
            *(contract.id for contract in formal_contracts),
        ],
        "formal_paper_training_rows": [
            contract.id for contract in formal_contracts
        ],
        "paper_training_seeds": list(PAPER_SEEDS),
        "paper_training_run_ids": [
            f"{row.row_id}:{seed}" for row in all_rows for seed in PAPER_SEEDS
        ]
        + formal_run_ids,
        "source_modes": [
            "completed_pivot_training_run_root",
            "completed_s3_rank_diagnostic_only",
            "explicit_historical_pure_gdino_config_checkpoint",
        ],
        "profiles": {
            SCREEN_PROFILE: {
                "processes": ["validation_calibration"],
                "release_policy": "development_selection_only",
            },
            MATRIX_PROFILE: {
                "processes": ["validation_calibration"],
                "release_policy": "ablation_matrix_validation_only",
            },
            FINAL_PROFILE: {
                "processes": ["ref8_strict2031", "strict1607_skip_ref"],
                "release_policy": "single_final_release",
            },
        },
        "processes": ["ref8_strict2031", "strict1607_skip_ref"],
    }


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-run-root", type=Path)
    parser.add_argument(
        "--training-phase",
        choices=("final", "rank"),
        default="final",
        help=(
            "select the final checkpoint, or S3's completed rank checkpoint "
            "as a matrix-validation-only diagnostic source"
        ),
    )
    parser.add_argument(
        "--training-queue-dir",
        type=Path,
        help=(
            "completed serial queue attesting a formal token run outside the "
            "default canonical output root"
        ),
    )
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--baseline-id", default="gdino_stageb_data_ft_b58")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=EVALUATION_PROFILES,
        default=FINAL_PROFILE,
        help=(
            "screen_validation is the predeclared L0-L4 seed-17 screen; "
            "matrix_validation exposes validation/calibration for completed "
            "formal PIVOT runs; final runs the sealed "
            "Ref8+strict2031+strict1607 release."
        ),
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PIVOT_PYTHON", str(DEFAULT_PYTHON)),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", str(DEFAULT_DATA_ROOT))),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--matrix-queue-spec",
        type=Path,
        help=(
            "immutable queue input specification to hash-bind a "
            "matrix_validation launch"
        ),
    )
    parser.add_argument(
        "--final-gate",
        type=Path,
        help="canonical one-time headline gate; required for profile=final",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch sealed Stage-B paper evaluations from completed training runs."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--json", action="store_true")
    subparsers.add_parser("seal-selection-receipt")
    subparsers.add_parser("seal-final-gate")
    for mode in ("dry-run", "run"):
        child = subparsers.add_parser(mode)
        _add_source_arguments(child)
        _add_runtime_arguments(child)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "list":
        payload = _list_payload()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Formal Stage-B paper evaluation")
            print(f"  eval seed: {payload['eval_seed']}")
            print(f"  REF splits: {', '.join(payload['ref_splits'])}")
            print(
                "  training runs: "
                + ", ".join(payload["paper_training_run_ids"])
            )
        return 0
    if args.mode in {"seal-selection-receipt", "seal-final-gate"}:
        from tools import stageb_headline_release_contract as headline_release

        try:
            payload = (
                headline_release.seal_selection_receipt()
                if args.mode == "seal-selection-receipt"
                else headline_release.seal_final_gate()
            )
        except (
            headline_release.HeadlineReleaseError,
            FileExistsError,
            FileNotFoundError,
            OSError,
        ) as exc:
            print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.profile == FINAL_PROFILE and args.final_gate is None:
        print("[FAIL] final profile requires --final-gate", file=sys.stderr)
        return 2
    if args.profile != FINAL_PROFILE and args.final_gate is not None:
        print("[FAIL] validation profiles cannot accept --final-gate", file=sys.stderr)
        return 2
    try:
        runtime = _runtime_from_args(args)
        cache = HashCache()
        source = _resolve_source(args, cache)
        plan = build_plan(
            runtime,
            source,
            args.output_dir,
            cache,
            profile=args.profile,
            final_gate=args.final_gate,
            matrix_queue_spec=args.matrix_queue_spec,
        )
    except (PaperEvaluationError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.mode == "dry-run":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _execute(plan, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
