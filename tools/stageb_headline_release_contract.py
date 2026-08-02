#!/usr/bin/env python3
"""One-time provenance contract for the Stage-B headline evaluation.

The validation surface is the only surface used to decide whether the fixed
successful-update batch-slot-matched M0 main run (S2F architecture/objective)
is eligible for final
evaluation.  There is no row-selection fallback.  A sealed gate authorizes
exactly four fresh final evaluations: the fixed historical b58 checkpoint and
M0 seeds 17, 42, and 73.  Each authorized
instance can be claimed once.

This module is imported lazily by the evaluator and results builder so the
receipt logic can be replayed without importing the training stack.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import runpy
import shlex
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

SELECTION_RECEIPT_SCHEMA = "pivot.stageb.headline_selection_receipt/v1"
BASELINE_CONTRACT_SCHEMA = "pivot.stageb.headline_baseline_contract/v1"
CANDIDATE_CONTRACT_SCHEMA = "pivot.stageb.headline_candidate_contract/v1"
PARITY_CONTRACT_SCHEMA = "pivot.stageb.headline_evaluation_parity/v1"
FINAL_GATE_SCHEMA = "pivot.stageb.headline_final_evaluation_gate/v1"
FINAL_CONSUMPTION_SCHEMA = "pivot.stageb.headline_final_consumption/v1"
INSTANCE_SCHEMA = "pivot.stageb.headline_final_instance/v1"
RELEASE_PROVENANCE_SCHEMA = "stageb-headline-release-provenance-v1"
COMPLETION_RECEIPT_SCHEMA = "pivot.stageb.paper_ablation_completion_receipt/v1"
M0_ANCESTRY_SCHEMA = "pivot.stageb.headline_m0_model_state_ancestry/v1"
M0_ATTEMPT_SCHEMA = "pivot.stageb.headline_m0_training_attempt/v1"
M0_ATTEMPT_TELEMETRY_SCHEMA = "pivot.stageb.headline_m0_attempt_telemetry/v1"
M0_STABLE_CLOSURE_SCHEMA = "pivot.stageb.headline_m0_stable_input_closure/v1"

BASELINE_ID = "gdino_stageb_data_ft_b58"
CANDIDATE_ID = "M0"
CANDIDATE_ARCHITECTURE_OBJECTIVE = "S2F"
CANDIDATE_SEEDS = (17, 42, 73)
BASELINE_BATCH_SIZE = 19
BASELINE_OPTIMIZER_UPDATES = 49539
BASELINE_OPTIMIZER_STATE_COUNT = 661
BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS = 941241
CANDIDATE_BATCH_SIZE = 40
CANDIDATE_UPDATES = 23532
CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS = 941280
M0_CONFIG_SHA256 = "65d32d27e4003a7b318c2beb8f9f8fd4e4d6a255c66e41d84961dab361c8ded2"
BASELINE_TRAIN_SEED = 42
VALIDATION_REF_SPLITS = (
    "refcoco_val",
    "refcocop_val",
    "refcocog_val",
)
SELECTION_POLICY = (
    "fixed_M0_compute_matched_S2F_objective_three_seeds_validation_eligibility_only_"
    "no_posthoc_row_selection"
)
SELECTION_THRESHOLDS = {
    "ref_split_mean_delta_min": -0.01,
    "ref_val3_macro_mean_delta_strict_min": 0.0,
    "calibration_fpr95_mean_delta_strict_max": 0.0,
    "calibration_positive_q05_mean_delta_min": -0.02,
}
HEADLINE_BOOTSTRAP_CONTRACT = {
    "iterations": 5000,
    "confidence": 0.95,
    "seed": 20260717,
    "unit": "global_canonical_coco_image_cluster_seed_first_candidate_mean",
}

DEFAULT_PYTHON = Path("/home/haoyi/miniconda/envs/gdino5090/bin/python")
DEFAULT_DATA_ROOT = Path("/media/haoyi/T9/data")
CANONICAL_RUNTIME = {
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

_BASELINE_TRAIN_ROOT = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch"
)
FIXED_BASELINE: Mapping[str, Any] = {
    "id": BASELINE_ID,
    "train_seed": BASELINE_TRAIN_SEED,
    "role": "fixed_historical_checkpoint",
    "config": str((_BASELINE_TRAIN_ROOT / "config_cfg.py").resolve(strict=False)),
    "config_sha256": "f0dc568c6f35225176712618d5f3449b253478ef32a2b65fa9e089da1ad8a05f",
    "checkpoint": str(
        (_BASELINE_TRAIN_ROOT / "checkpoint0001.pth").resolve(strict=False)
    ),
    "checkpoint_sha256": "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157",
    "legacy_results_manifest": str(
        (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/manifests/"
            "baseline_b58_seed42_results_manifest.json"
        ).resolve(strict=False)
    ),
    "legacy_results_manifest_sha256": (
        "70b0752359419615793ec7e9d8009065cb3f66ddcfd081d2ac30c0fff6134b4b"
    ),
    "legacy_roots": {
        "ref8": {
            "path": str(
                (
                    REPO_ROOT
                    / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42"
                ).resolve(strict=False)
            ),
            "summary_sha256": (
                "a873bfbf25ca467f28684373e744234df526a725896e3c0361ab93d45dcb7266"
            ),
        },
        "strict2031": {
            "path": str(
                (
                    REPO_ROOT
                    / "outputs/paper_cvpr_v1/"
                    "baseline_b58_strict2031_seed42_v3_contract"
                ).resolve(strict=False)
            ),
            "summary_sha256": (
                "c127032cdbd29052bca59e1ea9e9de3bd45ade049a9deb6ae9b0f235ddd328d7"
            ),
        },
        "strict1607": {
            "path": str(
                (
                    REPO_ROOT
                    / "outputs/paper_cvpr_v1/"
                    "baseline_b58_strict1607_seed42_v3_contract"
                ).resolve(strict=False)
            ),
            "summary_sha256": (
                "1512caa67e434e3385067ed122ac93e48ca9a37c4aba11b7655d9a79e8ad6e44"
            ),
        },
    },
}

VALIDATION_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/evaluations/headline_selection"
FINAL_RELEASE_ROOT = REPO_ROOT / "outputs/paper_cvpr_v1/evaluations/final_release"
SELECTION_RECEIPT_PATH = (
    REPO_ROOT / "outputs/paper_cvpr_v1/gates/headline_selection_receipt.json"
)
FINAL_CONTRACT_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/gates/headline_final_release"
)
BASELINE_CONTRACT_PATH = FINAL_CONTRACT_ROOT / "baseline_contract.json"
CANDIDATE_CONTRACT_PATH = FINAL_CONTRACT_ROOT / "candidate_contract.json"
PARITY_CONTRACT_PATH = FINAL_CONTRACT_ROOT / "evaluation_parity_contract.json"
FINAL_GATE_PATH = FINAL_CONTRACT_ROOT / "final_gate.json"
FINAL_CONSUMPTION_ROOT = FINAL_CONTRACT_ROOT / "consumptions"
PAPER_ABLATION_COMPLETION_RECEIPT_PATH = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/paper_ablation_completion_receipt.json"
)
BASELINE_EXPOSURE_RECEIPT_SCHEMA = "pivot.stageb.b58_exposure_derivation_receipt/v1"
BASELINE_EXPOSURE_RECEIPT_PATH = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/b58_exposure_derivation_receipt.json"
)
COMPLETION_BLOCKS = ("A", "B", "C", "D", "G0c")
BASELINE_RECORD_MAX_ABS_TOLERANCE = 1e-6

RECEIPT_ARTIFACT_NAMES = (
    "selection_receipt",
    "baseline_contract",
    "candidate_contract",
    "evaluation_parity_contract",
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")

M0_COMPLETE_STATE_COMPONENTS = {
    "model": True,
    "criterion": True,
    "optimizer": True,
    "lr_scheduler": True,
    "scaler": True,
    "epoch": True,
    "iteration": True,
    "optimizer_updates": True,
    "epoch_finished": True,
    "rng_state": True,
    "epoch_rng_state": True,
    "args": True,
}


class HeadlineReleaseError(RuntimeError):
    """Raised when headline release provenance is missing or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
        raise HeadlineReleaseError(f"receipt artifact is not a file: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _verify_file_record(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HeadlineReleaseError(f"{label} is not an artifact record")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HeadlineReleaseError(f"{label} has no path")
    path = Path(raw_path).expanduser().resolve(strict=True)
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise HeadlineReleaseError(f"{label} path is not canonical")
    observed = file_record(path)
    expected_sha = str(value.get("sha256", ""))
    try:
        expected_size = int(value.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise HeadlineReleaseError(f"{label} has invalid size_bytes") from exc
    if (
        _SHA_RE.fullmatch(expected_sha) is None
        or expected_sha != observed["sha256"]
        or expected_size != observed["size_bytes"]
    ):
        raise HeadlineReleaseError(f"{label} digest/size changed")
    return observed


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HeadlineReleaseError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HeadlineReleaseError(f"{label} must be a JSON object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def _self_bound_payload(
    path: Path,
    *,
    schema: str,
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    payload = _read_json(path, label=label)
    expected = str(payload.pop(hash_field, ""))
    if _SHA_RE.fullmatch(expected) is None or expected != canonical_json_sha256(payload):
        raise HeadlineReleaseError(f"{label} self SHA-256 mismatch")
    payload[hash_field] = expected
    if payload.get("schema") != schema:
        raise HeadlineReleaseError(f"{label} schema mismatch")
    return payload


def canonical_validation_root(role: str, seed: int | None = None) -> Path:
    if role == "baseline":
        if seed not in (None, BASELINE_TRAIN_SEED):
            raise HeadlineReleaseError("fixed b58 validation has only seed42")
        return VALIDATION_ROOT / BASELINE_ID / "fixed"
    if role == "candidate" and seed in CANDIDATE_SEEDS:
        return VALIDATION_ROOT / CANDIDATE_ID / f"seed{int(seed)}"
    raise HeadlineReleaseError("unknown headline validation instance")


def canonical_final_root(role: str, seed: int | None = None) -> Path:
    if role == "baseline":
        if seed not in (None, BASELINE_TRAIN_SEED):
            raise HeadlineReleaseError("fixed b58 final has only seed42")
        return FINAL_RELEASE_ROOT / BASELINE_ID / "fixed"
    if role == "candidate" and seed in CANDIDATE_SEEDS:
        return FINAL_RELEASE_ROOT / CANDIDATE_ID / f"seed{int(seed)}"
    raise HeadlineReleaseError("unknown headline final instance")


def canonical_candidate_training_root(seed: int) -> Path:
    if seed not in CANDIDATE_SEEDS:
        raise HeadlineReleaseError(f"unexpected M0 seed {seed}")
    return (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/headline_main_compute_matched"
        / CANDIDATE_ID
        / f"seed{seed}"
    )


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
        raise HeadlineReleaseError(
            "source is not the fixed b58 historical checkpoint/config identity"
        )
    return {
        "id": expected["id"],
        "train_seed": int(expected["train_seed"]),
        "role": expected["role"],
        "config": file_record(config),
        "checkpoint": file_record(checkpoint),
    }


def _fingerprint_records(
    records: Iterable[Mapping[str, Any]], roles: set[str]
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise HeadlineReleaseError(f"input record {index} is invalid")
        record_roles = record.get("roles")
        if not isinstance(record_roles, list):
            raise HeadlineReleaseError(f"input record {index} has no role list")
        if not roles.intersection(str(value) for value in record_roles):
            continue
        path = Path(str(record.get("path", ""))).expanduser().resolve(strict=False)
        sha = str(record.get("sha256", ""))
        try:
            size = int(record.get("size_bytes"))
        except (TypeError, ValueError) as exc:
            raise HeadlineReleaseError(
                f"input record {index} has invalid size"
            ) from exc
        if _SHA_RE.fullmatch(sha) is None:
            raise HeadlineReleaseError(f"input record {index} has invalid SHA-256")
        selected.append({"path": str(path), "sha256": sha, "size_bytes": size})
    selected.sort(key=lambda value: value["path"])
    if not selected:
        raise HeadlineReleaseError(f"input closure for roles {sorted(roles)} is empty")
    if len({value["path"] for value in selected}) != len(selected):
        raise HeadlineReleaseError("input closure contains duplicate paths")
    return {
        "algorithm": "sha256_canonical_path_content_size_v1",
        "digest": canonical_json_sha256({"records": selected}),
        "records": selected,
    }


def common_evaluation_fingerprints(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    records = list(records)
    return {
        "code": _fingerprint_records(records, {"evaluation_code_dependency"}),
        "data": _fingerprint_records(records, {"evaluation_data_input"}),
    }


def runtime_projection(runtime: Mapping[str, Any]) -> dict[str, Any]:
    required = tuple(CANONICAL_RUNTIME)
    missing = [key for key in required if key not in runtime]
    if missing:
        raise HeadlineReleaseError(f"evaluation runtime lacks fields {missing}")
    projection = {key: runtime[key] for key in required}
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
    expected["python"] = str(Path(expected["python"]).resolve(strict=True))
    expected["data_root"] = str(Path(expected["data_root"]).resolve(strict=True))
    if projection != expected:
        raise HeadlineReleaseError(
            f"headline evaluation runtime drifted: expected {expected}, got {projection}"
        )
    return projection


def evaluate_selection_eligibility(
    baseline: Mapping[str, Any],
    candidates: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(candidates) != set(CANDIDATE_SEEDS):
        raise HeadlineReleaseError(
            f"selection requires exactly M0 seeds {CANDIDATE_SEEDS}"
        )
    baseline_ref = baseline.get("ref")
    if not isinstance(baseline_ref, Mapping) or set(baseline_ref) != set(
        VALIDATION_REF_SPLITS
    ):
        raise HeadlineReleaseError("baseline validation split set is not exact")
    candidate_means: dict[str, float] = {}
    split_deltas: dict[str, float] = {}
    for split in VALIDATION_REF_SPLITS:
        values = []
        for seed in CANDIDATE_SEEDS:
            ref = candidates[seed].get("ref")
            if not isinstance(ref, Mapping) or set(ref) != set(VALIDATION_REF_SPLITS):
                raise HeadlineReleaseError(
                    f"M0 seed{seed} validation split set is not exact"
                )
            values.append(float(ref[split]))
        if not all(math.isfinite(value) for value in values):
            raise HeadlineReleaseError("validation Ref metric is non-finite")
        candidate_means[split] = float(np.mean(values))
        split_deltas[split] = candidate_means[split] - float(baseline_ref[split])

    baseline_macro = float(
        np.mean([float(baseline_ref[split]) for split in VALIDATION_REF_SPLITS])
    )
    candidate_macro = float(
        np.mean([candidate_means[split] for split in VALIDATION_REF_SPLITS])
    )
    fpr_mean = float(
        np.mean([float(candidates[seed]["fpr95"]) for seed in CANDIDATE_SEEDS])
    )
    q05_mean = float(
        np.mean(
            [float(candidates[seed]["positive_q05"]) for seed in CANDIDATE_SEEDS]
        )
    )
    fpr_delta = fpr_mean - float(baseline["fpr95"])
    q05_delta = q05_mean - float(baseline["positive_q05"])
    finite = [
        baseline_macro,
        candidate_macro,
        fpr_mean,
        q05_mean,
        fpr_delta,
        q05_delta,
        *split_deltas.values(),
    ]
    if not all(math.isfinite(value) for value in finite):
        raise HeadlineReleaseError("selection metric is non-finite")

    split_pass = all(
        value >= SELECTION_THRESHOLDS["ref_split_mean_delta_min"]
        for value in split_deltas.values()
    )
    macro_delta = candidate_macro - baseline_macro
    macro_pass = (
        macro_delta
        > SELECTION_THRESHOLDS["ref_val3_macro_mean_delta_strict_min"]
    )
    fpr_pass = (
        fpr_delta
        < SELECTION_THRESHOLDS["calibration_fpr95_mean_delta_strict_max"]
    )
    q05_pass = (
        q05_delta
        >= SELECTION_THRESHOLDS["calibration_positive_q05_mean_delta_min"]
    )
    gates = {
        "ref_split_noninferiority": {
            "passed": split_pass,
            "threshold": SELECTION_THRESHOLDS["ref_split_mean_delta_min"],
            "candidate_seed_mean_minus_baseline": split_deltas,
        },
        "ref_val3_macro_improvement": {
            "passed": macro_pass,
            "strict_threshold": SELECTION_THRESHOLDS[
                "ref_val3_macro_mean_delta_strict_min"
            ],
            "candidate_seed_mean_minus_baseline": macro_delta,
        },
        "calibration_fpr95_improvement": {
            "passed": fpr_pass,
            "strict_threshold": SELECTION_THRESHOLDS[
                "calibration_fpr95_mean_delta_strict_max"
            ],
            "candidate_seed_mean_minus_baseline": fpr_delta,
        },
        "calibration_positive_q05_noninferiority": {
            "passed": q05_pass,
            "threshold": SELECTION_THRESHOLDS[
                "calibration_positive_q05_mean_delta_min"
            ],
            "candidate_seed_mean_minus_baseline": q05_delta,
        },
    }
    return {
        "passed": all(value["passed"] for value in gates.values()),
        "candidate_id": CANDIDATE_ID,
        "candidate_seeds": list(CANDIDATE_SEEDS),
        "selection_policy": SELECTION_POLICY,
        "fallback_policy": "none_close_final_gate",
        "aggregation": "point_mean_across_three_training_seeds",
        "baseline_metrics": {
            "ref": {split: float(baseline_ref[split]) for split in VALIDATION_REF_SPLITS},
            "ref_val3_macro": baseline_macro,
            "fpr95": float(baseline["fpr95"]),
            "positive_q05": float(baseline["positive_q05"]),
        },
        "candidate_seed_mean_metrics": {
            "ref": candidate_means,
            "ref_val3_macro": candidate_macro,
            "fpr95": fpr_mean,
            "positive_q05": q05_mean,
        },
        "gates": gates,
    }


def _normalized_replay(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("validated_at_utc", None)
    result.pop("verified_at_utc", None)
    input_rehash = result.get("input_rehash")
    if isinstance(input_rehash, dict):
        input_rehash.pop("verified_at_utc", None)
    return result


def _command_option(command: Sequence[Any], option: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        return None
    return str(command[positions[0] + 1])


def _parse_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise HeadlineReleaseError(f"{label} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HeadlineReleaseError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise HeadlineReleaseError(f"{label} is not timezone-aware")
    return parsed


def _m0_attempt_record(
    value: Any,
    *,
    run_root: Path,
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_path = (
        run_root / "attempts" / f"{ordinal:03d}" / "attempt_manifest.json"
    ).resolve(strict=True)
    observed = _verify_file_record(
        value,
        label=f"M0 attempt {ordinal} manifest",
        expected_path=expected_path,
    )
    payload = _read_json(expected_path, label=f"M0 attempt {ordinal} manifest")
    return observed, payload


def _require_m0_mid_epoch_signal(value: Any, *, label: str) -> str:
    if value != "signal":
        raise HeadlineReleaseError(f"{label} must be a mid-epoch signal checkpoint")
    return "signal"


def _validate_m0_attempt_telemetry(
    value: Any,
    *,
    run_root: Path,
    ordinal: int,
) -> None:
    expected_keys = {
        "schema",
        "status",
        "attempt_ordinal",
        "sampling_interval_ms",
        "sample_rows",
        "devices",
        "artifacts",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("schema") != M0_ATTEMPT_TELEMETRY_SCHEMA
        or value.get("status") != "sealed"
        or value.get("attempt_ordinal") != ordinal
        or value.get("sampling_interval_ms") != 1000
        or type(value.get("sample_rows")) is not int
        or int(value["sample_rows"]) <= 0
        or not isinstance(value.get("devices"), list)
        or not value["devices"]
    ):
        raise HeadlineReleaseError(f"M0 attempt {ordinal} telemetry drifted")
    artifacts = value.get("artifacts")
    filenames = {
        "gpu_environment": "gpu_environment.json",
        "gpu_telemetry": "gpu_telemetry.csv",
        "gpu_telemetry_summary": "gpu_telemetry_summary.json",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(filenames):
        raise HeadlineReleaseError(f"M0 attempt {ordinal} telemetry artifacts drifted")
    attempt_dir = run_root / "attempts" / f"{ordinal:03d}"
    for name, filename in filenames.items():
        _verify_file_record(
            artifacts.get(name),
            label=f"M0 attempt {ordinal} {name}",
            expected_path=attempt_dir / filename,
        )


def _stable_training_input_closure(records: Any) -> dict[str, Any]:
    if not isinstance(records, list) or not records:
        raise HeadlineReleaseError("M0 stable training input closure is empty")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise HeadlineReleaseError(f"M0 stable input {index} is invalid")
        raw_roles = record.get("roles")
        if isinstance(raw_roles, list):
            roles = sorted({str(value) for value in raw_roles})
        elif isinstance(record.get("role"), str):
            roles = [str(record["role"])]
        else:
            raise HeadlineReleaseError(f"M0 stable input {index} has no role")
        if not roles:
            raise HeadlineReleaseError(f"M0 stable input {index} role set is empty")
        if set(roles).intersection(
            {
                "resume_checkpoint",
                "recovery_checkpoint",
                "attempt_manifest",
                "attempt_input_closure",
                "training_attempt_specific",
            }
        ):
            raise HeadlineReleaseError(
                "attempt-specific ancestry leaked into M0 stable input closure"
            )
        observed = _verify_file_record(record, label=f"M0 stable input {index}")
        normalized.append({**observed, "roles": roles})
    normalized.sort(key=lambda value: (value["path"], value["roles"]))
    identities = [(value["path"], tuple(value["roles"])) for value in normalized]
    if len(identities) != len(set(identities)):
        raise HeadlineReleaseError("M0 stable input closure has duplicate identities")
    digest = canonical_json_sha256(
        {"schema": M0_STABLE_CLOSURE_SCHEMA, "records": normalized}
    )
    return {"digest": digest, "records": normalized}


def _validate_m0_stable_closure(
    value: Any,
    *,
    run_root: Path,
    ordinal: int,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    expected_path = (
        run_root / "attempts" / f"{ordinal:03d}" / "input_closure.json"
    ).resolve(strict=True)
    record = _verify_file_record(
        value,
        label=f"M0 attempt {ordinal} stable input closure",
        expected_path=expected_path,
    )
    payload = _read_json(
        expected_path, label=f"M0 attempt {ordinal} stable input closure"
    )
    if set(payload) != {"schema", "status", "algorithm", "records", "digest"}:
        raise HeadlineReleaseError("M0 stable input closure field set drifted")
    replay = _stable_training_input_closure(payload.get("records"))
    if (
        payload.get("schema") != M0_STABLE_CLOSURE_SCHEMA
        or payload.get("status") != "sealed"
        or payload.get("algorithm")
        != "sha256_canonical_path_content_size_roles_v1"
        or payload.get("records") != replay["records"]
        or payload.get("digest") != replay["digest"]
        or replay != expected
    ):
        raise HeadlineReleaseError("M0 stable input closure replay drifted")
    return record


def _inspect_m0_training_checkpoint(path: Path) -> dict[str, Any]:
    """Delegate trusted checkpoint loading to the future strict M0 runner."""

    try:
        from tools import run_stageb_headline_m0 as m0_runner
    except ImportError as exc:
        raise HeadlineReleaseError(
            "M0 checkpoint inspector adapter is unsealed"
        ) from exc
    inspector = getattr(m0_runner, "inspect_training_checkpoint_for_release", None)
    if not callable(inspector):
        raise HeadlineReleaseError("M0 checkpoint inspector adapter is unsealed")
    try:
        value = inspector(path)
    except Exception as exc:
        raise HeadlineReleaseError(f"M0 checkpoint safe inspection failed: {exc}") from exc
    if not isinstance(value, Mapping):
        raise HeadlineReleaseError("M0 checkpoint inspector returned no mapping")
    return dict(value)


def _validate_m0_ancestry(
    ancestry: Any,
    scorer_initializer_audit: Any,
    *,
    run_id: str,
    seed: int,
    run_root: Path,
    stage_a_path: Path,
    stage_a_sha256: str,
    final_checkpoint: Mapping[str, Any],
    scorer_audit_artifact: Mapping[str, Any],
    stable_input_closure: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ancestry_keys = {
        "schema",
        "fresh_start",
        "resume_ancestry",
        "ultimate_pretrain",
        "ultimate_scorer",
        "ultimate_same_stage_a_source",
        "resume_chain_contiguous",
        "b58_ancestry_count",
        "b58_ancestry_paths",
        "b58_ancestry_sha256s",
    }
    if not isinstance(ancestry, Mapping) or set(ancestry) != expected_ancestry_keys:
        raise HeadlineReleaseError(f"{run_id} M0 ancestry field set drifted")
    rendered = json.dumps(ancestry, sort_keys=True)
    if (
        ancestry.get("schema") != M0_ANCESTRY_SCHEMA
        or ancestry.get("ultimate_same_stage_a_source") is not True
        or ancestry.get("resume_chain_contiguous") is not True
        or ancestry.get("b58_ancestry_count") != 0
        or ancestry.get("b58_ancestry_paths") != []
        or ancestry.get("b58_ancestry_sha256s") != []
        or FIXED_BASELINE["checkpoint_sha256"] in rendered
        or str(FIXED_BASELINE["checkpoint"]) in rendered
    ):
        raise HeadlineReleaseError(f"{run_id} ancestry is not Stage-A-only/no-b58")

    stage_a_record = file_record(stage_a_path)
    if stage_a_record["sha256"] != stage_a_sha256:
        raise HeadlineReleaseError(f"{run_id} Stage-A initializer digest drifted")
    expected_pretrain = {**stage_a_record, "role": "stage_a_initializer"}
    expected_scorer = {**stage_a_record, "role": "scorer_warmstart"}
    fresh = ancestry.get("fresh_start")
    if not isinstance(fresh, Mapping) or set(fresh) != {
        "run_id",
        "initialization_mode",
        "pretrain",
        "scorer",
        "same_source",
        "resume_argument",
        "attempt_manifest",
    }:
        raise HeadlineReleaseError(f"{run_id} fresh-start ancestry is incomplete")
    if (
        fresh.get("run_id") != run_id
        or fresh.get("initialization_mode")
        != "pretrain_model_path_plus_same_source_scorer_init"
        or fresh.get("pretrain") != expected_pretrain
        or fresh.get("scorer") != expected_scorer
        or fresh.get("same_source") is not True
        or fresh.get("resume_argument") is not None
        or ancestry.get("ultimate_pretrain")
        != {
            "path": stage_a_record["path"],
            "sha256": stage_a_sha256,
            "role": "stage_a_initializer",
        }
        or ancestry.get("ultimate_scorer")
        != {
            "path": stage_a_record["path"],
            "sha256": stage_a_sha256,
            "role": "scorer_warmstart",
        }
    ):
        raise HeadlineReleaseError(f"{run_id} fresh/ultimate Stage-A ancestry drifted")

    expected_scorer_wrapper_keys = {
        "status",
        "applied",
        "source_path",
        "source_sha256",
        "loaded_tensor_count",
        "loaded_num_layers",
        "artifact",
        "same_as_stage_a_initializer",
        "b58_source",
    }
    if (
        not isinstance(scorer_initializer_audit, Mapping)
        or set(scorer_initializer_audit) != expected_scorer_wrapper_keys
        or scorer_initializer_audit.get("status") != "passed"
        or scorer_initializer_audit.get("applied") is not True
        or Path(str(scorer_initializer_audit.get("source_path", ""))).resolve(
            strict=True
        )
        != stage_a_path
        or scorer_initializer_audit.get("source_sha256") != stage_a_sha256
        or scorer_initializer_audit.get("loaded_tensor_count") != 90
        or scorer_initializer_audit.get("loaded_num_layers") != 3
        or scorer_initializer_audit.get("artifact") != scorer_audit_artifact
        or scorer_initializer_audit.get("same_as_stage_a_initializer") is not True
        or scorer_initializer_audit.get("b58_source") is not False
    ):
        raise HeadlineReleaseError(f"{run_id} scorer initializer wrapper drifted")
    _verify_file_record(
        scorer_audit_artifact,
        label=f"{run_id} scorer initializer artifact",
    )
    scorer_payload = _read_json(
        Path(str(scorer_audit_artifact["path"])),
        label=f"{run_id} scorer initializer artifact",
    )
    if (
        scorer_payload.get("schema") != "stage_b_v15_scorer_init/v1"
        or scorer_payload.get("status") != "applied"
        or scorer_payload.get("source_sha256") != stage_a_sha256
        or scorer_payload.get("loaded_tensor_count") != 90
        or scorer_payload.get("loaded_num_layers") != 3
        or Path(str(scorer_payload.get("resolved_source_path", ""))).resolve(
            strict=True
        )
        != stage_a_path
        or FIXED_BASELINE["checkpoint_sha256"]
        in json.dumps(scorer_payload, sort_keys=True)
    ):
        raise HeadlineReleaseError(f"{run_id} scorer initializer audit replay drifted")

    resume_edges = ancestry.get("resume_ancestry")
    if not isinstance(resume_edges, list):
        raise HeadlineReleaseError(f"{run_id} resume ancestry is not a list")
    fresh_record, fresh_attempt = _m0_attempt_record(
        fresh.get("attempt_manifest"),
        run_root=run_root,
        ordinal=0,
    )
    attempt_records = [fresh_record]
    attempts = [fresh_attempt]
    edge_records: list[dict[str, Any]] = []
    for ordinal, edge in enumerate(resume_edges, start=1):
        if not isinstance(edge, Mapping) or set(edge) != {
            "ordinal",
            "run_id",
            "source_checkpoint",
            "source_optimizer_updates",
            "source_checkpoint_reason",
            "complete_training_state",
            "same_run",
            "resume_authorization",
            "attempt_manifest",
        }:
            raise HeadlineReleaseError(f"{run_id} resume edge {ordinal} drifted")
        _require_m0_mid_epoch_signal(
            edge.get("source_checkpoint_reason"),
            label=f"{run_id} resume edge {ordinal} source reason",
        )
        if (
            edge.get("ordinal") != ordinal
            or edge.get("run_id") != run_id
            or edge.get("complete_training_state") is not True
            or edge.get("same_run") is not True
        ):
            raise HeadlineReleaseError(f"{run_id} resume edge {ordinal} is invalid")
        try:
            source_updates = int(edge.get("source_optimizer_updates"))
        except (TypeError, ValueError) as exc:
            raise HeadlineReleaseError(
                f"{run_id} resume edge {ordinal} update is invalid"
            ) from exc
        source_record = _verify_file_record(
            edge.get("source_checkpoint"),
            label=f"{run_id} resume edge {ordinal} checkpoint",
        )
        source_path = Path(source_record["path"])
        expected_name = (
            f"attempt_{ordinal:03d}_from_u{source_updates:06d}_"
            f"{source_record['sha256'][:12]}.pth"
        )
        if (
            source_path.parent != (run_root / "recovery").resolve(strict=True)
            or source_path.name != expected_name
        ):
            raise HeadlineReleaseError(
                f"{run_id} resume edge {ordinal} checkpoint is not canonical"
            )
        authorization_path = (
            run_root / "control" / "resume_requests" / f"{ordinal:03d}.json"
        ).resolve(strict=True)
        authorization_record = _verify_file_record(
            edge.get("resume_authorization"),
            label=f"{run_id} resume edge {ordinal} authorization",
            expected_path=authorization_path,
        )
        authorization = _read_json(
            authorization_path,
            label=f"{run_id} resume edge {ordinal} authorization",
        )
        if (
            set(authorization)
            != {
                "schema",
                "status",
                "run_id",
                "next_attempt_ordinal",
                "recovery_checkpoint",
                "policy",
                "authorized_at_utc",
                "authorizer_pid",
                "detached_controller_identity",
            }
            or
            authorization.get("schema")
            != "pivot.stageb.headline_m0_resume_request/v1"
            or authorization.get("status") != "authorized"
            or authorization.get("run_id") != run_id
            or authorization.get("next_attempt_ordinal") != ordinal
            or authorization.get("recovery_checkpoint") != source_record
            or authorization.get("policy")
            != "explicit_one_attempt_mid_epoch_signal_resume"
            or not isinstance(authorization.get("authorized_at_utc"), str)
            or type(authorization.get("authorizer_pid")) is not int
            or int(authorization["authorizer_pid"]) <= 0
            or (
                authorization.get("detached_controller_identity") is not None
                and not isinstance(
                    authorization.get("detached_controller_identity"), Mapping
                )
            )
        ):
            raise HeadlineReleaseError(
                f"{run_id} resume edge {ordinal} authorization drifted"
            )
        authorization_time = _parse_utc(
            authorization["authorized_at_utc"],
            label=f"{run_id} resume edge {ordinal} authorization time",
        )
        attempt_record, attempt = _m0_attempt_record(
            edge.get("attempt_manifest"),
            run_root=run_root,
            ordinal=ordinal,
        )
        attempt_records.append(attempt_record)
        attempts.append(attempt)
        edge_records.append(
            {
                "source_checkpoint": source_record,
                "source_optimizer_updates": source_updates,
                "source_checkpoint_reason": edge["source_checkpoint_reason"],
                "resume_authorization": authorization_record,
                "authorization_time": authorization_time,
            }
        )

    attempt_keys = {
        "schema",
        "status",
        "run_id",
        "seed",
        "attempt_ordinal",
        "initialization_mode",
        "parent_attempt_manifest",
        "resume_checkpoint",
        "resume_authorization",
        "source_optimizer_updates",
        "target_optimizer_updates",
        "command",
        "command_shell",
        "runtime",
        "input_closure_digest",
        "input_closure",
        "telemetry",
        "process",
        "termination",
        "complete_state_components",
        "checkpoint_at_exit",
        "checkpoint_metadata",
        "started_at_utc",
        "finished_at_utc",
    }
    previous_exit_updates = 0
    previous_finish: datetime | None = None
    closure_digest: str | None = None
    closure_records: list[dict[str, Any]] | None = None
    final_checkpoint_record = dict(final_checkpoint)
    for ordinal, (attempt_record, attempt) in enumerate(
        zip(attempt_records, attempts)
    ):
        if not isinstance(attempt, Mapping) or set(attempt) != attempt_keys:
            raise HeadlineReleaseError(f"{run_id} attempt {ordinal} field set drifted")
        command = attempt.get("command")
        runtime = attempt.get("runtime")
        process = attempt.get("process")
        termination = attempt.get("termination")
        metadata = attempt.get("checkpoint_metadata")
        if (
            attempt.get("schema") != M0_ATTEMPT_SCHEMA
            or attempt.get("status") != "completed"
            or attempt.get("run_id") != run_id
            or attempt.get("seed") != seed
            or attempt.get("attempt_ordinal") != ordinal
            or attempt.get("target_optimizer_updates") != CANDIDATE_UPDATES
            or attempt.get("complete_state_components")
            != M0_COMPLETE_STATE_COMPONENTS
            or not isinstance(command, list)
            or not all(isinstance(value, str) for value in command)
            or attempt.get("command_shell") != shlex.join(command)
            or not isinstance(runtime, Mapping)
            or runtime.get("batch_size") != CANDIDATE_BATCH_SIZE
            or runtime.get("num_workers") != 2
            or runtime.get("prefetch_factor") != 1
            or runtime.get("amp") is not True
            or runtime.get("gradient_accumulation_steps") != 1
            or runtime.get("max_train_iters") != CANDIDATE_UPDATES
            or runtime.get("iter_checkpoint_interval") != 500
            or runtime.get("gradient_diagnostic_interval") != 100
            or runtime.get("telemetry_interval_seconds") != 1
            or runtime.get("pin_memory") is not True
            or runtime.get("persistent_workers") is not False
            or not isinstance(process, Mapping)
            or process.get("returncode") != 0
            or not isinstance(termination, Mapping)
            or not isinstance(metadata, Mapping)
        ):
            raise HeadlineReleaseError(f"{run_id} attempt {ordinal} contract drifted")
        _validate_m0_attempt_telemetry(
            attempt.get("telemetry"), run_root=run_root, ordinal=ordinal
        )
        digest = attempt.get("input_closure_digest")
        if _SHA_RE.fullmatch(str(digest)) is None:
            raise HeadlineReleaseError(f"{run_id} attempt closure digest is invalid")
        if closure_digest is None:
            closure_digest = str(digest)
        elif digest != closure_digest:
            raise HeadlineReleaseError(f"{run_id} attempt input closure forked")
        _validate_m0_stable_closure(
            attempt.get("input_closure"),
            run_root=run_root,
            ordinal=ordinal,
            expected=stable_input_closure,
        )
        if digest != stable_input_closure.get("digest"):
            raise HeadlineReleaseError(
                f"{run_id} attempt input closure digest is not launch-bound"
            )
        current_records = stable_input_closure.get("records")
        if not isinstance(current_records, list):
            raise HeadlineReleaseError(f"{run_id} stable closure records are missing")
        if closure_records is None:
            closure_records = list(current_records)
        elif current_records != closure_records:
            raise HeadlineReleaseError(f"{run_id} stable closure records forked")
        started = _parse_utc(
            attempt.get("started_at_utc"), label=f"{run_id} attempt start"
        )
        finished = _parse_utc(
            attempt.get("finished_at_utc"), label=f"{run_id} attempt finish"
        )
        parent_finished = previous_finish
        if finished < started or (
            parent_finished is not None and started < parent_finished
        ):
            raise HeadlineReleaseError(f"{run_id} attempt chronology forked")
        previous_finish = finished

        source_updates = attempt.get("source_optimizer_updates")
        if type(source_updates) is not int or source_updates != previous_exit_updates:
            raise HeadlineReleaseError(
                f"{run_id} attempt {ordinal} source update is not contiguous"
            )
        scorer_option = f"stage_b_v15_scorer_init_checkpoint={stage_a_path}"
        if command.count(scorer_option) != 1:
            raise HeadlineReleaseError(
                f"{run_id} attempt {ordinal} scorer source drifted"
            )
        if ordinal == 0:
            if (
                attempt.get("initialization_mode") != "fresh_stage_a"
                or attempt.get("parent_attempt_manifest") is not None
                or attempt.get("resume_checkpoint") is not None
                or attempt.get("resume_authorization") is not None
                or _command_option(command, "--pretrain_model_path")
                != str(stage_a_path)
                or "--resume" in command
            ):
                raise HeadlineReleaseError(f"{run_id} fresh attempt is not Stage-A")
        else:
            edge = resume_edges[ordinal - 1]
            authorization_time = edge_records[ordinal - 1]["authorization_time"]
            if (
                attempt.get("initialization_mode") != "same_run_resume"
                or attempt.get("parent_attempt_manifest")
                != attempt_records[ordinal - 1]
                or attempt.get("resume_checkpoint") != edge["source_checkpoint"]
                or attempt.get("resume_authorization")
                != edge_records[ordinal - 1]["resume_authorization"]
                or _command_option(command, "--resume")
                != edge_records[ordinal - 1]["source_checkpoint"]["path"]
                or "--pretrain_model_path" in command
                or parent_finished is None
                or not parent_finished <= authorization_time <= started
            ):
                raise HeadlineReleaseError(
                    f"{run_id} attempt {ordinal} is not a contiguous same-run resume"
                )

        checkpoint_at_exit = _verify_file_record(
            attempt.get("checkpoint_at_exit"),
            label=f"{run_id} attempt {ordinal} exit checkpoint",
        )
        exit_updates = metadata.get("optimizer_updates")
        if type(exit_updates) is not int or not previous_exit_updates < exit_updates:
            raise HeadlineReleaseError(
                f"{run_id} attempt {ordinal} optimizer updates are not monotonic"
            )
        expected_checkpoint_metadata = {
            "optimizer_updates": exit_updates,
            "optimizer_state_count": 94,
            "optimizer_step_values": [exit_updates],
            "complete_state_components": M0_COMPLETE_STATE_COMPONENTS,
            "checkpoint_reason": metadata.get("checkpoint_reason"),
        }
        if (
            set(metadata) != set(expected_checkpoint_metadata)
            or metadata.get("optimizer_updates") != exit_updates
            or metadata.get("optimizer_state_count") != 94
            or metadata.get("optimizer_step_values") != [exit_updates]
            or metadata.get("complete_state_components")
            != M0_COMPLETE_STATE_COMPONENTS
        ):
            raise HeadlineReleaseError(
                f"{run_id} attempt {ordinal} checkpoint state is incomplete"
            )
        inspected_metadata = _inspect_m0_training_checkpoint(
            Path(checkpoint_at_exit["path"])
        )
        if inspected_metadata != dict(metadata):
            raise HeadlineReleaseError(
                f"{run_id} attempt {ordinal} checkpoint metadata differs from safe-load replay"
            )
        is_final = ordinal == len(attempts) - 1
        if is_final:
            if (
                termination.get("kind") != "target_completed"
                or termination.get("reason") != "max_train_iters"
                or metadata.get("checkpoint_reason") != "max_train_iters"
                or exit_updates != CANDIDATE_UPDATES
                or checkpoint_at_exit != final_checkpoint_record
            ):
                raise HeadlineReleaseError(f"{run_id} final attempt is not U23532")
        else:
            edge = edge_records[ordinal]
            _require_m0_mid_epoch_signal(
                termination.get("reason"),
                label=f"{run_id} attempt {ordinal} termination reason",
            )
            if (
                termination.get("kind") != "graceful_signal_checkpoint"
                or metadata.get("checkpoint_reason") != termination.get("reason")
                or edge["source_checkpoint_reason"] != termination.get("reason")
                or edge["source_optimizer_updates"] != exit_updates
                or checkpoint_at_exit != edge["source_checkpoint"]
                or exit_updates >= CANDIDATE_UPDATES
            ):
                raise HeadlineReleaseError(
                    f"{run_id} attempt {ordinal} recovery edge is not sealed"
                )
        previous_exit_updates = exit_updates

    return {
        "status": "passed",
        "attempt_count": len(attempts),
        "resume_count": len(resume_edges),
        "attempt_manifests": attempt_records,
        "stable_input_closure_digest": closure_digest,
        "ultimate_stage_a_sha256": stage_a_sha256,
        "b58_ancestry_count": 0,
        "final_optimizer_updates": previous_exit_updates,
    }


def _candidate_source_projection(source: Mapping[str, Any], seed: int) -> dict[str, Any]:
    run_id = f"{CANDIDATE_ID}:{seed}"
    expected_root = canonical_candidate_training_root(seed).resolve(strict=True)
    if (
        source.get("kind") != "pivot_paper_training_run"
        or source.get("evaluation_id") != f"{CANDIDATE_ID}_seed{seed}"
        or source.get("training_run_id") != run_id
        or int(source.get("training_seed", -1)) != seed
        or source.get("training_phase") != "final"
        or source.get("diagnostic_only") is not False
        or Path(str(source.get("training_run_root", ""))).resolve(strict=True)
        != expected_root
    ):
        raise HeadlineReleaseError(
            f"headline candidate is not the fixed formal {run_id} source"
        )
    queue_fields = (
        "training_queue_manifest",
        "training_queue_detached_launch",
        "training_queue_detached_status",
        "training_queue_id",
        "training_queue_plan_sha256",
    )
    if any(not source.get(field) for field in queue_fields):
        raise HeadlineReleaseError(
            f"{run_id} lacks a complete formal training queue attestation"
        )
    queue_plan_sha = str(source["training_queue_plan_sha256"])
    if _SHA_RE.fullmatch(queue_plan_sha) is None:
        raise HeadlineReleaseError(f"{run_id} training queue plan SHA is invalid")
    checkpoint = Path(str(source.get("checkpoint", ""))).resolve(strict=True)
    config = Path(str(source.get("config", ""))).resolve(strict=True)
    checkpoint_sha = str(source.get("checkpoint_sha256", ""))
    if (
        _SHA_RE.fullmatch(checkpoint_sha) is None
        or sha256_file(checkpoint) != checkpoint_sha
        or config
        != (
            REPO_ROOT / "config/ablations/cfg_stageb_v25_m0_compute_matched.py"
        ).resolve(strict=True)
        or sha256_file(config) != M0_CONFIG_SHA256
    ):
        raise HeadlineReleaseError(f"{run_id} checkpoint identity changed")
    try:
        config_values = runpy.run_path(str(config))
    except (OSError, RuntimeError, ImportError, SyntaxError) as exc:
        raise HeadlineReleaseError(f"{run_id} config cannot be evaluated: {exc}") from exc
    if (
        config_values.get("stage_b_v22_table_id") != "S2F"
        or config_values.get("stage_b_v25_main_id") != "M0"
        or config_values.get("stage_b_v25_compute_contract")
        != "b58_successful_update_batch_slot_matched"
        or config_values.get("stage_b_v25_budget_unit")
        != "successful_optimizer_update_global_batch_slots"
        or config_values.get("stage_b_v25_successful_update_batch_slots")
        != CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        or config_values.get("stage_b_v25_initializer_contract")
        != "same_stage_a_model_and_scorer_no_b58"
        or config_values.get("stage_b_v25_strict_resume") is not True
        or config_values.get("stage_b_v15_separate_grad_clip") is not True
        or config_values.get("stage_b_v22_gradient_diagnostic_interval") != 100
        or Path(
            str(config_values.get("stage_b_v15_scorer_init_checkpoint", ""))
        ).resolve(strict=True)
        != Path("/media/haoyi/T9/gdino/checkpoint0004.pth").resolve(strict=True)
    ):
        raise HeadlineReleaseError(
            f"{run_id} config is not the M0 leaf over the S2F objective"
        )
    queue_manifest = Path(str(source["training_queue_manifest"])).resolve(strict=True)
    if queue_manifest.name != "queue.json":
        raise HeadlineReleaseError(f"{run_id} queue manifest path is not canonical")
    sequence_path = Path(str(source.get("sequence_manifest", ""))).resolve(
        strict=True
    )
    if sequence_path != (expected_root / "sequence_manifest.json").resolve(
        strict=True
    ):
        raise HeadlineReleaseError(f"{run_id} sequence manifest path is not canonical")
    sequence = _read_json(sequence_path, label=f"{run_id} sequence manifest")
    row = sequence.get("row")
    expected_budget = {
        "batch_size": CANDIDATE_BATCH_SIZE,
        "optimizer_updates": CANDIDATE_UPDATES,
        "contributing_phase_updates": {"joint": CANDIDATE_UPDATES},
        "successful_update_batch_slots": (
            CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        ),
    }
    if (
        sequence.get("status") != "completed"
        or sequence.get("run_id") != run_id
        or int(sequence.get("seed", -1)) != seed
        or sequence.get("equal_budget_contract") != expected_budget
        or not isinstance(row, Mapping)
        or row.get("row_id") != CANDIDATE_ID
        or row.get("architecture_objective")
        != CANDIDATE_ARCHITECTURE_OBJECTIVE
        or row.get("compute_contract")
        != "b58_successful_update_batch_slot_matched"
        or row.get("datasets")
        != "config/datasets_stageb_v21_single_edit_train.json"
        or row.get("config")
        != "config/ablations/cfg_stageb_v25_m0_compute_matched.py"
    ):
        raise HeadlineReleaseError(
            f"{run_id} is not the successful-update batch-slot-matched "
            "B40/U23532 M0(S2F) main run"
        )
    selected_launch_path = Path(
        str(source.get("selected_phase_manifest", ""))
    ).resolve(strict=True)
    selected_launch = _read_json(
        selected_launch_path, label=f"{run_id} final training launch"
    )
    phase = selected_launch.get("phase")
    training_runtime = selected_launch.get("runtime")
    training_inputs = selected_launch.get("inputs")
    input_records = (
        training_inputs.get("records")
        if isinstance(training_inputs, Mapping)
        else None
    )
    stage_a_path = Path("/media/haoyi/T9/gdino/checkpoint0004.pth").resolve(
        strict=True
    )
    stage_a_sha = (
        "7f4cdd0ab94fc74d46fc7658b2014588a06d7de44be2c1d482ed073bbd7ca1b1"
    )

    def training_roles(value: Mapping[str, Any]) -> set[str]:
        roles = set()
        if isinstance(value.get("role"), str):
            roles.add(str(value["role"]))
        if isinstance(value.get("roles"), list):
            roles.update(str(item) for item in value["roles"])
        return roles

    stage_a_records = [
        value
        for value in input_records or []
        if isinstance(value, Mapping)
        and "stage_a_initializer" in training_roles(value)
        and value.get("sha256") == stage_a_sha
        and Path(str(value.get("path", ""))).resolve(strict=True) == stage_a_path
    ]
    scorer_init_records = [
        value
        for value in input_records or []
        if isinstance(value, Mapping)
        and "scorer_warmstart" in training_roles(value)
        and value.get("sha256") == stage_a_sha
        and Path(str(value.get("path", ""))).resolve(strict=True) == stage_a_path
    ]
    forbidden_model_roles = {
        "stage_a_initializer",
        "scorer_warmstart",
        "pretrain_model",
        "resume_checkpoint",
    }
    if any(
        isinstance(value, Mapping)
        and training_roles(value).intersection(forbidden_model_roles)
        and value.get("sha256") == FIXED_BASELINE["checkpoint_sha256"]
        for value in input_records or []
    ):
        raise HeadlineReleaseError(
            f"{run_id} model/scorer ancestry illegally contains b58"
        )
    stable_input_closure = _stable_training_input_closure(input_records)
    if (
        training_inputs.get("stable_closure_digest")
        != stable_input_closure["digest"]
        or sequence.get("stable_input_closure_digest")
        != stable_input_closure["digest"]
    ):
        raise HeadlineReleaseError(
            f"{run_id} stable training input closure is not sequence/launch-bound"
        )
    if (
        selected_launch.get("status") != "completed"
        or not isinstance(phase, Mapping)
        or phase.get("phase_id") != "joint"
        or phase.get("config")
        != "config/ablations/cfg_stageb_v25_m0_compute_matched.py"
        or int(phase.get("updates", -1)) != CANDIDATE_UPDATES
        or int(phase.get("diagnostic_interval", -1)) != 100
        or not isinstance(training_runtime, Mapping)
        or training_runtime.get("amp") is not True
        or int(training_runtime.get("batch_size", -1)) != CANDIDATE_BATCH_SIZE
        or int(training_runtime.get("num_workers", -1)) != 2
        or int(training_runtime.get("prefetch_factor", -1)) != 1
        or training_runtime.get("pin_memory") is not True
        or training_runtime.get("persistent_workers") is not False
        or int(training_runtime.get("iter_checkpoint_interval", -1)) != 500
        or int(training_runtime.get("telemetry_interval_seconds", -1)) != 1
        or int(training_runtime.get("phase_train_iters", -1)) != CANDIDATE_UPDATES
        or int(training_runtime.get("total_paper_train_iters", -1))
        != CANDIDATE_UPDATES
        or len(stage_a_records) != 1
        or len(scorer_init_records) != 1
    ):
        raise HeadlineReleaseError(
            f"{run_id} final launch runtime/objective/stage-A contract drifted"
        )
    selected_postflight_path = Path(
        str(source.get("selected_training_postflight", ""))
    ).resolve(strict=True)
    selected_postflight = _read_json(
        selected_postflight_path, label=f"{run_id} final training postflight"
    )
    numerical = selected_postflight.get("numerical_status")
    telemetry = selected_postflight.get("gpu_telemetry_summary")
    progress = selected_postflight.get("optimizer_progress")
    ancestry = selected_postflight.get("model_state_ancestry")
    scorer_initializer_audit = selected_postflight.get(
        "scorer_initializer_audit"
    )
    artifacts = selected_postflight.get("artifacts")
    scorer_audit_record = (
        artifacts.get("scorer_init_audit") if isinstance(artifacts, Mapping) else None
    )
    lineage = _validate_m0_ancestry(
        ancestry,
        scorer_initializer_audit,
        run_id=run_id,
        seed=seed,
        run_root=expected_root,
        stage_a_path=stage_a_path,
        stage_a_sha256=stage_a_sha,
        final_checkpoint=file_record(checkpoint),
        scorer_audit_artifact=scorer_audit_record,
        stable_input_closure=stable_input_closure,
    )
    expected_progress = {
        "status": "passed",
        "optimizer_updates": CANDIDATE_UPDATES,
        "consumed_microbatches": CANDIDATE_UPDATES,
        "gradient_accumulation_steps": 1,
        "data_loader_microbatches_per_epoch": 8388,
        "checkpoint_epoch": 2,
        "checkpoint_iteration": 6756,
        "checkpoint_epoch_finished": False,
        "checkpoint_reason": "max_train_iters",
        "optimizer_state_count": 94,
        "optimizer_step_values": [CANDIDATE_UPDATES],
        "checkpoint_optimizer_step": CANDIDATE_UPDATES,
        "successful_update_batch_slots": (
            CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        ),
        "successful_updates_equal_consumed_microbatches": True,
    }
    if (
        selected_postflight.get("status") != "passed"
        or not isinstance(numerical, Mapping)
        or numerical.get("status") != "passed"
        or numerical.get("amp_enabled") is not True
        or int(numerical.get("finite_loss_observations", 0)) <= 0
        or numerical.get("loss_values_all_finite") is not True
        or int(numerical.get("amp_skip_observations", 0)) <= 0
        or float(numerical.get("max_amp_step_skipped", math.inf)) != 0.0
        or progress != expected_progress
        or not isinstance(telemetry, Mapping)
        or int(telemetry.get("sampling_interval_ms", -1)) != 1000
        or int(telemetry.get("sample_rows", 0)) <= 0
    ):
        raise HeadlineReleaseError(
            f"{run_id} lacks finite/zero-AMP-skip/1s-telemetry postflight evidence"
        )
    queue = _read_json(queue_manifest, label="M0 training queue")
    queue_plan = queue.get("plan")
    queue_items = queue_plan.get("items") if isinstance(queue_plan, Mapping) else None
    expected_queue_ids = [f"{CANDIDATE_ID}:{value}" for value in CANDIDATE_SEEDS]
    observed_queue_ids = [
        value.get("run_id") if isinstance(value, Mapping) else None
        for value in queue_items or []
    ]
    if (
        queue.get("status") != "completed"
        or observed_queue_ids != expected_queue_ids
        or any(
            not isinstance(value, Mapping) or value.get("runner") != "paper"
            for value in queue_items or []
        )
    ):
        raise HeadlineReleaseError(
            "M0 dedicated training queue must contain exactly seeds 17/42/73"
        )

    from tools import run_stageb_paper_evaluations as evaluator

    try:
        observed = evaluator._resolve_pivot_source(
            expected_root,
            evaluator.HashCache(),
            training_queue_dir=queue_manifest.parent,
        )
    except (OSError, ValueError, evaluator.PaperEvaluationError) as exc:
        raise HeadlineReleaseError(
            f"{run_id} formal source replay failed: {exc}"
        ) from exc
    observed_projection = {
        "training_run_id": observed.training_run_id,
        "training_seed": observed.training_seed,
        "training_run_root": str(observed.training_run_root),
        "checkpoint": str(observed.checkpoint),
        "checkpoint_sha256": observed.checkpoint_sha256,
        "config": str(observed.config),
        "training_queue_manifest": str(observed.training_queue_manifest),
        "training_queue_detached_launch": str(
            observed.training_queue_detached_launch
        ),
        "training_queue_detached_status": str(
            observed.training_queue_detached_status
        ),
        "training_queue_id": observed.training_queue_id,
        "training_queue_plan_sha256": observed.training_queue_plan_sha256,
    }
    declared_projection = {
        key: (
            str(Path(str(source[key])).resolve(strict=True))
            if key
            in {
                "training_run_root",
                "checkpoint",
                "config",
                "training_queue_manifest",
                "training_queue_detached_launch",
                "training_queue_detached_status",
            }
            else source[key]
        )
        for key in observed_projection
    }
    if declared_projection != observed_projection:
        raise HeadlineReleaseError(f"{run_id} source differs from provenance replay")
    return {
        "candidate_id": CANDIDATE_ID,
        "train_seed": seed,
        "training_run_id": run_id,
        "training_run_root": str(expected_root),
        "architecture_objective": CANDIDATE_ARCHITECTURE_OBJECTIVE,
        "batch_size": CANDIDATE_BATCH_SIZE,
        "optimizer_updates": CANDIDATE_UPDATES,
        "successful_update_batch_slots": (
            CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        ),
        "num_workers": 2,
        "prefetch_factor": 1,
        "amp": True,
        "iter_checkpoint_interval": 500,
        "gradient_diagnostic_interval": 100,
        "telemetry_interval_seconds": 1,
        "max_amp_step_skipped": 0.0,
        "stage_a_initializer_sha256": (
            stage_a_sha
        ),
        "scorer_initializer_sha256": stage_a_sha,
        "b58_model_ancestry": False,
        "training_attempt_count": lineage["attempt_count"],
        "same_run_resume_count": lineage["resume_count"],
        "config": file_record(config),
        "checkpoint": file_record(checkpoint),
        "training_queue": {
            "queue_id": str(source["training_queue_id"]),
            "plan_sha256": queue_plan_sha,
            "manifest": file_record(queue_manifest),
            "detached_launch": file_record(
                Path(str(source["training_queue_detached_launch"]))
            ),
            "detached_status": file_record(
                Path(str(source["training_queue_detached_status"]))
            ),
        },
    }


def _baseline_source_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_fixed_baseline_identity(
            evaluation_id=str(source.get("evaluation_id", "")),
            config=Path(str(source.get("config", ""))),
            checkpoint=Path(str(source.get("checkpoint", ""))),
            checkpoint_sha256=str(source.get("checkpoint_sha256", "")),
        )
    except (OSError, ValueError) as exc:
        raise HeadlineReleaseError(f"fixed b58 source is invalid: {exc}") from exc


def _calibration_metrics(postflight: Mapping[str, Any]) -> tuple[float, float]:
    from tools.compare_stageb_fpr95_records import (
        RecordComparisonError,
        exact_fpr95,
        load_manifest,
        load_tn_records,
    )

    artifacts = postflight.get("artifacts")
    calibration = (
        artifacts.get("matrix_calibration")
        if isinstance(artifacts, Mapping)
        else None
    )
    if not isinstance(calibration, Mapping):
        raise HeadlineReleaseError("validation postflight lacks matrix calibration")
    derived = calibration.get("derived_manifest")
    records = calibration.get("records")
    if not isinstance(derived, Mapping) or not isinstance(records, Mapping):
        raise HeadlineReleaseError("matrix calibration artifacts are incomplete")
    manifest_path = Path(str(derived.get("path", ""))).resolve(strict=True)
    records_path = Path(str(records.get("path", ""))).resolve(strict=True)
    _verify_file_record(derived, label="matrix calibration derived manifest")
    _verify_file_record(records, label="matrix calibration records")
    try:
        manifest = load_manifest(manifest_path)
        loaded = load_tn_records(records_path, manifest, label="headline selection")
        metric = exact_fpr95(loaded.positive, loaded.negative)
    except (OSError, ValueError, RecordComparisonError) as exc:
        raise HeadlineReleaseError(
            f"matrix calibration record replay failed: {exc}"
        ) from exc
    if not bool(loaded.valid.all()):
        raise HeadlineReleaseError("matrix calibration contains invalid records")
    fpr95 = float(metric["fpr"])
    positive_q05 = float(metric["threshold"])
    if not math.isclose(
        fpr95,
        float(calibration.get("summary_fpr95")),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise HeadlineReleaseError("matrix calibration FPR95 summary replay mismatch")
    return fpr95, positive_q05


def _unique_legacy_records(root_label: str, suffix: str) -> Path:
    root = Path(str(FIXED_BASELINE["legacy_roots"][root_label]["path"]))
    matches = sorted((root / "per_example_records").glob(f"*{suffix}"))
    if len(matches) != 1:
        raise HeadlineReleaseError(
            f"fixed b58 legacy {root_label} records are not unique for {suffix}"
        )
    return matches[0].resolve(strict=True)


def _iter_jsonl(path: Path, *, label: str):
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise HeadlineReleaseError(
                        f"{label}:{line_number}: blank JSONL row"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HeadlineReleaseError(
                        f"{label}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise HeadlineReleaseError(
                        f"{label}:{line_number}: row is not an object"
                    )
                yield line_number, value
    except (OSError, UnicodeDecodeError) as exc:
        raise HeadlineReleaseError(f"{label} is not readable: {exc}") from exc


def _aligned_record_consistency(
    fresh_path: Path,
    legacy_path: Path,
    *,
    task: str,
    label: str,
) -> dict[str, Any]:
    identity_fields = (
        "sample_id",
        "image_id",
        "ann_id",
        "ref_id",
        "sent_id",
    )
    numeric_fields = (
        ("top1_iou", "all_query_best_iou")
        if task == "ref"
        else ("pos_score", "neg_score", "pos_iou", "neg_iou")
    )
    fresh_iterator = _iter_jsonl(fresh_path, label=f"fresh {label}")
    legacy_iterator = _iter_jsonl(legacy_path, label=f"legacy {label}")
    sentinel = object()
    count = 0
    max_abs = {field: 0.0 for field in numeric_fields}
    correct_fresh = 0
    correct_legacy = 0
    fresh_positive: list[float] = []
    fresh_negative: list[float] = []
    legacy_positive: list[float] = []
    legacy_negative: list[float] = []
    while True:
        fresh_value = next(fresh_iterator, sentinel)
        legacy_value = next(legacy_iterator, sentinel)
        if fresh_value is sentinel or legacy_value is sentinel:
            if fresh_value is not legacy_value:
                raise HeadlineReleaseError(f"{label} fresh/legacy row counts differ")
            break
        fresh_line, fresh = fresh_value
        legacy_line, legacy = legacy_value
        if fresh_line != legacy_line or any(
            fresh.get(field) != legacy.get(field) for field in identity_fields
        ):
            raise HeadlineReleaseError(
                f"{label}:{fresh_line} fresh/legacy identities are not aligned"
            )
        if (
            fresh.get("task") != task
            or legacy.get("task") != task
            or fresh.get("valid") is not True
            or legacy.get("valid") is not True
        ):
            raise HeadlineReleaseError(f"{label}:{fresh_line} record validity drifted")
        for field in numeric_fields:
            try:
                left = float(fresh[field])
                right = float(legacy[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise HeadlineReleaseError(
                    f"{label}:{fresh_line} lacks numeric field {field}"
                ) from exc
            if not math.isfinite(left) or not math.isfinite(right):
                raise HeadlineReleaseError(
                    f"{label}:{fresh_line} has non-finite {field}"
                )
            delta = abs(left - right)
            max_abs[field] = max(max_abs[field], delta)
            if delta > BASELINE_RECORD_MAX_ABS_TOLERANCE:
                raise HeadlineReleaseError(
                    f"{label}:{fresh_line} {field} drift {delta} exceeds "
                    f"{BASELINE_RECORD_MAX_ABS_TOLERANCE}"
                )
        if task == "ref":
            if type(fresh.get("correct50")) is not bool or type(
                legacy.get("correct50")
            ) is not bool or fresh["correct50"] != legacy["correct50"]:
                raise HeadlineReleaseError(
                    f"{label}:{fresh_line} correct50 differs"
                )
            correct_fresh += int(fresh["correct50"])
            correct_legacy += int(legacy["correct50"])
        else:
            fresh_positive.append(float(fresh["pos_score"]))
            fresh_negative.append(float(fresh["neg_score"]))
            legacy_positive.append(float(legacy["pos_score"]))
            legacy_negative.append(float(legacy["neg_score"]))
        count += 1
    if count <= 0:
        raise HeadlineReleaseError(f"{label} consistency surface is empty")
    metrics: dict[str, Any]
    if task == "ref":
        fresh_acc = correct_fresh / count
        legacy_acc = correct_legacy / count
        if fresh_acc != legacy_acc:
            raise HeadlineReleaseError(f"{label} record-replayed Acc50 differs")
        metrics = {"fresh_acc50": fresh_acc, "legacy_acc50": legacy_acc}
    else:
        from tools.compare_stageb_fpr95_records import exact_fpr95

        fresh_metric = exact_fpr95(
            np.asarray(fresh_positive, dtype=np.float64),
            np.asarray(fresh_negative, dtype=np.float64),
        )
        legacy_metric = exact_fpr95(
            np.asarray(legacy_positive, dtype=np.float64),
            np.asarray(legacy_negative, dtype=np.float64),
        )
        if float(fresh_metric["fpr"]) != float(legacy_metric["fpr"]):
            raise HeadlineReleaseError(f"{label} record-replayed FPR95 differs")
        if abs(
            float(fresh_metric["threshold"])
            - float(legacy_metric["threshold"])
        ) > BASELINE_RECORD_MAX_ABS_TOLERANCE:
            raise HeadlineReleaseError(f"{label} record-replayed q05 differs")
        metrics = {
            "fresh_fpr95": float(fresh_metric["fpr"]),
            "legacy_fpr95": float(legacy_metric["fpr"]),
            "fresh_positive_q05": float(fresh_metric["threshold"]),
            "legacy_positive_q05": float(legacy_metric["threshold"]),
        }
    return {
        "status": "passed",
        "task": task,
        "rows": count,
        "identity_order_exact": True,
        "numeric_max_abs_tolerance": BASELINE_RECORD_MAX_ABS_TOLERANCE,
        "max_abs_observed": max_abs,
        "metrics_recomputed_from_records": metrics,
        "fresh_records": file_record(fresh_path),
        "legacy_records": file_record(legacy_path),
    }


def _baseline_validation_consistency(postflight: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = postflight.get("artifacts")
    ref = artifacts.get("ref_validation") if isinstance(artifacts, Mapping) else None
    if not isinstance(ref, Mapping) or set(ref) != set(VALIDATION_REF_SPLITS):
        raise HeadlineReleaseError("fresh b58 validation Ref records are incomplete")
    return {
        "status": "passed",
        "scope": "fresh_validation_val3_vs_legacy_ref8",
        "splits": {
            split: _aligned_record_consistency(
                Path(str(ref[split]["records"]["path"])).resolve(strict=True),
                _unique_legacy_records("ref8", f"__{split}.records.jsonl"),
                task="ref",
                label=f"b58 {split}",
            )
            for split in VALIDATION_REF_SPLITS
        },
    }


def _baseline_final_consistency(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    ref = artifacts.get("ref8")
    if not isinstance(ref, Mapping):
        raise HeadlineReleaseError("fresh b58 final Ref8 records are incomplete")
    from tools.stageb_ref_split_contract import REF_SPLITS

    if set(ref) != set(REF_SPLITS):
        raise HeadlineReleaseError("fresh b58 final Ref8 split set is not exact")
    result = {
        "status": "passed",
        "scope": "fresh_final_ref8_strict_vs_legacy_b58_contract",
        "ref8": {
            split: _aligned_record_consistency(
                Path(str(ref[split]["records"]["path"])).resolve(strict=True),
                _unique_legacy_records("ref8", f"__{split}.records.jsonl"),
                task="ref",
                label=f"b58 {split}",
            )
            for split in REF_SPLITS
        },
        "tn": {},
    }
    for split in ("strict2031", "strict1607"):
        evidence = artifacts.get(split)
        if not isinstance(evidence, Mapping) or not isinstance(
            evidence.get("records"), Mapping
        ):
            raise HeadlineReleaseError(f"fresh b58 {split} records are incomplete")
        result["tn"][split] = _aligned_record_consistency(
            Path(str(evidence["records"]["path"])).resolve(strict=True),
            _unique_legacy_records(split, "__tn_global.records.jsonl"),
            task="tn",
            label=f"b58 {split}",
        )
    return result


def _validation_evidence(role: str, seed: int) -> dict[str, Any]:
    from tools import run_stageb_paper_evaluations as evaluator

    root = canonical_validation_root(role, seed).resolve(strict=True)
    launch_path = (root / "launch_manifest.json").resolve(strict=True)
    postflight_path = (root / "postflight.json").resolve(strict=True)
    launch = _read_json(launch_path, label=f"{role} validation launch")
    postflight = _read_json(postflight_path, label=f"{role} validation postflight")
    if (
        launch.get("schema") != evaluator.SCHEMA
        or launch.get("status") != "completed"
        or Path(str(launch.get("output_dir", ""))).resolve(strict=True) != root
        or launch.get("output_dir_fresh_at_plan") is not True
    ):
        raise HeadlineReleaseError(f"{role} validation launch is not completed/fresh")
    protocol = launch.get("protocol")
    if not isinstance(protocol, Mapping) or protocol != {
        **dict(protocol),
        "profile": evaluator.MATRIX_PROFILE,
    } or protocol.get("processes") != ["validation_calibration"]:
        raise HeadlineReleaseError(f"{role} validation profile/process drifted")
    completed = launch.get("completed_phases")
    if not (
        isinstance(completed, list)
        and len(completed) == 1
        and completed[0].get("phase_id") == "validation_calibration"
        and completed[0].get("status") == "completed"
        and completed[0].get("returncode") == 0
    ):
        raise HeadlineReleaseError(f"{role} validation phase is incomplete")
    source = launch.get("source")
    runtime = launch.get("runtime")
    inputs = launch.get("inputs")
    if not all(isinstance(value, Mapping) for value in (source, runtime, inputs)):
        raise HeadlineReleaseError(f"{role} validation provenance is incomplete")
    runtime_contract = validate_canonical_runtime(runtime)
    records = inputs.get("records")
    if not isinstance(records, list):
        raise HeadlineReleaseError(f"{role} validation input records are missing")
    fingerprints = common_evaluation_fingerprints(records)

    expected_commands = evaluator._commands(
        evaluator.Runtime(
            python=Path(runtime_contract["python"]),
            data_root=Path(runtime_contract["data_root"]),
            device=str(runtime_contract["device"]),
            batch_size=int(runtime_contract["batch_size"]),
            num_workers=int(runtime_contract["num_workers"]),
            amp=bool(runtime_contract["amp"]),
            log_every=int(runtime_contract["log_every"]),
        ),
        SimpleNamespace(
            config=Path(str(source.get("config", ""))).resolve(strict=True),
            checkpoint=Path(str(source.get("checkpoint", ""))).resolve(strict=True),
        ),
        root,
        profile=evaluator.MATRIX_PROFILE,
    )
    if launch.get("commands") != expected_commands:
        raise HeadlineReleaseError(f"{role} validation command surface drifted")

    observed_rehash = evaluator._rehash_inputs(launch)
    persisted_rehash = postflight.get("input_rehash")
    if not isinstance(persisted_rehash, Mapping) or _normalized_replay(
        observed_rehash
    ) != _normalized_replay(persisted_rehash):
        raise HeadlineReleaseError(f"{role} validation input rehash replay differs")
    replay = evaluator._postflight_screen(launch, observed_rehash)
    if _normalized_replay(replay) != _normalized_replay(postflight):
        raise HeadlineReleaseError(f"{role} validation postflight replay differs")
    if launch.get("postflight") != postflight:
        raise HeadlineReleaseError(f"{role} validation embedded postflight differs")
    _verify_file_record(
        launch.get("postflight_artifact"),
        label=f"{role} validation postflight",
        expected_path=postflight_path,
    )

    if role == "baseline":
        identity = _baseline_source_projection(source)
        baseline_consistency = _baseline_validation_consistency(postflight)
    elif role == "candidate":
        identity = _candidate_source_projection(source, seed)
        baseline_consistency = None
    else:
        raise HeadlineReleaseError(f"invalid validation role {role!r}")
    ref_artifacts = postflight.get("artifacts", {}).get("ref_validation")
    if not isinstance(ref_artifacts, Mapping) or set(ref_artifacts) != set(
        VALIDATION_REF_SPLITS
    ):
        raise HeadlineReleaseError(f"{role} validation Ref artifact set drifted")
    ref = {
        split: float(ref_artifacts[split]["summary_acc50"])
        for split in VALIDATION_REF_SPLITS
    }
    fpr95, positive_q05 = _calibration_metrics(postflight)
    return {
        "role": role,
        "seed": seed,
        "root": str(root),
        "launch": file_record(launch_path),
        "postflight": file_record(postflight_path),
        "source": identity,
        "runtime": runtime_contract,
        "fingerprints": fingerprints,
        "metrics": {
            "ref": ref,
            "fpr95": fpr95,
            "positive_q05": positive_q05,
        },
        "legacy_baseline_consistency": baseline_consistency,
        "contracts": dict(postflight.get("contracts", {})),
    }


def _selection_payload() -> dict[str, Any]:
    final_roots = [
        canonical_final_root("baseline", BASELINE_TRAIN_SEED),
        *(canonical_final_root("candidate", seed) for seed in CANDIDATE_SEEDS),
    ]
    existing = [str(path) for path in final_roots if path.exists()]
    if existing or FINAL_CONTRACT_ROOT.exists():
        raise HeadlineReleaseError(
            f"selection receipt must predate all final access: {existing}"
        )
    exposure = validate_baseline_exposure_receipt()
    exposure_record = file_record(BASELINE_EXPOSURE_RECEIPT_PATH)
    baseline = _validation_evidence("baseline", BASELINE_TRAIN_SEED)
    candidates = {
        seed: _validation_evidence("candidate", seed) for seed in CANDIDATE_SEEDS
    }
    common_runtime = baseline["runtime"]
    common_fingerprints = baseline["fingerprints"]
    for seed, evidence in candidates.items():
        if evidence["runtime"] != common_runtime:
            raise HeadlineReleaseError(
                f"M0 seed{seed} validation runtime differs from fixed b58"
            )
        if evidence["fingerprints"] != common_fingerprints:
            raise HeadlineReleaseError(
                f"M0 seed{seed} validation code/data differs from fixed b58"
            )
    candidate_config_shas = {
        evidence["source"]["config"]["sha256"] for evidence in candidates.values()
    }
    candidate_queue_identities = {
        (
            evidence["source"]["training_queue"]["queue_id"],
            evidence["source"]["training_queue"]["plan_sha256"],
        )
        for evidence in candidates.values()
    }
    if len(candidate_config_shas) != 1 or len(candidate_queue_identities) != 1:
        raise HeadlineReleaseError(
            "M0 three seeds must share one config and one dedicated queue plan"
        )
    eligibility = evaluate_selection_eligibility(
        baseline["metrics"],
        {seed: evidence["metrics"] for seed, evidence in candidates.items()},
    )
    payload = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "status": "eligible" if eligibility["passed"] else "ineligible",
        "candidate_predeclared": True,
        "selection_policy": SELECTION_POLICY,
        "fallback_policy": "none_close_final_gate",
        "headline_bootstrap": dict(HEADLINE_BOOTSTRAP_CONTRACT),
        "candidate_id": CANDIDATE_ID,
        "candidate_seeds": list(CANDIDATE_SEEDS),
        "baseline_id": BASELINE_ID,
        "architecture_objective": CANDIDATE_ARCHITECTURE_OBJECTIVE,
        "compute_matching": {
            "matching_unit": "successful_optimizer_update_global_batch_slots",
            "baseline_successful_update_batch_slots": (
                BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "candidate_successful_update_batch_slots": (
                CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "candidate_minus_baseline_successful_update_batch_slots": (
                CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
                - BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "scope_limitations": {
                "total_consumed_batch_slots_derived": False,
                "flops_matched": False,
                "wall_clock_compute_matched": False,
                "statement": (
                    "This receipt matches batch slots attached to successful optimizer "
                    "updates. Optimizer state does not count AMP-skipped attempted "
                    "batches, total consumed samples, FLOPs, or wall-clock compute."
                ),
            },
            "exposure_receipt": exposure_record,
            "receipt_status": exposure["status"],
        },
        "baseline_validation": baseline,
        "candidate_validation": [candidates[seed] for seed in CANDIDATE_SEEDS],
        "validation_parity": {
            "runtime": common_runtime,
            "code_closure": common_fingerprints["code"],
            "data_inputs": common_fingerprints["data"],
        },
        "eligibility": eligibility,
        "created_before_first_final_evaluation": True,
        "sealed_at_utc": _utc_now(),
    }
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    return payload


def seal_selection_receipt(
    path: Path = SELECTION_RECEIPT_PATH,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if path != SELECTION_RECEIPT_PATH.resolve(strict=False):
        raise HeadlineReleaseError("selection receipt path is not canonical")
    if path.exists():
        raise FileExistsError(f"selection receipt already exists: {path}")
    payload = _selection_payload()
    _write_json_exclusive(path, payload)
    validate_selection_receipt(path)
    return payload


def validate_selection_receipt(
    path: Path = SELECTION_RECEIPT_PATH,
    *,
    replay_validation: bool = True,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if path != SELECTION_RECEIPT_PATH.resolve(strict=True):
        raise HeadlineReleaseError("selection receipt path is not canonical")
    payload = _self_bound_payload(
        path,
        schema=SELECTION_RECEIPT_SCHEMA,
        hash_field="receipt_sha256",
        label="headline selection receipt",
    )
    if (
        payload.get("candidate_predeclared") is not True
        or payload.get("selection_policy") != SELECTION_POLICY
        or payload.get("fallback_policy") != "none_close_final_gate"
        or payload.get("headline_bootstrap") != HEADLINE_BOOTSTRAP_CONTRACT
        or payload.get("candidate_id") != CANDIDATE_ID
        or payload.get("candidate_seeds") != list(CANDIDATE_SEEDS)
        or payload.get("baseline_id") != BASELINE_ID
        or payload.get("architecture_objective")
        != CANDIDATE_ARCHITECTURE_OBJECTIVE
        or payload.get("created_before_first_final_evaluation") is not True
    ):
        raise HeadlineReleaseError("selection receipt fixed identity/policy drifted")
    compute_matching = payload.get("compute_matching")
    if not isinstance(compute_matching, Mapping) or (
        compute_matching.get("matching_unit")
        != "successful_optimizer_update_global_batch_slots"
        or compute_matching.get(
            "baseline_successful_update_batch_slots"
        )
        != BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        or compute_matching.get(
            "candidate_successful_update_batch_slots"
        )
        != CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        or compute_matching.get(
            "candidate_minus_baseline_successful_update_batch_slots"
        )
        != CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        - BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        or compute_matching.get("scope_limitations")
        != {
            "total_consumed_batch_slots_derived": False,
            "flops_matched": False,
            "wall_clock_compute_matched": False,
            "statement": (
                "This receipt matches batch slots attached to successful optimizer "
                "updates. Optimizer state does not count AMP-skipped attempted "
                "batches, total consumed samples, FLOPs, or wall-clock compute."
            ),
        }
        or compute_matching.get("receipt_status") != "verified"
    ):
        raise HeadlineReleaseError(
            "selection receipt successful-update batch-slot matching drifted"
        )
    _verify_file_record(
        compute_matching.get("exposure_receipt"),
        label="b58 exposure derivation receipt",
        expected_path=BASELINE_EXPOSURE_RECEIPT_PATH,
    )
    validate_baseline_exposure_receipt()
    eligibility = payload.get("eligibility")
    expected_status = (
        "eligible"
        if isinstance(eligibility, Mapping) and eligibility.get("passed") is True
        else "ineligible"
    )
    if payload.get("status") != expected_status:
        raise HeadlineReleaseError("selection receipt eligibility/status disagree")
    if replay_validation:
        replay = _selection_payload()
        for field in (
            "status",
            "candidate_predeclared",
            "selection_policy",
            "fallback_policy",
            "headline_bootstrap",
            "candidate_id",
            "candidate_seeds",
            "baseline_id",
            "architecture_objective",
            "compute_matching",
            "baseline_validation",
            "candidate_validation",
            "validation_parity",
            "eligibility",
            "created_before_first_final_evaluation",
        ):
            if replay.get(field) != payload.get(field):
                raise HeadlineReleaseError(
                    f"selection receipt replay differs for {field}"
                )
    return payload


def validate_paper_ablation_completion_receipt(
    path: Path = PAPER_ABLATION_COMPLETION_RECEIPT_PATH,
) -> dict[str, Any]:
    """Replay the code-registered completion adapters for every paper block."""

    from tools import build_stageb_paper_ablation_completion_receipt as builder

    try:
        canonical = PAPER_ABLATION_COMPLETION_RECEIPT_PATH.resolve(strict=False)
        requested = path.expanduser().resolve(strict=True)
        if (
            builder.CANONICAL_RECEIPT_PATH.resolve(strict=False) != canonical
            or requested != canonical
            or builder.SCHEMA != COMPLETION_RECEIPT_SCHEMA
            or builder.BLOCKS != COMPLETION_BLOCKS
        ):
            raise HeadlineReleaseError(
                "paper completion builder/release contract drifted"
            )
        return builder.verify_receipt(requested)
    except builder.CompletionReceiptError as exc:
        raise HeadlineReleaseError(
            f"paper ablation completion replay failed: {exc}"
        ) from exc


def validate_baseline_exposure_receipt(
    path: Path = BASELINE_EXPOSURE_RECEIPT_PATH,
) -> dict[str, Any]:
    """Replay the canonical builder's single authoritative exposure audit."""

    from tools import build_stageb_b58_exposure_receipt as exposure_builder

    try:
        canonical = BASELINE_EXPOSURE_RECEIPT_PATH.resolve(strict=False)
        builder_canonical = exposure_builder.CANONICAL_RECEIPT_PATH.resolve(
            strict=False
        )
        requested = path.expanduser().resolve(strict=True)
        if canonical != builder_canonical or requested != canonical:
            raise HeadlineReleaseError(
                "b58 exposure builder/release receipt path contract drifted"
            )
        if (
            exposure_builder.SCHEMA != BASELINE_EXPOSURE_RECEIPT_SCHEMA
            or exposure_builder.BASELINE_BATCH_SIZE != BASELINE_BATCH_SIZE
            or exposure_builder.BASELINE_OPTIMIZER_UPDATES
            != BASELINE_OPTIMIZER_UPDATES
            or exposure_builder.BASELINE_OPTIMIZER_STATE_COUNT
            != BASELINE_OPTIMIZER_STATE_COUNT
            or exposure_builder.BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            != BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            or exposure_builder.CANDIDATE_BATCH_SIZE != CANDIDATE_BATCH_SIZE
            or exposure_builder.CANDIDATE_OPTIMIZER_UPDATES != CANDIDATE_UPDATES
            or exposure_builder.CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            != CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            or exposure_builder.LOCKED_SHA256.get("checkpoint0001")
            != FIXED_BASELINE["checkpoint_sha256"]
        ):
            raise HeadlineReleaseError(
                "b58 exposure builder/release constants drifted"
            )
        return exposure_builder.verify_receipt(requested)
    except exposure_builder.ExposureReceiptError as exc:
        raise HeadlineReleaseError(
            f"b58 exposure derivation replay failed: {exc}"
        ) from exc


def _validate_receipt_artifacts(
    artifacts: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        RECEIPT_ARTIFACT_NAMES
    ):
        raise HeadlineReleaseError("headline gate receipt artifact set is not exact")
    expected_paths = {
        "selection_receipt": SELECTION_RECEIPT_PATH,
        "baseline_contract": BASELINE_CONTRACT_PATH,
        "candidate_contract": CANDIDATE_CONTRACT_PATH,
        "evaluation_parity_contract": PARITY_CONTRACT_PATH,
    }
    return {
        name: _verify_file_record(
            artifacts[name],
            label=name,
            expected_path=expected_paths[name],
        )
        for name in RECEIPT_ARTIFACT_NAMES
    }


def _validate_gate_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    if path != FINAL_GATE_PATH.resolve(strict=True):
        raise HeadlineReleaseError("headline final gate path is not canonical")
    gate = _self_bound_payload(
        path,
        schema=FINAL_GATE_SCHEMA,
        hash_field="gate_sha256",
        label="headline final gate",
    )
    if (
        gate.get("status") != "sealed"
        or gate.get("selection_frozen") is not True
        or gate.get("created_before_first_final_evaluation") is not True
        or gate.get("selection_policy") != SELECTION_POLICY
        or gate.get("fallback_policy") != "none_close_final_gate"
        or gate.get("headline_bootstrap") != HEADLINE_BOOTSTRAP_CONTRACT
    ):
        raise HeadlineReleaseError("headline final gate is not sealed exactly")
    artifacts = _validate_receipt_artifacts(gate.get("receipt_artifacts"))
    completion_record = gate.get("paper_ablation_completion_receipt")
    _verify_file_record(
        completion_record,
        label="paper ablation completion receipt",
        expected_path=PAPER_ABLATION_COMPLETION_RECEIPT_PATH,
    )
    validate_paper_ablation_completion_receipt(
        Path(str(completion_record["path"]))
    )
    exposure_record = gate.get("baseline_exposure_receipt")
    _verify_file_record(
        exposure_record,
        label="b58 exposure derivation receipt",
        expected_path=BASELINE_EXPOSURE_RECEIPT_PATH,
    )
    validate_baseline_exposure_receipt(Path(str(exposure_record["path"])))
    instances = gate.get("instances")
    if not isinstance(instances, list) or len(instances) != 4:
        raise HeadlineReleaseError("headline final gate must authorize four instances")
    expected_ids = {BASELINE_ID, *(f"{CANDIDATE_ID}:{seed}" for seed in CANDIDATE_SEEDS)}
    observed_ids = {
        str(value.get("instance_id"))
        for value in instances
        if isinstance(value, Mapping)
    }
    if observed_ids != expected_ids or len(observed_ids) != len(instances):
        raise HeadlineReleaseError("headline final gate instance set is not exact")
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise HeadlineReleaseError(f"gate instance {index} is invalid")
        payload = dict(instance)
        expected_sha = str(payload.pop("instance_sha256", ""))
        if (
            instance.get("schema") != INSTANCE_SCHEMA
            or _SHA_RE.fullmatch(expected_sha) is None
            or expected_sha != canonical_json_sha256(payload)
        ):
            raise HeadlineReleaseError(f"gate instance {index} self binding failed")
    return gate, artifacts


def _consumption_path(instance: Mapping[str, Any]) -> Path:
    digest = str(instance.get("instance_sha256", ""))
    if _SHA_RE.fullmatch(digest) is None:
        raise HeadlineReleaseError("headline final instance has no valid SHA-256")
    return FINAL_CONSUMPTION_ROOT / f"{digest}.json"


def consume_final_instance(plan: Mapping[str, Any]) -> dict[str, Any]:
    release = plan.get("headline_release")
    if not isinstance(release, Mapping):
        raise HeadlineReleaseError("final launch has no headline release binding")
    instance = release.get("instance")
    gate_record = release.get("final_gate")
    artifacts = release.get("receipt_artifacts")
    if not isinstance(instance, Mapping) or not isinstance(gate_record, Mapping):
        raise HeadlineReleaseError("final launch has no instance/gate binding")
    output = Path(str(plan.get("output_dir", ""))).expanduser().resolve(strict=False)
    if output.exists():
        raise HeadlineReleaseError("final output must be fresh before gate consumption")
    gate_path = _verify_file_record(
        gate_record,
        label="headline final gate",
        expected_path=FINAL_GATE_PATH,
    )["path"]
    gate, observed_artifacts = _validate_gate_payload(Path(gate_path))
    if dict(artifacts or {}) != dict(gate.get("receipt_artifacts") or {}):
        raise HeadlineReleaseError("final launch receipt artifacts differ from gate")
    matches = [
        value
        for value in gate["instances"]
        if value.get("instance_id") == instance.get("instance_id")
        and value.get("instance_sha256") == instance.get("instance_sha256")
    ]
    if len(matches) != 1 or dict(matches[0]) != dict(instance):
        raise HeadlineReleaseError("gate does not uniquely authorize final instance")
    if Path(str(instance.get("output_dir", ""))).resolve(strict=False) != output:
        raise HeadlineReleaseError("final instance output root differs from plan")
    path = _consumption_path(instance).resolve(strict=False)
    payload = {
        "schema": FINAL_CONSUMPTION_SCHEMA,
        "status": "claimed",
        "instance_id": instance["instance_id"],
        "instance_sha256": instance["instance_sha256"],
        "gate": _verify_file_record(
            gate_record,
            label="headline final gate",
            expected_path=FINAL_GATE_PATH,
        ),
        "receipt_artifacts": observed_artifacts,
        "paper_ablation_completion_receipt": dict(
            gate["paper_ablation_completion_receipt"]
        ),
        "baseline_exposure_receipt": dict(gate["baseline_exposure_receipt"]),
        "output_dir": str(output),
        "claimed_at_utc": _utc_now(),
    }
    payload["consumption_sha256"] = canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_json_exclusive(path, payload)
    except FileExistsError as exc:
        raise HeadlineReleaseError(
            "headline final instance was already consumed; rerun is forbidden"
        ) from exc
    return file_record(path)


def validate_final_consumption(plan: Mapping[str, Any]) -> dict[str, Any]:
    release = plan.get("headline_release")
    if not isinstance(release, Mapping):
        raise HeadlineReleaseError("final launch has no headline release binding")
    instance = release.get("instance")
    record = release.get("final_consumption")
    gate_record = release.get("final_gate")
    if not all(isinstance(value, Mapping) for value in (instance, record, gate_record)):
        raise HeadlineReleaseError("final launch consumption receipt is missing")
    path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
    if path != _consumption_path(instance).resolve(strict=True):
        raise HeadlineReleaseError("final consumption path is not canonical")
    _verify_file_record(record, label="final consumption", expected_path=path)
    payload = _self_bound_payload(
        path,
        schema=FINAL_CONSUMPTION_SCHEMA,
        hash_field="consumption_sha256",
        label="final consumption",
    )
    if (
        payload.get("status") != "claimed"
        or payload.get("instance_id") != instance.get("instance_id")
        or payload.get("instance_sha256") != instance.get("instance_sha256")
        or payload.get("output_dir") != plan.get("output_dir")
        or payload.get("gate") != gate_record
        or payload.get("receipt_artifacts") != release.get("receipt_artifacts")
        or payload.get("paper_ablation_completion_receipt")
        != release.get("paper_ablation_completion_receipt")
        or payload.get("baseline_exposure_receipt")
        != release.get("baseline_exposure_receipt")
    ):
        raise HeadlineReleaseError("final consumption receipt drifted")
    return file_record(path)


def _legacy_baseline_evidence() -> dict[str, Any]:
    manifest = Path(str(FIXED_BASELINE["legacy_results_manifest"]))
    if sha256_file(manifest) != FIXED_BASELINE["legacy_results_manifest_sha256"]:
        raise HeadlineReleaseError("fixed b58 legacy results manifest changed")
    roots: dict[str, Any] = {}
    for label, value in FIXED_BASELINE["legacy_roots"].items():
        root = Path(str(value["path"])).resolve(strict=True)
        summary = (root / "summary.json").resolve(strict=True)
        if sha256_file(summary) != value["summary_sha256"]:
            raise HeadlineReleaseError(f"fixed b58 legacy {label} summary changed")
        roots[label] = {
            "root": str(root),
            "summary": file_record(summary),
        }
    return {
        "role": "historical_consistency_only_not_headline_metric_source",
        "results_manifest": file_record(manifest),
        "roots": roots,
    }


def _final_parity_payload(selection: Mapping[str, Any]) -> dict[str, Any]:
    from tools import run_stageb_paper_evaluations as evaluator
    from tools.stageb_ref_split_contract import REF_SPLIT_CONTRACT, REF_SPLITS

    validation_parity = selection.get("validation_parity")
    if not isinstance(validation_parity, Mapping):
        raise HeadlineReleaseError("selection receipt lacks validation parity")
    current_code = _fingerprint_records(
        (
            {
                **file_record(path),
                "roles": ["evaluation_code_dependency"],
            }
            for path in evaluator._evaluation_code_paths()
        ),
        {"evaluation_code_dependency"},
    )
    current_data = _fingerprint_records(
        (
            {
                **file_record(path),
                "roles": ["evaluation_data_input"],
            }
            for path in evaluator._data_input_paths(DEFAULT_DATA_ROOT.resolve(strict=True))
        ),
        {"evaluation_data_input"},
    )
    if (
        current_code != validation_parity.get("code_closure")
        or current_data != validation_parity.get("data_inputs")
        or dict(CANONICAL_RUNTIME) != validation_parity.get("runtime")
    ):
        raise HeadlineReleaseError(
            "evaluator code/data/runtime changed after validation selection"
        )
    strict = {
        label: {
            "sha256": str(specification["sha256"]),
            "rows": int(specification["rows"]),
        }
        for label, specification in evaluator.STRICT_SPECS.items()
    }
    ref = {
        split: {
            "sha256": str(REF_SPLIT_CONTRACT[split]["sha256"]),
            "rows": int(REF_SPLIT_CONTRACT[split]["rows"]),
        }
        for split in REF_SPLITS
    }
    return {
        "schema": PARITY_CONTRACT_SCHEMA,
        "status": "sealed",
        "profile": evaluator.FINAL_PROFILE,
        "processes": ["ref8_strict2031", "strict1607"],
        "runtime": dict(validation_parity["runtime"]),
        "code_closure": dict(current_code),
        "data_inputs": dict(current_data),
        "ref_manifest_digests": ref,
        "strict_manifest_digests": strict,
        "model_specific_inputs_excluded_from_parity": [
            "evaluation_checkpoint",
            "evaluation_config",
            "config_dependency",
            "training_provenance",
        ],
        "shared_inputs_required_equal": [
            "evaluation_runtime",
            "evaluation_code_dependency",
            "evaluation_data_input",
            "ref_manifest_contract",
            "strict_manifest_contract",
        ],
    }


def _source_for_instance(identity: Mapping[str, Any], role: str) -> dict[str, Any]:
    if role == "baseline":
        return {
            "id": BASELINE_ID,
            "train_seed": BASELINE_TRAIN_SEED,
            "role": "fixed_historical_checkpoint",
            "config": dict(identity["config"]),
            "checkpoint": dict(identity["checkpoint"]),
        }
    queue = identity.get("training_queue")
    if not isinstance(queue, Mapping):
        raise HeadlineReleaseError("candidate identity lacks training queue")
    return {
        "id": CANDIDATE_ID,
        "train_seed": int(identity["train_seed"]),
        "role": "three_seed_candidate",
        "architecture_objective": identity["architecture_objective"],
        "batch_size": identity["batch_size"],
        "optimizer_updates": identity["optimizer_updates"],
        "successful_update_batch_slots": identity[
            "successful_update_batch_slots"
        ],
        "num_workers": identity["num_workers"],
        "prefetch_factor": identity["prefetch_factor"],
        "amp": identity["amp"],
        "iter_checkpoint_interval": identity["iter_checkpoint_interval"],
        "gradient_diagnostic_interval": identity[
            "gradient_diagnostic_interval"
        ],
        "telemetry_interval_seconds": identity["telemetry_interval_seconds"],
        "max_amp_step_skipped": identity["max_amp_step_skipped"],
        "stage_a_initializer_sha256": identity["stage_a_initializer_sha256"],
        "scorer_initializer_sha256": identity["scorer_initializer_sha256"],
        "b58_model_ancestry": identity["b58_model_ancestry"],
        "training_attempt_count": identity["training_attempt_count"],
        "same_run_resume_count": identity["same_run_resume_count"],
        "training_run_id": identity["training_run_id"],
        "training_run_root": identity["training_run_root"],
        "config": dict(identity["config"]),
        "checkpoint": dict(identity["checkpoint"]),
        "training_queue_id": queue["queue_id"],
        "training_queue_plan_sha256": queue["plan_sha256"],
        "training_queue_manifest": dict(queue["manifest"]),
    }


def seal_final_gate(
    path: Path = FINAL_GATE_PATH,
    *,
    selection_path: Path = SELECTION_RECEIPT_PATH,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if path != FINAL_GATE_PATH.resolve(strict=False):
        raise HeadlineReleaseError("headline final gate path is not canonical")
    if selection_path.expanduser().resolve(strict=True) != SELECTION_RECEIPT_PATH.resolve(
        strict=True
    ):
        raise HeadlineReleaseError("headline selection receipt path is not canonical")
    if FINAL_CONTRACT_ROOT.exists():
        raise FileExistsError(
            f"headline final contract root already exists: {FINAL_CONTRACT_ROOT}"
        )
    selection = validate_selection_receipt(selection_path, replay_validation=True)
    if selection.get("status") != "eligible":
        raise HeadlineReleaseError(
            "fixed M0 candidate is ineligible; no fallback final gate is permitted"
        )
    completion = validate_paper_ablation_completion_receipt()
    completion_record = file_record(PAPER_ABLATION_COMPLETION_RECEIPT_PATH)
    exposure = validate_baseline_exposure_receipt()
    exposure_record = file_record(BASELINE_EXPOSURE_RECEIPT_PATH)
    final_roots = [
        canonical_final_root("baseline", BASELINE_TRAIN_SEED),
        *(canonical_final_root("candidate", seed) for seed in CANDIDATE_SEEDS),
    ]
    if any(root.exists() for root in final_roots):
        raise HeadlineReleaseError("final evaluation predates gate sealing")

    baseline_validation = selection.get("baseline_validation")
    candidate_validation = selection.get("candidate_validation")
    if not isinstance(baseline_validation, Mapping) or not isinstance(
        candidate_validation, list
    ) or len(candidate_validation) != 3:
        raise HeadlineReleaseError("selection receipt validation identities are incomplete")
    candidate_by_seed = {
        int(value.get("seed", -1)): value
        for value in candidate_validation
        if isinstance(value, Mapping)
    }
    if set(candidate_by_seed) != set(CANDIDATE_SEEDS):
        raise HeadlineReleaseError("selection receipt candidate seed set drifted")
    baseline_source = _source_for_instance(
        baseline_validation["source"], "baseline"
    )
    candidate_sources = {
        seed: _source_for_instance(candidate_by_seed[seed]["source"], "candidate")
        for seed in CANDIDATE_SEEDS
    }
    parity = _final_parity_payload(selection)
    baseline_contract = {
        "schema": BASELINE_CONTRACT_SCHEMA,
        "status": "sealed",
        "id": BASELINE_ID,
        "train_seed": BASELINE_TRAIN_SEED,
        "role": "fixed_historical_checkpoint",
        "source": baseline_source,
        "validation_root": baseline_validation["root"],
        "validation_launch": baseline_validation["launch"],
        "validation_postflight": baseline_validation["postflight"],
        "final_evaluation_root": str(
            canonical_final_root("baseline", BASELINE_TRAIN_SEED).resolve(
                strict=False
            )
        ),
        "legacy_consistency_evidence": _legacy_baseline_evidence(),
    }
    candidate_contract = {
        "schema": CANDIDATE_CONTRACT_SCHEMA,
        "status": "sealed",
        "id": CANDIDATE_ID,
        "architecture_objective": CANDIDATE_ARCHITECTURE_OBJECTIVE,
        "seeds": list(CANDIDATE_SEEDS),
        "compute_contract": {
            "batch_size": CANDIDATE_BATCH_SIZE,
            "optimizer_updates": CANDIDATE_UPDATES,
            "successful_update_batch_slots": (
                CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "baseline_successful_update_batch_slots": (
                BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "successful_update_batch_slot_delta": (
                CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
                - BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "total_sample_or_flop_exposure_matched": False,
        },
        "selection_policy": SELECTION_POLICY,
        "fallback_policy": "none_close_final_gate",
        "runs": [
            {
                **candidate_sources[seed],
                "validation_root": candidate_by_seed[seed]["root"],
                "validation_launch": candidate_by_seed[seed]["launch"],
                "validation_postflight": candidate_by_seed[seed]["postflight"],
                "final_evaluation_root": str(
                    canonical_final_root("candidate", seed).resolve(strict=False)
                ),
            }
            for seed in CANDIDATE_SEEDS
        ],
    }
    instances = [
        _instance_payload(
            role="baseline",
            seed=BASELINE_TRAIN_SEED,
            source=baseline_source,
            output_dir=canonical_final_root("baseline", BASELINE_TRAIN_SEED),
            parity=parity,
        ),
        *(
            _instance_payload(
                role="candidate",
                seed=seed,
                source=candidate_sources[seed],
                output_dir=canonical_final_root("candidate", seed),
                parity=parity,
            )
            for seed in CANDIDATE_SEEDS
        ),
    ]

    parent = FINAL_CONTRACT_ROOT.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".headline-final-stage-", dir=parent))
    try:
        stage_paths = {
            "baseline_contract": stage / BASELINE_CONTRACT_PATH.name,
            "candidate_contract": stage / CANDIDATE_CONTRACT_PATH.name,
            "evaluation_parity_contract": stage / PARITY_CONTRACT_PATH.name,
        }
        payloads = {
            "baseline_contract": baseline_contract,
            "candidate_contract": candidate_contract,
            "evaluation_parity_contract": parity,
        }
        for name, payload in payloads.items():
            _write_json_exclusive(stage_paths[name], payload)
        receipt_artifacts = {
            "selection_receipt": file_record(selection_path),
        }
        final_paths = {
            "baseline_contract": BASELINE_CONTRACT_PATH,
            "candidate_contract": CANDIDATE_CONTRACT_PATH,
            "evaluation_parity_contract": PARITY_CONTRACT_PATH,
        }
        for name in stage_paths:
            record = file_record(stage_paths[name])
            record["path"] = str(final_paths[name].resolve(strict=False))
            receipt_artifacts[name] = record
        gate = {
            "schema": FINAL_GATE_SCHEMA,
            "status": "sealed",
            "selection_frozen": True,
            "created_before_first_final_evaluation": True,
            "selection_policy": SELECTION_POLICY,
            "fallback_policy": "none_close_final_gate",
            "headline_bootstrap": dict(HEADLINE_BOOTSTRAP_CONTRACT),
            "instances": instances,
            "receipt_artifacts": receipt_artifacts,
            "paper_ablation_completion_receipt": completion_record,
            "paper_ablation_completion_status": completion["status"],
            "baseline_exposure_receipt": exposure_record,
            "baseline_exposure_status": exposure["status"],
            "sealed_at_utc": _utc_now(),
        }
        gate["gate_sha256"] = canonical_json_sha256(gate)
        _write_json_exclusive(stage / FINAL_GATE_PATH.name, gate)
        os.rename(stage, FINAL_CONTRACT_ROOT)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    validated, _ = _validate_gate_payload(path)
    return validated


def _load_semantic_contracts(
    artifacts: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = validate_selection_receipt(
        Path(str(artifacts["selection_receipt"]["path"])),
        replay_validation=False,
    )
    baseline, _ = _read_contract(
        Path(str(artifacts["baseline_contract"]["path"])),
        schema=BASELINE_CONTRACT_SCHEMA,
        label="headline baseline contract",
    )
    candidate, _ = _read_contract(
        Path(str(artifacts["candidate_contract"]["path"])),
        schema=CANDIDATE_CONTRACT_SCHEMA,
        label="headline candidate contract",
    )
    parity, _ = _read_contract(
        Path(str(artifacts["evaluation_parity_contract"]["path"])),
        schema=PARITY_CONTRACT_SCHEMA,
        label="headline evaluation parity contract",
    )
    if selection.get("status") != "eligible":
        raise HeadlineReleaseError("headline selection receipt is not eligible")
    if (
        baseline.get("id") != BASELINE_ID
        or baseline.get("role") != "fixed_historical_checkpoint"
        or int(baseline.get("train_seed", -1)) != BASELINE_TRAIN_SEED
        or candidate.get("id") != CANDIDATE_ID
        or candidate.get("seeds") != list(CANDIDATE_SEEDS)
        or candidate.get("selection_policy") != SELECTION_POLICY
    ):
        raise HeadlineReleaseError("headline baseline/candidate identity drifted")
    return selection, baseline, candidate, parity


def _plan_source_projection(plan: Mapping[str, Any]) -> tuple[str, int, dict[str, Any]]:
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise HeadlineReleaseError("final plan source is missing")
    if source.get("kind") == "historical_pure_gdino_explicit":
        identity = _baseline_source_projection(source)
        return "baseline", BASELINE_TRAIN_SEED, _source_for_instance(
            identity, "baseline"
        )
    if source.get("kind") == "pivot_paper_training_run":
        try:
            seed = int(source.get("training_seed"))
        except (TypeError, ValueError) as exc:
            raise HeadlineReleaseError("candidate final seed is invalid") from exc
        identity = _candidate_source_projection(source, seed)
        return "candidate", seed, _source_for_instance(identity, "candidate")
    raise HeadlineReleaseError("final plan source is neither fixed b58 nor M0")


def _validate_plan_against_gate(
    plan: Mapping[str, Any],
    gate_path: Path,
    *,
    require_consumption: bool,
) -> dict[str, Any]:
    from tools import run_stageb_paper_evaluations as evaluator

    protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("profile") != evaluator.FINAL_PROFILE
        or protocol.get("processes") != ["ref8_strict2031", "strict1607"]
        or protocol.get("ref_splits") != list(evaluator.REF_SPLITS)
        or protocol.get("strict1607_skip_ref") is not True
    ):
        raise HeadlineReleaseError("headline final plan surface is not canonical")
    gate, artifacts = _validate_gate_payload(gate_path)
    selection, baseline, candidate, parity = _load_semantic_contracts(artifacts)
    runtime = validate_canonical_runtime(plan.get("runtime", {}))
    if runtime != parity.get("runtime"):
        raise HeadlineReleaseError("final plan runtime differs from parity contract")
    records = plan.get("inputs", {}).get("records")
    if not isinstance(records, list):
        raise HeadlineReleaseError("final plan input closure is missing")
    fingerprints = common_evaluation_fingerprints(records)
    if (
        fingerprints.get("code") != parity.get("code_closure")
        or fingerprints.get("data") != parity.get("data_inputs")
    ):
        raise HeadlineReleaseError("final plan code/data parity fingerprint drifted")
    strict_records = protocol.get("strict_manifests")
    if not isinstance(strict_records, Mapping):
        raise HeadlineReleaseError("final plan strict manifest contract is missing")
    observed_strict = {
        label: {
            "sha256": value.get("sha256"),
            "rows": int(value.get("rows", -1)),
        }
        for label, value in strict_records.items()
        if isinstance(value, Mapping)
    }
    if observed_strict != parity.get("strict_manifest_digests"):
        raise HeadlineReleaseError("final plan strict manifest digests drifted")
    observed_ref = {
        split: {
            "sha256": str(evaluator.REF_SPLIT_CONTRACT[split]["sha256"]),
            "rows": int(evaluator.REF_SPLIT_CONTRACT[split]["rows"]),
        }
        for split in evaluator.REF_SPLITS
    }
    if observed_ref != parity.get("ref_manifest_digests"):
        raise HeadlineReleaseError("final plan Ref manifest digests drifted")
    role, seed, source = _plan_source_projection(plan)
    output = Path(str(plan.get("output_dir", ""))).resolve(strict=False)
    expected_output = canonical_final_root(role, seed).resolve(strict=False)
    if output != expected_output:
        raise HeadlineReleaseError("final plan output root is not canonical")
    expected_instance = _instance_payload(
        role=role,
        seed=seed,
        source=source,
        output_dir=output,
        parity=parity,
    )
    matches = [
        value
        for value in gate["instances"]
        if value.get("instance_id") == expected_instance["instance_id"]
        and value.get("instance_sha256") == expected_instance["instance_sha256"]
    ]
    if len(matches) != 1 or dict(matches[0]) != expected_instance:
        raise HeadlineReleaseError("final gate does not authorize reconstructed plan")
    release = plan.get("headline_release")
    if isinstance(release, Mapping):
        if (
            release.get("instance") != expected_instance
            or release.get("receipt_artifacts") != gate.get("receipt_artifacts")
            or release.get("final_gate") != file_record(gate_path)
            or release.get("paper_ablation_completion_receipt")
            != gate.get("paper_ablation_completion_receipt")
            or release.get("baseline_exposure_receipt")
            != gate.get("baseline_exposure_receipt")
        ):
            raise HeadlineReleaseError("final plan release binding drifted")
    if require_consumption:
        consumption = validate_final_consumption(plan)
    else:
        consumption_path = _consumption_path(expected_instance)
        if consumption_path.exists():
            raise HeadlineReleaseError(
                "headline final instance was already consumed; rerun is forbidden"
            )
        consumption = None
    candidate_run = None
    baseline_run = None
    if role == "baseline":
        baseline_run = {
            "id": BASELINE_ID,
            "seed": BASELINE_TRAIN_SEED,
            "checkpoint_sha256": source["checkpoint"]["sha256"],
            "role": "fixed_historical_checkpoint",
            "evaluation_root": str(output),
        }
        if source != baseline.get("source"):
            raise HeadlineReleaseError("final b58 source differs from baseline contract")
    else:
        candidate_runs = {
            int(value["train_seed"]): value for value in candidate.get("runs", [])
        }
        declared = candidate_runs.get(seed)
        if not isinstance(declared, Mapping):
            raise HeadlineReleaseError(f"candidate contract lacks M0 seed{seed}")
        declared_source = {
            key: declared[key]
            for key in source
        }
        if source != declared_source:
            raise HeadlineReleaseError(
                f"final M0 seed{seed} source differs from candidate contract"
            )
        candidate_run = {
            "training_run_id": source["training_run_id"],
            "seed": seed,
            "queue_id": source["training_queue_id"],
            "queue_plan_sha256": source["training_queue_plan_sha256"],
            "checkpoint_sha256": source["checkpoint"]["sha256"],
            "evaluation_root": str(output),
        }
    return {
        "instance": expected_instance,
        "artifacts": {
            **artifacts,
            "final_gate": file_record(gate_path),
            "paper_ablation_completion_receipt": dict(
                gate["paper_ablation_completion_receipt"]
            ),
            "baseline_exposure_receipt": dict(
                gate["baseline_exposure_receipt"]
            ),
        },
        "selection": selection,
        "baseline_contract": baseline,
        "candidate_contract": candidate,
        "parity": parity,
        "consumption": consumption,
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
    }


def bind_final_gate(plan: Mapping[str, Any], gate_path: Path) -> dict[str, Any]:
    evidence = _validate_plan_against_gate(
        plan, gate_path, require_consumption=False
    )
    return {
        "instance": evidence["instance"],
        "receipt_artifacts": {
            name: evidence["artifacts"][name]
            for name in RECEIPT_ARTIFACT_NAMES
        },
        "paper_ablation_completion_receipt": evidence["artifacts"][
            "paper_ablation_completion_receipt"
        ],
        "baseline_exposure_receipt": evidence["artifacts"][
            "baseline_exposure_receipt"
        ],
        "final_gate": evidence["artifacts"]["final_gate"],
        "final_consumption": None,
    }


def release_input_paths(binding: Mapping[str, Any]) -> tuple[Path, ...]:
    artifacts = binding.get("receipt_artifacts")
    gate = binding.get("final_gate")
    completion = binding.get("paper_ablation_completion_receipt")
    exposure = binding.get("baseline_exposure_receipt")
    if not all(
        isinstance(value, Mapping)
        for value in (artifacts, gate, completion, exposure)
    ):
        raise HeadlineReleaseError("headline release binding is incomplete")
    return tuple(
        Path(str(artifacts[name]["path"])).resolve(strict=True)
        for name in RECEIPT_ARTIFACT_NAMES
    ) + (
        Path(str(gate["path"])).resolve(strict=True),
        Path(str(completion["path"])).resolve(strict=True),
        Path(str(exposure["path"])).resolve(strict=True),
    )


def validate_completed_final_plan(
    plan: Mapping[str, Any],
    *,
    final_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    release = plan.get("headline_release")
    if not isinstance(release, Mapping):
        raise HeadlineReleaseError("completed final plan has no release binding")
    gate = release.get("final_gate")
    if not isinstance(gate, Mapping):
        raise HeadlineReleaseError("completed final plan has no final gate")
    evidence = _validate_plan_against_gate(
        plan,
        Path(str(gate.get("path", ""))),
        require_consumption=True,
    )
    if evidence["instance"]["role"] == "baseline":
        if not isinstance(final_artifacts, Mapping):
            raise HeadlineReleaseError(
                "fresh b58 final requires record-level legacy consistency replay"
            )
        baseline_consistency = _baseline_final_consistency(final_artifacts)
    else:
        baseline_consistency = None
    return {
        "status": "passed",
        "instance": evidence["instance"],
        "artifacts": evidence["artifacts"],
        "parity": evidence["parity"],
        "baseline_run": evidence["baseline_run"],
        "candidate_run": evidence["candidate_run"],
        "final_consumption": evidence["consumption"],
        "legacy_baseline_consistency": baseline_consistency,
    }


def _read_contract(
    path: Path, *, schema: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(path, label=label)
    if payload.get("schema") != schema or payload.get("status") != "sealed":
        raise HeadlineReleaseError(f"{label} is not sealed")
    return payload, file_record(path)


def _instance_payload(
    *,
    role: str,
    seed: int,
    source: Mapping[str, Any],
    output_dir: Path,
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    instance_id = BASELINE_ID if role == "baseline" else f"{CANDIDATE_ID}:{seed}"
    payload = {
        "schema": INSTANCE_SCHEMA,
        "instance_id": instance_id,
        "role": role,
        "candidate_id": None if role == "baseline" else CANDIDATE_ID,
        "train_seed": seed,
        "source": dict(source),
        "output_dir": str(output_dir.resolve(strict=False)),
        "profile": "final",
        "processes": ["ref8_strict2031", "strict1607"],
        "runtime": dict(parity["runtime"]),
        "code_closure_digest": parity["code_closure"]["digest"],
        "data_input_digest": parity["data_inputs"]["digest"],
        "ref_manifest_digests": dict(parity["ref_manifest_digests"]),
        "strict_manifest_digests": dict(parity["strict_manifest_digests"]),
    }
    payload["instance_sha256"] = canonical_json_sha256(payload)
    return payload


def build_release_provenance(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Consolidate four individually verified final roots for the builder."""

    if len(evaluations) != 4:
        raise HeadlineReleaseError("headline manifest requires exactly four final roots")
    by_id = {str(value.get("instance", {}).get("instance_id")): value for value in evaluations}
    expected = {BASELINE_ID, *(f"{CANDIDATE_ID}:{seed}" for seed in CANDIDATE_SEEDS)}
    if set(by_id) != expected or len(by_id) != len(evaluations):
        raise HeadlineReleaseError("headline final evaluation instance set is not exact")
    first = evaluations[0]
    artifact_names = (
        *RECEIPT_ARTIFACT_NAMES,
        "final_gate",
        "paper_ablation_completion_receipt",
        "baseline_exposure_receipt",
    )
    artifacts = first.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(artifact_names):
        raise HeadlineReleaseError("headline release artifact set is not exact")
    for value in evaluations[1:]:
        if value.get("artifacts") != artifacts:
            raise HeadlineReleaseError("final roots bind different release receipts")
        if value.get("parity") != first.get("parity"):
            raise HeadlineReleaseError("final roots have different parity contracts")
    baseline_eval = by_id[BASELINE_ID]
    candidate_rows = [by_id[f"{CANDIDATE_ID}:{seed}"] for seed in CANDIDATE_SEEDS]
    candidate = {
        "id": CANDIDATE_ID,
        "seeds": list(CANDIDATE_SEEDS),
        "runs": [dict(value["candidate_run"]) for value in candidate_rows],
    }
    if (
        int(bootstrap.get("iterations", -1))
        != HEADLINE_BOOTSTRAP_CONTRACT["iterations"]
        or not math.isclose(
            float(bootstrap.get("confidence", -1)),
            HEADLINE_BOOTSTRAP_CONTRACT["confidence"],
        )
        or isinstance(bootstrap.get("seed"), bool)
        or bootstrap.get("seed") != HEADLINE_BOOTSTRAP_CONTRACT["seed"]
    ):
        raise HeadlineReleaseError(
            "headline bootstrap contract must be exact 5000/.95/20260717"
        )
    return {
        "schema": RELEASE_PROVENANCE_SCHEMA,
        "status": "passed",
        **{name: dict(artifacts[name]) for name in artifact_names},
        "baseline": dict(baseline_eval["baseline_run"]),
        "candidate": candidate,
        "evaluation_parity": dict(first["parity"]),
        "artifact_contract": {
            "fresh_canonical_roots": True,
            "records_contained_in_each_evaluation_root": True,
            "postrun_input_rehash_passed": True,
            "one_consumption_receipt_per_final_instance": True,
            "repeated_final_instance_rejected": True,
        },
        "bootstrap": dict(HEADLINE_BOOTSTRAP_CONTRACT),
    }


def verify_manifest_release_provenance(
    provenance: Any,
    *,
    experiments: Any,
    baseline_id: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a builder-produced receipt bundle at aggregation time."""

    if not isinstance(provenance, Mapping):
        raise HeadlineReleaseError("manifest headline provenance is missing")
    required_artifacts = (
        *RECEIPT_ARTIFACT_NAMES,
        "final_gate",
        "paper_ablation_completion_receipt",
        "baseline_exposure_receipt",
    )
    if (
        provenance.get("schema") != RELEASE_PROVENANCE_SCHEMA
        or provenance.get("status") != "passed"
        or any(not isinstance(provenance.get(name), Mapping) for name in required_artifacts)
    ):
        raise HeadlineReleaseError("manifest headline provenance is not passed/exact")
    expected_paths = {
        "selection_receipt": SELECTION_RECEIPT_PATH,
        "baseline_contract": BASELINE_CONTRACT_PATH,
        "candidate_contract": CANDIDATE_CONTRACT_PATH,
        "evaluation_parity_contract": PARITY_CONTRACT_PATH,
        "final_gate": FINAL_GATE_PATH,
        "paper_ablation_completion_receipt": PAPER_ABLATION_COMPLETION_RECEIPT_PATH,
        "baseline_exposure_receipt": BASELINE_EXPOSURE_RECEIPT_PATH,
    }
    observed_artifacts = {
        name: _verify_file_record(
            provenance[name], label=name, expected_path=expected_paths[name]
        )
        for name in required_artifacts
    }
    gate, gate_artifacts = _validate_gate_payload(FINAL_GATE_PATH)
    if any(
        observed_artifacts[name] != gate_artifacts[name]
        for name in RECEIPT_ARTIFACT_NAMES
    ) or observed_artifacts["final_gate"] != file_record(FINAL_GATE_PATH):
        raise HeadlineReleaseError("manifest artifacts differ from final gate")
    if (
        observed_artifacts["paper_ablation_completion_receipt"]
        != gate.get("paper_ablation_completion_receipt")
        or observed_artifacts["baseline_exposure_receipt"]
        != gate.get("baseline_exposure_receipt")
    ):
        raise HeadlineReleaseError("manifest auxiliary receipts differ from gate")
    _, baseline_contract, candidate_contract, parity = _load_semantic_contracts(
        gate_artifacts
    )
    if provenance.get("evaluation_parity") != parity:
        raise HeadlineReleaseError("manifest parity projection differs from receipt")
    for closure_name in ("code_closure", "data_inputs"):
        closure = parity.get(closure_name)
        records = closure.get("records") if isinstance(closure, Mapping) else None
        if not isinstance(records, list) or not records:
            raise HeadlineReleaseError(f"parity {closure_name} is empty")
        for index, record in enumerate(records):
            _verify_file_record(
                record, label=f"parity {closure_name} record {index}"
            )
        if closure.get("digest") != canonical_json_sha256({"records": records}):
            raise HeadlineReleaseError(f"parity {closure_name} digest drifted")
    expected_bootstrap = provenance.get("bootstrap")
    if (
        expected_bootstrap != HEADLINE_BOOTSTRAP_CONTRACT
        or int(bootstrap.get("iterations", -1))
        != HEADLINE_BOOTSTRAP_CONTRACT["iterations"]
        or not math.isclose(
            float(bootstrap.get("confidence", -1)),
            HEADLINE_BOOTSTRAP_CONTRACT["confidence"],
        )
        or bootstrap.get("seed") != HEADLINE_BOOTSTRAP_CONTRACT["seed"]
    ):
        raise HeadlineReleaseError("manifest headline bootstrap contract drifted")
    artifact_contract = provenance.get("artifact_contract")
    required_artifact_flags = {
        "fresh_canonical_roots",
        "records_contained_in_each_evaluation_root",
        "postrun_input_rehash_passed",
        "one_consumption_receipt_per_final_instance",
        "repeated_final_instance_rejected",
    }
    if not isinstance(artifact_contract, Mapping) or any(
        artifact_contract.get(field) is not True for field in required_artifact_flags
    ):
        raise HeadlineReleaseError("manifest fresh/single-use artifact contract drifted")
    if baseline_id != BASELINE_ID or provenance.get("baseline") != {
        "id": BASELINE_ID,
        "seed": BASELINE_TRAIN_SEED,
        "checkpoint_sha256": FIXED_BASELINE["checkpoint_sha256"],
        "role": "fixed_historical_checkpoint",
        "evaluation_root": str(
            canonical_final_root("baseline", BASELINE_TRAIN_SEED).resolve(
                strict=True
            )
        ),
    }:
        raise HeadlineReleaseError("manifest fixed b58 baseline projection drifted")
    candidate = provenance.get("candidate")
    candidate_runs = candidate.get("runs") if isinstance(candidate, Mapping) else None
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("id") != CANDIDATE_ID
        or candidate.get("seeds") != list(CANDIDATE_SEEDS)
        or not isinstance(candidate_runs, list)
        or len(candidate_runs) != 3
    ):
        raise HeadlineReleaseError("manifest M0 candidate projection drifted")
    raw_experiments = experiments if isinstance(experiments, list) else []
    by_id = {
        value.get("id"): value
        for value in raw_experiments
        if isinstance(value, Mapping) and isinstance(value.get("id"), str)
    }
    if set(by_id) != {BASELINE_ID, CANDIDATE_ID}:
        raise HeadlineReleaseError("headline manifest experiment set is not exact")
    expected_runs = {
        BASELINE_ID: {BASELINE_TRAIN_SEED: provenance["baseline"]},
        CANDIDATE_ID: {
            int(value["seed"]): value for value in candidate_runs
        },
    }
    replayed_evaluations: list[Mapping[str, Any]] = []
    for experiment_id, expected_by_seed in expected_runs.items():
        experiment = by_id[experiment_id]
        runs = experiment.get("runs")
        if not isinstance(runs, list):
            raise HeadlineReleaseError(f"experiment {experiment_id} runs are invalid")
        observed_seeds = {
            int(value.get("train_seed", -1)) for value in runs if isinstance(value, Mapping)
        }
        if observed_seeds != set(expected_by_seed) or len(runs) != len(expected_by_seed):
            raise HeadlineReleaseError(f"experiment {experiment_id} seed set drifted")
        for run in runs:
            seed = int(run["train_seed"])
            expected = expected_by_seed[seed]
            checkpoint = run.get("artifacts", {}).get("checkpoint")
            if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != expected[
                "checkpoint_sha256"
            ]:
                raise HeadlineReleaseError(
                    f"experiment {experiment_id} seed{seed} checkpoint drifted"
                )
            root = Path(str(run.get("evaluation_root", ""))).resolve(strict=True)
            if str(root) != expected["evaluation_root"]:
                raise HeadlineReleaseError(
                    f"experiment {experiment_id} seed{seed} final root drifted"
                )
            launch = _read_json(root / "launch_manifest.json", label="final launch")
            postflight = _read_json(root / "postflight.json", label="final postflight")
            replay = validate_completed_final_plan(
                launch, final_artifacts=postflight.get("artifacts")
            )
            if postflight.get("headline_release") != replay:
                raise HeadlineReleaseError(
                    f"experiment {experiment_id} seed{seed} postflight replay drifted"
                )
            replayed_evaluations.append(replay)
    rebuilt = build_release_provenance(
        replayed_evaluations,
        bootstrap=bootstrap,
    )
    if rebuilt != dict(provenance):
        raise HeadlineReleaseError("manifest headline provenance differs from full replay")
    return {
        "passed": True,
        "verdict": "verified_one_time_final_release",
        "data_contract_status": "complete",
        "unmet": [],
        "receipt_artifacts": observed_artifacts,
        "candidate_id": CANDIDATE_ID,
        "candidate_seeds": list(CANDIDATE_SEEDS),
        "baseline_id": BASELINE_ID,
    }
