"""Process-scoped contract for the class-aligned Table-B matched panel.

This module is intentionally additive.  The historical Table-B contract remains
the default implementation.  The v2 runner installs this module under the
historical import name in a fresh training process *before* importing ``main``;
there is therefore no global behavior change for legacy runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from util.path_compat import remap_legacy_path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ID = "table_b_matched_class_aligned_v2_supplemental_source_policy"
SCOPE_SCHEMA = "pivot.stageb.table_b_v2_process_scope/v2"
FORMAL_SOURCE_PLAN_SCHEMA = "pivot.stageb.table_b_v2_formal_source_plan/v1"
FORMAL_SCOPE_PLAN_SCHEMA = "pivot.stageb.table_b_v2_formal_scope_plan/v1"
FORMAL_GENERIC_EXTENSION_SCHEMA = (
    "pivot.stageb.table_b_v2_generic_queue_extension/v1"
)
FORMAL_PROFILE = "table_b_v2_formal_b40_u1000_i1000"
NONFORMAL_PROFILE = "table_b_v2_nonformal_fixture"
FORMAL_BATCH_SIZE = 40
FORMAL_TRAIN_UPDATES = 1_000
FORMAL_CHECKPOINT_INTERVAL = 1_000
FORMAL_SUCCESSFUL_UPDATE_BATCH_SLOTS = 40_000
AUDIT_SCHEMA = "stage-b-paper-c2-parent-matched-tn-v2"
AUDIT_PATH = (
    REPO_ROOT
    / "data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/audit.json"
)
AUDIT_SHA256 = "5ff62a838a5123d580a72e353147b97bb69e9d7967348b55cba4ccb9ca36cb96"
# ``datasets.patch_episode`` consumes the outer scope-preserving envelope.  The
# class-aligned builder adds a separate matched-pair evidence schema inside it.
TABLE_B_PAIR_SCHEMA = "stage-b-paper-table-b-scope-preserving-pair-v1"
TABLE_B_MATCHED_PAIR_SCHEMA = "stage-b-paper-c2-parent-matched-pair-v2"
TABLE_B_AUDIT_SCHEMA = AUDIT_SCHEMA
TABLE_B_SCOPE_BY_ID = {
    "D2m": "traceable_counterfactual_edit",
    "D3m": "proposal_covered_verified",
}
DATASET_PATH_BY_ID = {
    "D2m": REPO_ROOT
    / "config/datasets_stageb_table_b_d2m_matched_class_aligned_v2_traceable.json",
    "D3m": REPO_ROOT
    / "config/datasets_stageb_table_b_d3m_matched_class_aligned_v2_proposal_covered.json",
}
DATASET_SHA256_BY_ID = {
    "D2m": "83371dbbd259e44abfc6b3ac1b95bcc7dbddb732df99f3f6422c72b2b12ea365",
    "D3m": "4a3fdcb6d40c2a27460226490421719a52a4d694df33418fa073bbfeadc27520",
}
TRAIN_OUTPUT_KEY_BY_ID = {"D2m": "d2m_train", "D3m": "d3m_train"}
TRAIN_ROWS = 7_074
SEEDS = (17, 42, 73)
PHASE_ID = "joint"
FORMAL_RUN_IDS = tuple(
    f"{table_b_id}:{seed}"
    for seed in SEEDS
    for table_b_id in ("D2m", "D3m")
)
FORMAL_TRAINING_CONTRACT = {
    "batch_size": FORMAL_BATCH_SIZE,
    "optimizer_updates": FORMAL_TRAIN_UPDATES,
    "iter_checkpoint_interval": FORMAL_CHECKPOINT_INTERVAL,
    "successful_update_batch_slots": FORMAL_SUCCESSFUL_UPDATE_BATCH_SLOTS,
    "contributing_phase_updates": {PHASE_ID: FORMAL_TRAIN_UPDATES},
}
RUNTIME_DATASET_CONTRACT = "v24_parent_matched_class_aligned_v2_fail_closed"
CLAIM_SCOPE = {
    "pairwise_effect_population": "matched_pairs_only",
    "primary_causal_stratum": "class_aligned_identical_complete_input",
    "primary_causal_stratum_requires_exact_model_input": True,
    "canonical_class_id_equality_required": True,
    "unmatched_d3_parent_rows_are_out_of_scope": True,
    "generalization_to_unmatched_d3_parent_rows_supported": False,
}
LEGACY_IMPORT_NAME = "util.stage_b_table_b_contract"
SCOPE_SHA_ENV = "PIVOT_TABLE_B_V2_SCOPE_SHA256"
FORMAL_SOURCE_PLAN_ENV = "PIVOT_TABLE_B_V2_FORMAL_SOURCE_PLAN"
FORMAL_SOURCE_PLAN_SHA_ENV = "PIVOT_TABLE_B_V2_FORMAL_SOURCE_PLAN_SHA256"
FORMAL_SCOPE_PLAN_ENV = "PIVOT_TABLE_B_V2_FORMAL_SCOPE_PLAN"
FORMAL_SCOPE_PLAN_SHA_ENV = "PIVOT_TABLE_B_V2_FORMAL_SCOPE_PLAN_SHA256"
TRAINING_IMPORT_NAMES = ("main", "engine", "datasets", "datasets.patch_episode")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIRED_SOURCES = {"sam3_tn_pair", "sam3_paired_tn", "sam3_and_tn"}
_PROCESS_SCOPE: dict[str, Any] | None = None
_PROCESS_SCOPE_SHA256: str | None = None


class TableBContractError(RuntimeError):
    """Raised when the v2 process or data boundary is not exact."""


@dataclass(frozen=True)
class TableBConfidenceContract:
    table_b_id: str
    scope: str
    scope_allowlist: tuple[str, ...]
    audit_path: Path
    audit_sha256: str
    train_path: Path
    train_sha256: str
    allow_single_edit_token_provenance: bool


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise TableBContractError(f"value is not canonical ASCII JSON: {error}") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2048)
def _sha256_file_identity_cached(
    path_text: str, size_bytes: int, mtime_ns: int, ctime_ns: int
) -> str:
    del size_bytes, mtime_ns, ctime_ns
    return sha256_file(Path(path_text))


def _sha256_file_stable(path: Path) -> str:
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _sha256_file_identity_cached(
        str(path),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _resolve_path(value: Any, *, field: str, strict: bool = True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TableBContractError(f"{field} must be a non-empty path")
    path = remap_legacy_path(value, repo_root=REPO_ROOT)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        return path.resolve(strict=strict)
    except FileNotFoundError as error:
        raise TableBContractError(f"{field} does not exist: {path}") from error


def file_record(path: Path, *, role: str) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "role": role,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TableBContractError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TableBContractError(f"{label} root must be an object")
    return value


def _verify_record(record: Any, *, role: str) -> Path:
    if not isinstance(record, Mapping) or record.get("role") != role:
        raise TableBContractError(f"scope record {role!r} is missing")
    path = _resolve_path(record.get("path"), field=f"{role}.path")
    if record.get("sha256") != sha256_file(path):
        raise TableBContractError(f"{role} SHA-256 drifted")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] != path.stat().st_size:
        raise TableBContractError(f"{role} size drifted")
    return path


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("semantic_sha256", None)
    return canonical_sha256(value)


def _verify_plan_record(record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise TableBContractError(f"{label} record is missing")
    path = _resolve_path(record.get("path"), field=f"{label}.path")
    sha256 = record.get("sha256")
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise TableBContractError(f"{label} SHA-256 is malformed")
    if _sha256_file_stable(path) != sha256:
        raise TableBContractError(f"{label} SHA-256 drifted")
    if type(record.get("size_bytes")) is not int or record["size_bytes"] != path.stat().st_size:
        raise TableBContractError(f"{label} size drifted")
    return path


def validate_formal_source_plan(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    require_live_closure: bool = True,
) -> dict[str, Any]:
    path = _resolve_path(path, field="formal source plan")
    if expected_file_sha256 is not None:
        if not isinstance(expected_file_sha256, str) or not _SHA256_RE.fullmatch(
            expected_file_sha256
        ):
            raise TableBContractError("formal source plan SHA-256 is malformed")
        if sha256_file(path) != expected_file_sha256:
            raise TableBContractError("formal source plan file SHA-256 drifted")
    plan = _json_object(path, label="Table-B v2 formal source plan")
    if not (
        plan.get("schema") == FORMAL_SOURCE_PLAN_SCHEMA
        and plan.get("status") == "sealed"
        and plan.get("profile") == FORMAL_PROFILE
        and plan.get("formal_training_contract") == FORMAL_TRAINING_CONTRACT
        and plan.get("ordered_run_ids") == list(FORMAL_RUN_IDS)
        and plan.get("semantic_sha256") == _semantic_sha256(plan)
    ):
        raise TableBContractError("formal source plan identity/semantic SHA-256 drifted")

    queue = plan.get("queue")
    runtime = plan.get("runtime")
    closure = plan.get("source_closure")
    common = plan.get("common_input_contract")
    if not (
        isinstance(queue, Mapping)
        and isinstance(queue.get("queue_id"), str)
        and bool(queue["queue_id"])
        and isinstance(queue.get("plan_sha256"), str)
        and bool(_SHA256_RE.fullmatch(str(queue["plan_sha256"])))
        and isinstance(runtime, Mapping)
        and runtime.get("batch_size") == FORMAL_BATCH_SIZE
        and runtime.get("total_train_iters") == FORMAL_TRAIN_UPDATES
        and runtime.get("iter_checkpoint_interval") == FORMAL_CHECKPOINT_INTERVAL
        and isinstance(runtime.get("output_root"), str)
        and Path(runtime["output_root"]).is_absolute()
        and isinstance(closure, Mapping)
        and closure.get("status") == "sealed"
        and isinstance(closure.get("records"), list)
        and bool(closure["records"])
        and closure.get("semantic_sha256") == canonical_sha256(closure["records"])
        and isinstance(common, Mapping)
        and common.get("status") == "passed"
        and common.get("only_declared_condition_inputs_differ") is True
        and common.get("common_inputs_identical") is True
    ):
        raise TableBContractError("formal source plan queue/runtime/closure is incomplete")

    queue_dir = _resolve_path(queue.get("queue_dir"), field="formal queue directory")
    if not queue_dir.is_dir():
        raise TableBContractError("formal queue path is not a directory")
    queue_payload = _json_object(queue_dir / "queue.json", label="generic training queue")
    generic_plan = queue_payload.get("plan")
    if not (
        isinstance(generic_plan, Mapping)
        and generic_plan.get("queue_id") == queue["queue_id"]
        and queue_payload.get("plan_sha256") == queue["plan_sha256"]
        and canonical_sha256(generic_plan) == queue["plan_sha256"]
        and [item.get("run_id") for item in generic_plan.get("items", [])]
        == list(FORMAL_RUN_IDS)
        and all(item.get("runner") == "paper" for item in generic_plan.get("items", []))
    ):
        raise TableBContractError("generic queue identity/order differs from formal source plan")
    paper_runner = generic_plan.get("runners", {}).get("paper")
    expected_runner = (REPO_ROOT / "tools/run_stageb_table_b_v2.py").resolve(strict=True)
    if not (
        isinstance(paper_runner, Mapping)
        and _resolve_path(paper_runner.get("path"), field="formal paper runner")
        == expected_runner
        and paper_runner.get("sha256") == sha256_file(expected_runner)
    ):
        raise TableBContractError("formal queue does not bind the exact Table-B v2 runner")
    controller_path = (
        REPO_ROOT / "tools/run_stageb_table_b_v2_queue.py"
    ).resolve(strict=True)
    controller_record = {
        "path": str(controller_path),
        "sha256": sha256_file(controller_path),
        "size_bytes": int(controller_path.stat().st_size),
    }
    expected_extension: dict[str, Any] = {
        "schema": FORMAL_GENERIC_EXTENSION_SCHEMA,
        "profile": FORMAL_PROFILE,
        "ordered_run_ids": list(FORMAL_RUN_IDS),
        "formal_training_contract": dict(FORMAL_TRAINING_CONTRACT),
        "explicit_output_root": runtime["output_root"],
        "source_closure_semantic_sha256": closure["semantic_sha256"],
        "common_input_identity_sha256": common.get("common_identity_sha256"),
        "dedicated_controller": controller_record,
    }
    expected_extension["semantic_sha256"] = _semantic_sha256(expected_extension)
    if not (
        generic_plan.get("extensions") == expected_extension
        and queue.get("extension_semantic_sha256")
        == expected_extension["semantic_sha256"]
    ):
        raise TableBContractError("generic queue formal extension drifted")
    environment = generic_plan.get("runtime_environment")
    if not (
        isinstance(environment, Mapping)
        and environment.get("PIVOT_TN_OUTPUT_ROOT") == runtime["output_root"]
        and environment.get("PIVOT_BATCH_SIZE") == str(FORMAL_BATCH_SIZE)
        and environment.get("PIVOT_MAX_TRAIN_ITERS") == str(FORMAL_TRAIN_UPDATES)
        and environment.get("PIVOT_ITER_CHECKPOINT_INTERVAL")
        == str(FORMAL_CHECKPOINT_INTERVAL)
    ):
        raise TableBContractError("generic queue did not seal the explicit formal runtime")

    records = closure["records"]
    seen: set[str] = set()
    required_paths = {
        expected_runner,
        Path(__file__).resolve(strict=True),
        (REPO_ROOT / "tools/run_stageb_serial_matrix_queue.py").resolve(strict=True),
        (REPO_ROOT / "tools/run_stageb_paper_ablation_matrices.py").resolve(strict=True),
        (REPO_ROOT / "main.py").resolve(strict=True),
        (REPO_ROOT / "engine.py").resolve(strict=True),
    }
    observed_paths: set[Path] = set()
    for index, record in enumerate(records):
        record_path = _verify_plan_record(record, label=f"source closure record {index}")
        key = str(record_path)
        if key in seen:
            raise TableBContractError("formal source closure contains duplicate paths")
        seen.add(key)
        observed_paths.add(record_path)
        if not require_live_closure:
            continue
    if not required_paths.issubset(observed_paths):
        missing = sorted(str(value) for value in required_paths - observed_paths)
        raise TableBContractError(f"formal source closure misses required sources: {missing}")
    return json.loads(canonical_bytes(plan))


def formal_context_from_source_plan(
    plan: Mapping[str, Any], *, source_plan_path: Path
) -> dict[str, Any]:
    validated = validate_formal_source_plan(source_plan_path)
    if dict(plan) != validated:
        raise TableBContractError("formal source plan payload/path differ")
    queue = validated["queue"]
    return {
        "profile": FORMAL_PROFILE,
        "claim_class": "supplemental_matched_source_policy_sensitivity",
        "training_contract": dict(FORMAL_TRAINING_CONTRACT),
        "output_root": str(Path(validated["runtime"]["output_root"]).resolve()),
        "queue": {
            "queue_id": queue["queue_id"],
            "plan_sha256": queue["plan_sha256"],
            "queue_dir": str(Path(queue["queue_dir"]).resolve()),
        },
        "source_plan": file_record(
            Path(source_plan_path), role="table_b_v2_formal_source_plan"
        ),
        "source_plan_semantic_sha256": validated["semantic_sha256"],
    }


def validate_formal_scope_plan(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    path = _resolve_path(path, field="formal scope plan")
    if expected_file_sha256 is not None:
        if not isinstance(expected_file_sha256, str) or not _SHA256_RE.fullmatch(
            expected_file_sha256
        ):
            raise TableBContractError("formal scope plan SHA-256 is malformed")
        if sha256_file(path) != expected_file_sha256:
            raise TableBContractError("formal scope plan file SHA-256 drifted")
    plan = _json_object(path, label="Table-B v2 formal scope plan")
    if not (
        plan.get("schema") == FORMAL_SCOPE_PLAN_SCHEMA
        and plan.get("status") == "sealed"
        and plan.get("profile") == FORMAL_PROFILE
        and plan.get("ordered_run_ids") == list(FORMAL_RUN_IDS)
        and plan.get("semantic_sha256") == _semantic_sha256(plan)
    ):
        raise TableBContractError("formal scope plan identity/semantic SHA-256 drifted")
    source_plan_path = _verify_record(
        plan.get("source_plan"), role="table_b_v2_formal_source_plan"
    )
    source_plan = validate_formal_source_plan(source_plan_path)
    if not (
        plan.get("source_plan_semantic_sha256") == source_plan["semantic_sha256"]
        and plan.get("queue") == source_plan["queue"]
    ):
        raise TableBContractError("formal scope plan/source plan identity drifted")
    runs = plan.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(FORMAL_RUN_IDS):
        raise TableBContractError("formal scope plan run inventory/order drifted")
    for run_id in FORMAL_RUN_IDS:
        record = runs[run_id]
        if not (
            isinstance(record, Mapping)
            and record.get("run_id") == run_id
            and record.get("table_b_id") == run_id.split(":", 1)[0]
            and record.get("seed") == int(run_id.split(":", 1)[1])
            and isinstance(record.get("scope_sha256"), str)
            and bool(_SHA256_RE.fullmatch(str(record["scope_sha256"])))
            and isinstance(record.get("input_identity_sha256"), str)
            and bool(_SHA256_RE.fullmatch(str(record["input_identity_sha256"])))
            and isinstance(record.get("command_sha256"), str)
            and bool(_SHA256_RE.fullmatch(str(record["command_sha256"])))
        ):
            raise TableBContractError(f"formal scope plan run {run_id} is malformed")
    return json.loads(canonical_bytes(plan))


def _all_true(mapping: Any, *, label: str) -> None:
    if not isinstance(mapping, Mapping) or not mapping:
        raise TableBContractError(f"{label} invariants are missing")
    for key, value in mapping.items():
        if key == "runtime_global_verified_true_rows":
            if value != 0:
                raise TableBContractError(f"{label}.{key} must be zero")
        elif value is not True:
            raise TableBContractError(f"{label}.{key} did not pass")


@lru_cache(maxsize=1)
def _validated_v2_audit_cached() -> dict[str, Any]:
    audit_path = AUDIT_PATH.resolve(strict=True)
    observed_sha = sha256_file(audit_path)
    if observed_sha != AUDIT_SHA256:
        raise TableBContractError(
            f"v2 audit SHA-256 mismatch: expected {AUDIT_SHA256}, observed {observed_sha}"
        )
    audit = _json_object(audit_path, label="Table-B v2 audit")
    if not (
        audit.get("schema") == AUDIT_SCHEMA
        and audit.get("kind") == "completed_c2_parent_matched_tn_panel"
        and audit.get("claim_scope") == CLAIM_SCOPE
    ):
        raise TableBContractError("Table-B v2 audit schema/kind/claim scope drifted")
    runtime = audit.get("runtime_contract")
    if not (
        isinstance(runtime, Mapping)
        and runtime.get("D2m_D3m_supported_by_current_v24") is True
        and runtime.get("current_v24_supported_table_ids") == ["D2m", "D3m"]
    ):
        raise TableBContractError("Table-B v2 runtime contract is incomplete")
    invariants = audit.get("invariants")
    if not isinstance(invariants, Mapping):
        raise TableBContractError("Table-B v2 invariants are missing")
    _all_true(invariants.get("train"), label="train")
    _all_true(invariants.get("calibration"), label="calibration")
    for key in (
        "strict_union_image_overlap",
        "train_calibration_image_overlap",
        "train_calibration_pair_id_overlap",
    ):
        if invariants.get(key) != 0:
            raise TableBContractError(f"Table-B v2 invariant {key} must be zero")
    for key in (
        "unique_train_parent_keys",
        "unique_calibration_parent_keys",
        "train_matching_yield_partition",
        "calibration_matching_yield_partition",
    ):
        if invariants.get(key) is not True:
            raise TableBContractError(f"Table-B v2 invariant {key} did not pass")

    for table_b_id in TABLE_B_SCOPE_BY_ID:
        scope = audit.get("scope_contract", {}).get(table_b_id)
        if scope != {
            "tn_scope": TABLE_B_SCOPE_BY_ID[table_b_id],
            "global_tn_verified": False,
        }:
            raise TableBContractError(f"Table-B v2 {table_b_id} scope contract drifted")
        declared_dataset = audit.get("dataset_configs", {}).get(table_b_id)
        expected_dataset = file_record(
            DATASET_PATH_BY_ID[table_b_id], role=f"{table_b_id}:dataset_manifest"
        )
        if not isinstance(declared_dataset, Mapping):
            raise TableBContractError(f"Table-B v2 lacks {table_b_id} dataset record")
        if (
            _resolve_path(declared_dataset.get("path"), field="dataset_configs.path")
            != DATASET_PATH_BY_ID[table_b_id].resolve(strict=True)
            or declared_dataset.get("sha256") != DATASET_SHA256_BY_ID[table_b_id]
            or declared_dataset.get("sha256") != expected_dataset["sha256"]
            or declared_dataset.get("size_bytes") != expected_dataset["size_bytes"]
        ):
            raise TableBContractError(f"Table-B v2 {table_b_id} dataset record drifted")
        output_key = TRAIN_OUTPUT_KEY_BY_ID[table_b_id]
        output = audit.get("outputs", {}).get(output_key)
        if not isinstance(output, Mapping) or output.get("rows") != TRAIN_ROWS:
            raise TableBContractError(f"Table-B v2 output {output_key} is malformed")
        output_path = _resolve_path(output.get("path"), field=f"outputs.{output_key}.path")
        if (
            output.get("sha256") != sha256_file(output_path)
            or output.get("size_bytes") != output_path.stat().st_size
        ):
            raise TableBContractError(f"Table-B v2 output {output_key} drifted")
    return audit


def validate_v2_audit() -> dict[str, Any]:
    """Return a detached copy so callers cannot mutate cached authority."""

    return json.loads(canonical_bytes(_validated_v2_audit_cached()))


def validate_dataset_manifest(table_b_id: str, dataset_path: Path | None = None) -> dict[str, Any]:
    if table_b_id not in TABLE_B_SCOPE_BY_ID:
        raise TableBContractError(f"unsupported Table-B v2 ID {table_b_id!r}")
    validate_v2_audit()
    expected = DATASET_PATH_BY_ID[table_b_id].resolve(strict=True)
    path = expected if dataset_path is None else _resolve_path(dataset_path, field="dataset")
    if path != expected or sha256_file(path) != DATASET_SHA256_BY_ID[table_b_id]:
        raise TableBContractError(f"{table_b_id} must use the exact class-aligned v2 dataset")
    payload = _json_object(path, label=f"{table_b_id} dataset manifest")
    train = payload.get("train")
    if not isinstance(train, list) or len(train) != 4 or payload.get("val") != []:
        raise TableBContractError(f"{table_b_id} dataset must contain 4 train sources and val=[]")
    if [source.get("mix_weight", 1.0) for source in train] != [1.0, 1.0, 1.0, 3.0]:
        raise TableBContractError(f"{table_b_id} dataset mix weights drifted")
    tn = train[-1]
    expected_tn = {
        "source": "sam3_tn_pair",
        "require_global_tn_verified": False,
        "require_single_edit_token_provenance": False,
        "paper_table_b_id": table_b_id,
        "paper_tn_scope": TABLE_B_SCOPE_BY_ID[table_b_id],
        "paper_matched_causal_panel": True,
        "paper_runtime_supported": True,
        "paper_runtime_contract": RUNTIME_DATASET_CONTRACT,
    }
    if not isinstance(tn, Mapping) or any(tn.get(key) != value for key, value in expected_tn.items()):
        raise TableBContractError(f"{table_b_id} TN source contract drifted")
    audit_path = _resolve_path(tn.get("paper_contract_audit"), field="paper_contract_audit")
    if audit_path != AUDIT_PATH.resolve(strict=True):
        raise TableBContractError(f"{table_b_id} dataset points to a different audit")
    output = validate_v2_audit()["outputs"][TRAIN_OUTPUT_KEY_BY_ID[table_b_id]]
    annotation = _resolve_path(tn.get("anno"), field="dataset annotation")
    if annotation != _resolve_path(output.get("path"), field="audit training output"):
        raise TableBContractError(f"{table_b_id} annotation is not the audited train output")
    return payload


def build_scope_binding(
    *,
    table_b_id: str,
    seed: int,
    phase_id: str,
    dataset_path: Path,
    config_path: Path,
    runner_path: Path,
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if seed not in SEEDS or type(seed) is not int:
        raise TableBContractError(f"training seed must be one of {SEEDS}")
    if phase_id != PHASE_ID:
        raise TableBContractError(f"Table-B v2 phase_id must be {PHASE_ID!r}")
    validate_dataset_manifest(table_b_id, dataset_path)
    audit = validate_v2_audit()
    output_path = _resolve_path(
        audit["outputs"][TRAIN_OUTPUT_KEY_BY_ID[table_b_id]]["path"],
        field="training source",
    )
    if execution_context is None:
        context: dict[str, Any] = {
            "profile": NONFORMAL_PROFILE,
            "claim_class": "nonformal_table_b_v2_fixture",
            "training_contract": {"formal": False},
            "output_root": None,
            "queue": None,
            "source_plan": None,
            "source_plan_semantic_sha256": None,
        }
    else:
        context = json.loads(canonical_bytes(dict(execution_context)))
    binding = {
        "schema": SCOPE_SCHEMA,
        "contract_id": CONTRACT_ID,
        "profile": context.get("profile"),
        "claim_class": context.get("claim_class"),
        "table_b_id": table_b_id,
        "seed": seed,
        "phase_id": phase_id,
        "training_contract": context.get("training_contract"),
        "output_root": context.get("output_root"),
        "queue": context.get("queue"),
        "source_plan": context.get("source_plan"),
        "source_plan_semantic_sha256": context.get(
            "source_plan_semantic_sha256"
        ),
        "audit": file_record(AUDIT_PATH, role="table_b_v2_audit"),
        "dataset_manifest": file_record(dataset_path, role="table_b_v2_dataset_manifest"),
        "training_source": file_record(output_path, role="table_b_v2_training_source"),
        "config": file_record(config_path, role="table_b_v2_base_config"),
        "runner": file_record(runner_path, role="table_b_v2_runner"),
        "contract_source": file_record(Path(__file__), role="table_b_v2_contract_source"),
        "claim_scope": dict(CLAIM_SCOPE),
        "evidence": {
            "phase_id": phase_id,
            "table_b_id": table_b_id,
            "source_policy": "class_aligned_parent_matched_v2",
            "legacy_evidence_mutated": False,
            "profile": context.get("profile"),
            "formal_queue_bound": context.get("profile") == FORMAL_PROFILE,
        },
    }
    validate_scope_binding(binding)
    return binding


def validate_scope_binding(binding: Any, *, expected_sha256: str | None = None) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise TableBContractError("Table-B v2 process scope must be an object")
    table_b_id = binding.get("table_b_id")
    seed = binding.get("seed")
    phase_id = binding.get("phase_id")
    profile = binding.get("profile")
    expected_claim_class = (
        "supplemental_matched_source_policy_sensitivity"
        if profile == FORMAL_PROFILE
        else "nonformal_table_b_v2_fixture"
    )
    if not (
        binding.get("schema") == SCOPE_SCHEMA
        and binding.get("contract_id") == CONTRACT_ID
        and profile in {FORMAL_PROFILE, NONFORMAL_PROFILE}
        and binding.get("claim_class") == expected_claim_class
        and table_b_id in TABLE_B_SCOPE_BY_ID
        and type(seed) is int
        and seed in SEEDS
        and phase_id == PHASE_ID
        and binding.get("claim_scope") == CLAIM_SCOPE
    ):
        raise TableBContractError("Table-B v2 process scope identity drifted")
    evidence = binding.get("evidence")
    if not (
        isinstance(evidence, Mapping)
        and evidence.get("phase_id") == phase_id
        and evidence.get("table_b_id") == table_b_id
        and evidence.get("source_policy") == "class_aligned_parent_matched_v2"
        and evidence.get("legacy_evidence_mutated") is False
        and evidence.get("profile") == profile
        and evidence.get("formal_queue_bound") is (profile == FORMAL_PROFILE)
    ):
        raise TableBContractError("Table-B v2 nested scope evidence drifted")

    if profile == FORMAL_PROFILE:
        queue = binding.get("queue")
        if not (
            binding.get("training_contract") == FORMAL_TRAINING_CONTRACT
            and isinstance(binding.get("output_root"), str)
            and Path(str(binding["output_root"])).is_absolute()
            and isinstance(queue, Mapping)
            and isinstance(queue.get("queue_id"), str)
            and bool(queue["queue_id"])
            and isinstance(queue.get("plan_sha256"), str)
            and bool(_SHA256_RE.fullmatch(str(queue["plan_sha256"])))
            and isinstance(queue.get("queue_dir"), str)
            and isinstance(binding.get("source_plan_semantic_sha256"), str)
            and bool(
                _SHA256_RE.fullmatch(
                    str(binding["source_plan_semantic_sha256"])
                )
            )
        ):
            raise TableBContractError("formal Table-B v2 execution context is incomplete")
        source_plan_path = _verify_record(
            binding.get("source_plan"), role="table_b_v2_formal_source_plan"
        )
        source_plan = validate_formal_source_plan(source_plan_path)
        if not (
            source_plan["semantic_sha256"]
            == binding["source_plan_semantic_sha256"]
            and source_plan["queue"]["queue_id"] == queue["queue_id"]
            and source_plan["queue"]["plan_sha256"] == queue["plan_sha256"]
            and Path(source_plan["queue"]["queue_dir"]).resolve()
            == Path(queue["queue_dir"]).resolve()
            and Path(source_plan["runtime"]["output_root"]).resolve()
            == Path(binding["output_root"]).resolve()
        ):
            raise TableBContractError("formal scope/source-plan identity drifted")
    elif not (
        binding.get("training_contract") == {"formal": False}
        and binding.get("output_root") is None
        and binding.get("queue") is None
        and binding.get("source_plan") is None
        and binding.get("source_plan_semantic_sha256") is None
    ):
        raise TableBContractError("nonformal Table-B v2 scope gained formal authority")
    audit_path = _verify_record(binding.get("audit"), role="table_b_v2_audit")
    dataset_path = _verify_record(
        binding.get("dataset_manifest"), role="table_b_v2_dataset_manifest"
    )
    training_path = _verify_record(
        binding.get("training_source"), role="table_b_v2_training_source"
    )
    _verify_record(binding.get("config"), role="table_b_v2_base_config")
    _verify_record(binding.get("runner"), role="table_b_v2_runner")
    _verify_record(binding.get("contract_source"), role="table_b_v2_contract_source")
    if audit_path != AUDIT_PATH.resolve(strict=True):
        raise TableBContractError("Table-B v2 scope audit path drifted")
    validate_dataset_manifest(str(table_b_id), dataset_path)
    output = validate_v2_audit()["outputs"][TRAIN_OUTPUT_KEY_BY_ID[str(table_b_id)]]
    if training_path != _resolve_path(output["path"], field="audited training source"):
        raise TableBContractError("Table-B v2 scope training source drifted")
    observed_sha = canonical_sha256(binding)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
            raise TableBContractError("scope SHA-256 must be 64 lowercase hex characters")
        if observed_sha != expected_sha256:
            raise TableBContractError("Table-B v2 process scope SHA-256 mismatch")
    return binding


def establish_process_scope(binding: dict[str, Any], expected_sha256: str) -> dict[str, Any]:
    """Bind one immutable v2 scope before any training-stack module is imported."""

    forbidden = tuple(name for name in TRAINING_IMPORT_NAMES if name in sys.modules)
    if forbidden:
        raise TableBContractError(
            "cannot establish Table-B v2 scope after training imports: " + ", ".join(forbidden)
        )
    validated = validate_scope_binding(binding, expected_sha256=expected_sha256)
    global _PROCESS_SCOPE, _PROCESS_SCOPE_SHA256
    if _PROCESS_SCOPE_SHA256 is not None and _PROCESS_SCOPE_SHA256 != expected_sha256:
        raise TableBContractError("process is already bound to a different Table-B v2 scope")
    inherited = os.environ.get(SCOPE_SHA_ENV)
    if inherited not in (None, "", expected_sha256):
        raise TableBContractError("inherited Table-B v2 scope identity conflicts")
    _PROCESS_SCOPE = json.loads(canonical_bytes(validated))
    _PROCESS_SCOPE_SHA256 = expected_sha256
    os.environ[SCOPE_SHA_ENV] = expected_sha256
    return json.loads(canonical_bytes(validated))


def install_as_training_contract() -> None:
    """Install this scoped implementation before ``engine``/dataset imports."""

    if _PROCESS_SCOPE is None or _PROCESS_SCOPE_SHA256 is None:
        raise TableBContractError("Table-B v2 process scope has not been established")
    current = sys.modules.get(LEGACY_IMPORT_NAME)
    this_module = sys.modules[__name__]
    if current is not None and current is not this_module:
        raise TableBContractError("legacy Table-B contract was imported before v2 bootstrap")
    sys.modules[LEGACY_IMPORT_NAME] = this_module


def require_process_scope(*, table_b_id: str | None = None, phase_id: str = PHASE_ID) -> dict[str, Any]:
    if _PROCESS_SCOPE is None or _PROCESS_SCOPE_SHA256 is None:
        raise TableBContractError("Table-B v2 runtime used outside its guarded process scope")
    validate_scope_binding(_PROCESS_SCOPE, expected_sha256=_PROCESS_SCOPE_SHA256)
    if table_b_id is not None and _PROCESS_SCOPE.get("table_b_id") != table_b_id:
        raise TableBContractError("Table-B v2 runtime table ID differs from process scope")
    if _PROCESS_SCOPE.get("phase_id") != phase_id:
        raise TableBContractError("Table-B v2 runtime phase_id differs from process scope")
    return json.loads(canonical_bytes(_PROCESS_SCOPE))


def _exact_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TableBContractError(f"{field} must be an exact boolean")
    return value


def table_b_contract_from_args(args: Any) -> TableBConfidenceContract | None:
    enabled = _exact_bool(
        getattr(args, "stage_b_v19_allow_scope_labeled_tn_ablation", False),
        field="stage_b_v19_allow_scope_labeled_tn_ablation",
    )
    if not enabled:
        raise TableBContractError("Table-B v2 runner forbids disabling scoped supervision")
    table_b_id = getattr(args, "stage_b_v19_table_b_id", None)
    scope = require_process_scope(table_b_id=table_b_id)
    if getattr(args, "stage_b_v2_scope_contract_sha256", None) != _PROCESS_SCOPE_SHA256:
        raise TableBContractError("training args do not bind the guarded v2 scope SHA-256")
    if getattr(args, "stage_b_v2_phase_id", None) != PHASE_ID:
        raise TableBContractError("training args do not bind phase_id='joint'")
    if getattr(args, "stage_b_v2_profile", None) != scope["profile"]:
        raise TableBContractError("training args do not bind the v2 execution profile")
    if scope["profile"] == FORMAL_PROFILE:
        queue = scope["queue"]
        expected_args = {
            "stage_b_v2_training_queue_id": queue["queue_id"],
            "stage_b_v2_training_queue_plan_sha256": queue["plan_sha256"],
            "stage_b_v2_formal_source_plan_sha256": scope[
                "source_plan_semantic_sha256"
            ],
        }
        if any(getattr(args, key, None) != value for key, value in expected_args.items()):
            raise TableBContractError("training args lost the formal queue/source-plan binding")
    else:
        for key in (
            "stage_b_v2_training_queue_id",
            "stage_b_v2_training_queue_plan_sha256",
            "stage_b_v2_formal_source_plan_sha256",
        ):
            if getattr(args, key, None) not in (None, "", "none"):
                raise TableBContractError("nonformal v2 args unexpectedly claim formal authority")
    if getattr(args, "stage_b_v14_global_tn_all_candidates", None) is not True:
        raise TableBContractError("Table-B v2 requires global candidate confidence scoring")
    for field in (
        "stage_b_v15_decoupled_confidence",
        "stage_b_v19_explicit_confidence_output_contract",
    ):
        if getattr(args, field, None) is not True:
            raise TableBContractError(f"Table-B v2 requires {field}=True")
    if getattr(args, "stage_b_v16_confidence_output_mode", None) != "base_plus_gate":
        raise TableBContractError("Table-B v2 requires base_plus_gate confidence output")
    if table_b_id not in TABLE_B_SCOPE_BY_ID:
        raise TableBContractError("Table-B v2 ID must be D2m or D3m")
    expected_scope = TABLE_B_SCOPE_BY_ID[table_b_id]
    allowlist = getattr(args, "stage_b_v19_table_b_scope_allowlist", None)
    if not isinstance(allowlist, (list, tuple)) or tuple(allowlist) != (expected_scope,):
        raise TableBContractError(f"Table-B v2 {table_b_id} requires one exact scope")
    if _exact_bool(
        getattr(args, "stage_b_v19_table_b_allow_single_edit_token_provenance", False),
        field="stage_b_v19_table_b_allow_single_edit_token_provenance",
    ):
        raise TableBContractError("Table-B v2 keeps edit-token provenance disabled")
    audit_path = _resolve_path(
        getattr(args, "stage_b_v19_table_b_audit", None),
        field="stage_b_v19_table_b_audit",
    )
    if audit_path != AUDIT_PATH.resolve(strict=True):
        raise TableBContractError("training args bind a different Table-B v2 audit")
    if getattr(args, "stage_b_v19_table_b_audit_sha256", None) != AUDIT_SHA256:
        raise TableBContractError("training args bind a different Table-B v2 audit SHA-256")
    output = validate_v2_audit()["outputs"][TRAIN_OUTPUT_KEY_BY_ID[table_b_id]]
    return TableBConfidenceContract(
        table_b_id=table_b_id,
        scope=expected_scope,
        scope_allowlist=(expected_scope,),
        audit_path=audit_path,
        audit_sha256=AUDIT_SHA256,
        train_path=_resolve_path(output["path"], field="audited train output"),
        train_sha256=str(output["sha256"]),
        allow_single_edit_token_provenance=False,
    )


def validate_table_b_dataset_binding(
    args: Any, datasetinfo: Mapping[str, Any]
) -> TableBConfidenceContract | None:
    contract = table_b_contract_from_args(args)
    source = str(datasetinfo.get("source", "")).strip().lower()
    declared_id = datasetinfo.get("paper_table_b_id")
    has_binding = declared_id is not None or any(
        key in datasetinfo for key in ("paper_tn_scope", "paper_contract_audit")
    )
    if not has_binding:
        if source in _PAIRED_SOURCES:
            raise TableBContractError("paired TN source lacks its Table-B v2 binding")
        return None
    if source not in _PAIRED_SOURCES or declared_id != contract.table_b_id:
        raise TableBContractError("Table-B v2 dataset source/ID mismatch")
    if datasetinfo.get("paper_tn_scope") != contract.scope:
        raise TableBContractError("Table-B v2 dataset scope mismatch")
    if datasetinfo.get("paper_runtime_contract") != RUNTIME_DATASET_CONTRACT:
        raise TableBContractError("Table-B v2 dataset runtime contract mismatch")
    if _resolve_path(datasetinfo.get("paper_contract_audit"), field="paper_contract_audit") != contract.audit_path:
        raise TableBContractError("Table-B v2 dataset audit mismatch")
    if datasetinfo.get("require_global_tn_verified") is not False:
        raise TableBContractError("Table-B v2 must retain global_tn_verified=false")
    if datasetinfo.get("require_single_edit_token_provenance") is not False:
        raise TableBContractError("Table-B v2 must keep edit-token provenance disabled")
    annotation = _resolve_path(datasetinfo.get("anno"), field="dataset annotation")
    if annotation != contract.train_path or sha256_file(annotation) != contract.train_sha256:
        raise TableBContractError("Table-B v2 dataset annotation drifted")
    return contract


def _target_scalar_bool(target: Mapping[str, Any], key: str, *, index: int) -> bool:
    value = target.get(key)
    if not (torch.is_tensor(value) and value.dtype == torch.bool and value.numel() == 1):
        raise TableBContractError(f"Table-B target {index} requires scalar bool tensor {key!r}")
    return bool(value.reshape(-1)[0].item())


def build_confidence_ablation_eligible(
    args: Any,
    targets: Sequence[Mapping[str, Any]],
    paired_tn: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    contract = table_b_contract_from_args(args)
    if not torch.is_tensor(paired_tn) or paired_tn.dtype != torch.bool:
        raise TableBContractError("paired_tn must be a boolean tensor")
    paired = paired_tn.to(device=device, dtype=torch.bool).reshape(-1)
    if paired.numel() != len(targets):
        raise TableBContractError("paired_tn/target length mismatch")
    eligible = torch.zeros(len(targets), dtype=torch.bool, device=device)
    for index, (target, is_paired) in enumerate(zip(targets, paired.tolist())):
        if not is_paired:
            if target.get("table_b_id") is not None:
                raise TableBContractError("non-paired target carries a Table-B v2 ID")
            continue
        if (
            target.get("table_b_id") != contract.table_b_id
            or target.get("tn_scope") != contract.scope
            or target.get("table_b_audit_sha256") != contract.audit_sha256
        ):
            raise TableBContractError(f"paired target {index} changed its v2 scope binding")
        if _target_scalar_bool(target, "global_tn_verified", index=index):
            raise TableBContractError("weak-scope v2 target was upgraded to global TN")
        eligible[index] = True
    return eligible


__all__ = [
    "AUDIT_PATH",
    "AUDIT_SCHEMA",
    "AUDIT_SHA256",
    "CLAIM_SCOPE",
    "CONTRACT_ID",
    "DATASET_PATH_BY_ID",
    "DATASET_SHA256_BY_ID",
    "FORMAL_BATCH_SIZE",
    "FORMAL_CHECKPOINT_INTERVAL",
    "FORMAL_GENERIC_EXTENSION_SCHEMA",
    "FORMAL_PROFILE",
    "FORMAL_RUN_IDS",
    "FORMAL_SCOPE_PLAN_ENV",
    "FORMAL_SCOPE_PLAN_SCHEMA",
    "FORMAL_SOURCE_PLAN_ENV",
    "FORMAL_SOURCE_PLAN_SCHEMA",
    "FORMAL_SUCCESSFUL_UPDATE_BATCH_SLOTS",
    "FORMAL_TRAINING_CONTRACT",
    "FORMAL_TRAIN_UPDATES",
    "NONFORMAL_PROFILE",
    "PHASE_ID",
    "SCOPE_SCHEMA",
    "SEEDS",
    "TABLE_B_AUDIT_SCHEMA",
    "TABLE_B_MATCHED_PAIR_SCHEMA",
    "TABLE_B_PAIR_SCHEMA",
    "TABLE_B_SCOPE_BY_ID",
    "TRAINING_IMPORT_NAMES",
    "TableBConfidenceContract",
    "TableBContractError",
    "build_confidence_ablation_eligible",
    "build_scope_binding",
    "canonical_sha256",
    "establish_process_scope",
    "formal_context_from_source_plan",
    "install_as_training_contract",
    "require_process_scope",
    "sha256_file",
    "table_b_contract_from_args",
    "validate_dataset_manifest",
    "validate_formal_source_plan",
    "validate_formal_scope_plan",
    "validate_scope_binding",
    "validate_table_b_dataset_binding",
    "validate_v2_audit",
]
