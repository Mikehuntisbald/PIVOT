#!/usr/bin/env python3
"""Plan or run Table-A continued-GDINO and print its matched eval command."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_stageb_table_a_continued_gdino import verify as verify_data  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_table_a_g0c_continued_gdino.py"
CONFIG_CHAIN = (
    REPO_ROOT / "config/ablations/cfg_stageb_from_gdino_ft_with_tn.py",
    REPO_ROOT / "config/cfg_odvg.py",
)
SOURCE_DEPENDENCY_ROOTS = (
    REPO_ROOT / "models/GroundingDINO",
    REPO_ROOT / "datasets",
    REPO_ROOT / "groundingdino",
)
SOURCE_DEPENDENCY_FILES = (
    REPO_ROOT / "models/__init__.py",
    REPO_ROOT / "models/registry.py",
    REPO_ROOT / "util/get_param_dicts.py",
    REPO_ROOT / "util/misc.py",
    REPO_ROOT / "util/slconfig.py",
)
NATIVE_OP_ROOT = REPO_ROOT / "models/GroundingDINO/ops"
SOAK_COMPATIBILITY_SCHEMA = "pivot.stageb.table_a.g0c_soak_compatibility/v1"
_DISTRIBUTED_ENV_KEYS = {
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "SLURM_LOCALID",
    "SLURM_NTASKS",
    "SLURM_NPROCS",
    "SLURM_NODELIST",
}
DATASET = REPO_ROOT / "config/datasets_stageb_table_a_g0c_continued_gdino.json"
AUDIT = REPO_ROOT / "data/ablations/stageb_table_a_continued_gdino_20260717/audit.json"
DEFAULT_CHECKPOINT = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
REQUIRED_EFFECTIVE_GLOBAL_BATCH = 40
FORMAL_SEEDS = (17, 42, 73)
FORMAL_MICRO_BATCH_SIZE = 10
FORMAL_GRADIENT_ACCUMULATION_STEPS = 4
FORMAL_OPTIMIZER_UPDATES = 1000
SOAK_OPTIMIZER_UPDATES = 50
MINIMUM_SOAK_HEADROOM_MIB = 1024.0
PLAN_SCHEMA = "stageb-table-a-g0c-launch-plan-v3"
POSTFLIGHT_SCHEMA = "stageb-table-a-g0c-postflight-v2"
SOAK_SEAL_SCHEMA = "pivot.stageb.table_a.g0c_soak_seal/v1"
DEFAULT_SOAK_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/memory_soaks/"
    "table_a_g0c_b10a4_u50_formal_v2/seed17"
)
DEFAULT_SOAK_PLAN = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/plans/"
    "table_a_g0c_b10a4_u50_formal_v2_seed17.json"
)
DEFAULT_SOAK_SEAL = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/memory_probes/"
    "table_a_g0c_b10a4_u50_formal_v2_seal.json"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_sha256", None)
    return _canonical_json_sha256(payload)


def _file_record(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _require_current_python_runtime(value: Path) -> Path:
    selected = value.expanduser().resolve(strict=True)
    current = Path(sys.executable).resolve(strict=True)
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise ValueError(f"G0c selected Python is not executable: {selected}")
    if selected != current:
        raise ValueError(
            "G0c planning must run under the selected Python so the imported "
            f"native extension is sealed correctly: caller={current}, selected={selected}"
        )
    return selected


def formal_output_root(seed: int) -> Path:
    return REPO_ROOT / f"outputs/paper_cvpr_v1/table_a/G0c/seed{int(seed)}"


def formal_plan_path(seed: int) -> Path:
    return REPO_ROOT / (
        "outputs/paper_cvpr_v1/plans/"
        f"table_a_g0c_seed{int(seed)}_b10a4_effective40_u1000_v3.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_dependency_tree() -> Dict[str, Any]:
    from tools.stageb_dependency_audit import local_python_dependency_paths

    explicit = [
        Path(__file__).resolve(),
        REPO_ROOT / "main.py",
        REPO_ROOT / "engine.py",
        CONFIG,
        *CONFIG_CHAIN,
        *SOURCE_DEPENDENCY_FILES,
    ]
    for root in SOURCE_DEPENDENCY_ROOTS:
        explicit.extend(root.rglob("*.py"))
    python_files = local_python_dependency_paths(
        [Path(__file__).resolve(), REPO_ROOT / "main.py", REPO_ROOT / "engine.py"],
        root=REPO_ROOT,
        include=explicit,
    )
    native_files = _native_runtime_dependency_paths()
    files = sorted(set(python_files).union(native_files), key=_dependency_label)
    digest = hashlib.sha256()
    records = []
    for path in files:
        relative = _dependency_label(path)
        sha256 = _sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
        records.append(
            {
                "path": str(path),
                "relative_path": relative,
                "sha256": sha256,
                "size_bytes": int(path.stat().st_size),
                "dependency_kind": (
                    "python"
                    if path.suffix == ".py"
                    else "native_extension"
                    if path.suffix == ".so"
                    else "native_source"
                    if path.suffix in {".cpp", ".cu", ".h"}
                    else "native_build_metadata"
                ),
            }
        )
    return {
        "algorithm": "recursive-python-plus-native-runtime-closure-v2",
        "file_count": len(files),
        "python_file_count": len(python_files),
        "native_file_count": len(native_files),
        "actual_native_extension": _file_record(_actual_native_extension_path()),
        "sha256": digest.hexdigest(),
        "records": records,
    }


def _native_runtime_dependency_paths() -> list[Path]:
    """Bind the compiled MSDeformAttn implementation and its build lineage."""

    patterns = (
        "MultiScaleDeformableAttention*.so",
        "build/**/MultiScaleDeformableAttention*.so",
        "src/**/*.cpp",
        "src/**/*.cu",
        "src/**/*.h",
        "build/**/build.ninja",
        "MultiScaleDeformableAttention.egg-info/*",
    )
    paths = {
        path.resolve()
        for pattern in patterns
        for path in NATIVE_OP_ROOT.glob(pattern)
        if path.is_file()
    }
    actual_extension = _actual_native_extension_path()
    paths.add(actual_extension)
    for metadata in actual_extension.parent.glob(
        "MultiScaleDeformableAttention-*.egg-info/*"
    ):
        if metadata.is_file():
            paths.add(metadata.resolve())
    root_extensions = [
        path
        for path in paths
        if path.parent == NATIVE_OP_ROOT.resolve()
        and path.name.startswith("MultiScaleDeformableAttention")
        and path.suffix == ".so"
    ]
    source_suffixes = {path.suffix for path in paths}
    if len(root_extensions) != 1:
        raise ValueError(
            "G0c requires exactly one importable root MultiScaleDeformableAttention extension"
        )
    if not {".cpp", ".cu", ".h"}.issubset(source_suffixes):
        raise ValueError("G0c native source closure is incomplete")
    if not any(path.name == "build.ninja" for path in paths):
        raise ValueError("G0c native build metadata is missing")
    return sorted(paths, key=_dependency_label)


def _dependency_label(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return "external-runtime:" + str(path)


def _actual_native_extension_path() -> Path:
    spec = importlib.util.find_spec("MultiScaleDeformableAttention")
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not isinstance(origin, str) or not origin:
        raise ValueError("the runtime MultiScaleDeformableAttention extension is missing")
    path = Path(origin).resolve(strict=True)
    if not path.is_file() or path.suffix != ".so":
        raise ValueError("the runtime MultiScaleDeformableAttention origin is not a .so")
    return path


def _registered_training_sources() -> Dict[str, Dict[str, str]]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = {}
    for index, source in enumerate(audit.get("sources", [])):
        path = Path(source["output"]).resolve(strict=True)
        expected = str(source["output_sha256"])
        if _sha256(path) != expected:
            raise ValueError(f"continued-GDINO source {index} differs from its audit")
        records[f"training_jsonl_{index}"] = {
            "path": str(path),
            "sha256": expected,
        }
    if len(records) != 4:
        raise ValueError("G0c requires exactly four registered training JSONL files")
    return records


def _single_process_environment(
    environment: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    for key in _DISTRIBUTED_ENV_KEYS:
        result.pop(key, None)
    return result


def _validate(checkpoint: Path) -> Dict[str, Any]:
    verify_data(dataset=DATASET, audit_path=AUDIT)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("G0c must start from the registered historical b58 checkpoint")
    cfg = SLConfig.fromfile(str(CONFIG))
    expected = {
        "paper_table_a_id": "G0c",
        "patch_only": False,
        "stage_b": False,
        "enable_patch_branch": False,
        "data_aug_hflip_prob": 0.0,
        "skip_eval": True,
        "amp_init_scale": 512.0,
        "amp_max_consecutive_skips": 8,
    }
    for key, value in expected.items():
        if getattr(cfg, key, None) != value:
            raise ValueError(f"G0c config drifted at {key}: {getattr(cfg, key, None)!r}")
    return {"checkpoint_sha256": checkpoint_sha, "config_contract": expected}


def resolve_batch_contract(
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    effective_global_batch: int = REQUIRED_EFFECTIVE_GLOBAL_BATCH,
) -> Dict[str, int]:
    micro_batch_size = int(micro_batch_size)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    effective_global_batch = int(effective_global_batch)
    if micro_batch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("micro batch size and gradient accumulation steps must be positive")
    observed = micro_batch_size * gradient_accumulation_steps
    if effective_global_batch != REQUIRED_EFFECTIVE_GLOBAL_BATCH:
        raise ValueError(
            f"G0c paper contract fixes effective global batch to "
            f"{REQUIRED_EFFECTIVE_GLOBAL_BATCH}, got {effective_global_batch}"
        )
    if observed != effective_global_batch:
        raise ValueError(
            "G0c effective global batch mismatch: "
            f"micro_batch_size={micro_batch_size} * "
            f"gradient_accumulation_steps={gradient_accumulation_steps} "
            f"= {observed}, expected {effective_global_batch}"
        )
    return {
        "world_size": 1,
        "distributed_environment_scrubbed": True,
        "micro_batch_size_per_rank": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch": observed,
    }


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_artifact_record(value: Any, *, label: str) -> Path:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} artifact record is missing")
    path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    if int(value.get("size_bytes", -1)) != int(path.stat().st_size):
        raise ValueError(f"{label} size changed")
    if str(value.get("sha256", "")) != _sha256(path):
        raise ValueError(f"{label} SHA-256 changed")
    return path


def _validate_plan_identity(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"G0c plan schema must be exactly {PLAN_SCHEMA}")
    expected = str(plan.get("plan_sha256", ""))
    if len(expected) != 64 or expected != _plan_sha256(plan):
        raise ValueError("G0c plan canonical SHA-256 mismatch")


def _normalized_soak_command(plan: Mapping[str, Any]) -> list[str]:
    command = plan.get("command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise ValueError("G0c soak compatibility requires an exact command list")
    normalized = list(command)
    for flag in (
        "--output_dir",
        "--seed",
        "--max_train_iters",
        "--iter_checkpoint_interval",
        "--note",
    ):
        if normalized.count(flag) != 1:
            raise ValueError(f"G0c command requires exactly one {flag}")
        index = normalized.index(flag)
        if index + 1 >= len(normalized):
            raise ValueError(f"G0c command has no value for {flag}")
        normalized[index + 1] = f"<{flag[2:]}>"
    return normalized


def _soak_semantic_contract(plan: Mapping[str, Any]) -> Dict[str, Any]:
    inputs = plan.get("inputs")
    matched = plan.get("matched_contract")
    if not isinstance(inputs, Mapping) or not isinstance(matched, Mapping):
        raise ValueError("G0c soak compatibility contract is incomplete")
    ignored_matched = {
        "optimizer_updates",
        "planned_micro_batches_without_amp_skips",
        "expected_checkpoint_iteration",
        "seed",
    }
    stable_inputs = {
        str(key): copy.deepcopy(value)
        for key, value in inputs.items()
        if key != "g0c_soak_seal"
    }
    return {
        "schema": SOAK_COMPATIBILITY_SCHEMA,
        "row_id": plan.get("row_id"),
        "inputs": stable_inputs,
        "source_dependency_tree": copy.deepcopy(plan.get("source_dependency_tree")),
        "matched_contract": {
            str(key): copy.deepcopy(value)
            for key, value in matched.items()
            if key not in ignored_matched
        },
        "normalized_command": _normalized_soak_command(plan),
        "runtime_evidence_required": plan.get("runtime_evidence_required"),
        "cuda_visible_devices": plan.get("cuda_visible_devices"),
    }


def _validate_formal_soak_compatibility(
    formal_plan: Mapping[str, Any], soak_plan: Mapping[str, Any]
) -> Dict[str, Any]:
    if formal_plan.get("purpose") != "formal" or soak_plan.get("purpose") != "soak":
        raise ValueError("G0c soak compatibility received the wrong plan purposes")
    formal = _soak_semantic_contract(formal_plan)
    soak = _soak_semantic_contract(soak_plan)
    if formal != soak:
        differing = sorted(
            key for key in set(formal).union(soak) if formal.get(key) != soak.get(key)
        )
        raise ValueError(
            "G0c formal plan is not semantically identical to its sealed soak: "
            + ", ".join(differing)
        )
    digest = _canonical_json_sha256(formal)
    return {
        "schema": SOAK_COMPATIBILITY_SCHEMA,
        "status": "passed",
        "semantic_sha256": digest,
        "allowed_differences": [
            "purpose",
            "seed",
            "optimizer_updates",
            "planned_micro_batches_without_amp_skips",
            "expected_checkpoint_iteration",
            "output_dir",
            "command.dynamic_seed_update_output_note_values",
        ],
    }


def _validate_soak_seal(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    seal = _read_json(path, label="G0c soak seal")
    if seal.get("schema") != SOAK_SEAL_SCHEMA or seal.get("status") != "sealed":
        raise ValueError("G0c formal run requires the v1 sealed-soak schema")
    contract = seal.get("contract")
    expected_contract = {
        "purpose": "soak",
        "seed": 17,
        "micro_batch_size_per_rank": FORMAL_MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch": REQUIRED_EFFECTIVE_GLOBAL_BATCH,
        "optimizer_updates": SOAK_OPTIMIZER_UPDATES,
        "minimum_headroom_mib": MINIMUM_SOAK_HEADROOM_MIB,
        "required_amp_skips": 0,
        "finite_losses_required": True,
    }
    if contract != expected_contract:
        raise ValueError("G0c soak seal contract drifted")
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("G0c soak seal has no artifacts")
    observed_paths = {
        key: _verify_artifact_record(value, label=f"G0c soak {key}")
        for key, value in artifacts.items()
    }
    required = {
        "plan",
        "postflight",
        "checkpoint",
        "native_info_log",
        "train_console_log",
        "gpu_environment",
        "gpu_telemetry",
        "gpu_telemetry_summary",
    }
    if set(observed_paths) != required:
        raise ValueError("G0c soak seal artifact set is not exact")
    if observed_paths["plan"] != DEFAULT_SOAK_PLAN.resolve(strict=True):
        raise ValueError("G0c soak seal does not bind the canonical soak plan")
    if observed_paths["checkpoint"] != (
        DEFAULT_SOAK_ROOT / "checkpoint_iter.pth"
    ).resolve(strict=True):
        raise ValueError("G0c soak seal checkpoint root is not canonical")
    expected_root_paths = {
        "postflight": DEFAULT_SOAK_ROOT / "postflight.json",
        "native_info_log": DEFAULT_SOAK_ROOT / "info.txt",
        "train_console_log": DEFAULT_SOAK_ROOT / "train_console.log",
        "gpu_environment": DEFAULT_SOAK_ROOT / "gpu_environment.json",
        "gpu_telemetry": DEFAULT_SOAK_ROOT / "gpu_telemetry.csv",
        "gpu_telemetry_summary": DEFAULT_SOAK_ROOT / "gpu_telemetry_summary.json",
    }
    for key, expected_path in expected_root_paths.items():
        if observed_paths[key] != expected_path.resolve(strict=True):
            raise ValueError(f"G0c soak seal {key} path is not canonical")
    soak_plan = _read_json(observed_paths["plan"], label="sealed G0c soak plan")
    _validate_plan_identity(soak_plan)
    if soak_plan.get("purpose") != "soak":
        raise ValueError("sealed G0c soak plan has the wrong purpose")
    postflight = _read_json(
        observed_paths["postflight"], label="sealed G0c soak postflight"
    )
    if (
        postflight.get("schema") != POSTFLIGHT_SCHEMA
        or postflight.get("status") != "PASS"
        or postflight.get("purpose") != "soak"
        or postflight.get("plan_sha256") != soak_plan.get("plan_sha256")
    ):
        raise ValueError("sealed G0c soak postflight identity mismatch")
    if seal.get("plan_sha256") != soak_plan.get("plan_sha256"):
        raise ValueError("G0c soak seal plan SHA-256 differs from its plan")
    if postflight.get("checkpoint") != artifacts.get("checkpoint"):
        raise ValueError("G0c soak seal checkpoint differs from postflight")
    runtime_artifacts = postflight.get("runtime_artifacts")
    expected_runtime_artifacts = {
        key: artifacts[key]
        for key in (
            "native_info_log",
            "train_console_log",
            "gpu_environment",
            "gpu_telemetry",
            "gpu_telemetry_summary",
        )
    }
    if runtime_artifacts != expected_runtime_artifacts:
        raise ValueError("G0c soak seal runtime artifacts differ from postflight")
    numerical = postflight.get("numerical_status")
    if (
        not isinstance(numerical, Mapping)
        or numerical.get("status") != "passed"
        or numerical.get("loss_values_all_finite") is not True
        or float(numerical.get("max_amp_step_skipped", 1.0)) != 0.0
    ):
        raise ValueError("sealed G0c soak lacks finite zero-skip evidence")
    telemetry = postflight.get("gpu_telemetry_summary")
    persisted_telemetry = _read_json(
        observed_paths["gpu_telemetry_summary"],
        label="sealed G0c GPU telemetry summary",
    )
    if telemetry != persisted_telemetry:
        raise ValueError("sealed G0c telemetry payload differs from its artifact")
    devices = telemetry.get("devices") if isinstance(telemetry, Mapping) else None
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError("sealed G0c soak must contain one GPU telemetry device")
    headroom = float(devices[0].get("min_free_memory_mib", -1.0))
    if not math.isfinite(headroom) or headroom < MINIMUM_SOAK_HEADROOM_MIB:
        raise ValueError("sealed G0c soak does not meet the headroom requirement")
    if not math.isclose(
        float(seal.get("observed_minimum_free_memory_mib", -1.0)),
        headroom,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("G0c soak seal headroom differs from postflight telemetry")
    gpu_identity = seal.get("gpu_identity")
    expected_identity = {
        key: devices[0].get(key)
        for key in ("uuid", "name", "driver_version", "total_memory_mib")
    }
    if gpu_identity != expected_identity:
        raise ValueError("G0c soak seal GPU identity differs from telemetry")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "payload": dict(seal),
        "plan": dict(soak_plan),
    }


def _purpose_contract(args: argparse.Namespace, output_dir: Path) -> str:
    purpose = str(getattr(args, "purpose", "formal"))
    if purpose not in {"formal", "soak", "probe"}:
        raise ValueError("G0c purpose must be formal, soak, or probe")
    if purpose in {"formal", "soak"}:
        expected_updates = (
            FORMAL_OPTIMIZER_UPDATES if purpose == "formal" else SOAK_OPTIMIZER_UPDATES
        )
        expected_root = (
            formal_output_root(int(args.seed)) if purpose == "formal" else DEFAULT_SOAK_ROOT
        ).resolve(strict=False)
        if int(args.seed) not in (FORMAL_SEEDS if purpose == "formal" else (17,)):
            raise ValueError(f"G0c {purpose} seed is not predeclared")
        if (
            int(args.batch_size) != FORMAL_MICRO_BATCH_SIZE
            or int(args.gradient_accumulation_steps)
            != FORMAL_GRADIENT_ACCUMULATION_STEPS
            or int(args.effective_batch_size) != REQUIRED_EFFECTIVE_GLOBAL_BATCH
            or int(args.updates) != expected_updates
        ):
            raise ValueError(
                f"G0c {purpose} requires b10 x accumulation4 = 40 and "
                f"exactly {expected_updates} optimizer updates"
            )
        if output_dir != expected_root:
            raise ValueError(
                f"G0c {purpose} output root must be canonical: {expected_root}"
            )
    return purpose


def build_plan(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    validation = _validate(checkpoint)
    if int(args.updates) <= 0:
        raise ValueError("updates must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    purpose = _purpose_contract(args, output_dir)
    batch_contract = resolve_batch_contract(
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_global_batch=args.effective_batch_size,
    )
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    dataset_rows = sum(int(source["rows"]) for source in audit["sources"])
    micro_batches_per_epoch = dataset_rows // batch_contract[
        "micro_batch_size_per_rank"
    ]
    planned_micro_batches = (
        int(args.updates) * batch_contract["gradient_accumulation_steps"]
    )
    if planned_micro_batches >= micro_batches_per_epoch:
        raise ValueError(
            "G0c paper runs must terminate inside epoch 0 so the registered "
            "micro-batch arithmetic remains exact: "
            f"planned={planned_micro_batches}, epoch_capacity={micro_batches_per_epoch}"
        )
    python = _require_current_python_runtime(Path(args.python))
    command = [
        str(python),
        str(REPO_ROOT / "main.py"),
        "-c",
        str(CONFIG),
        "--datasets",
        str(DATASET),
        "--output_dir",
        str(output_dir),
        "--pretrain_model_path",
        str(checkpoint),
        "--seed",
        str(int(args.seed)),
        "--num_workers",
        str(int(args.num_workers)),
        "--world_size",
        "1",
        "--rank",
        "0",
        "--max_train_iters",
        str(int(args.updates)),
        "--gradient_accumulation_steps",
        str(batch_contract["gradient_accumulation_steps"]),
        "--iter_checkpoint_interval",
        str(int(args.updates)),
        "--note",
        f"paper_cvpr_v1_G0c_seed{int(args.seed)}",
        "--amp",
        "--save_log",
        "--options",
        f"batch_size={batch_contract['micro_batch_size_per_rank']}",
        "skip_eval=True",
    ]
    if "--resume" in command:
        raise AssertionError("G0c must not resume optimizer state")
    inputs = {
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        **{
            f"config_parent_{index}": {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for index, path in enumerate(CONFIG_CHAIN)
        },
        "dataset": {"path": str(DATASET), "sha256": _sha256(DATASET)},
        "audit": {"path": str(AUDIT), "sha256": _sha256(AUDIT)},
        **_registered_training_sources(),
        "training_entrypoint": {
            "path": str(REPO_ROOT / "main.py"),
            "sha256": _sha256(REPO_ROOT / "main.py"),
        },
        "training_engine": {
            "path": str(REPO_ROOT / "engine.py"),
            "sha256": _sha256(REPO_ROOT / "engine.py"),
        },
        "launcher": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "python_runtime": _file_record(python),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": validation["checkpoint_sha256"],
        },
    }
    soak_seal = None
    if purpose == "formal":
        seal_path = Path(getattr(args, "soak_seal", DEFAULT_SOAK_SEAL))
        soak_seal = _validate_soak_seal(seal_path)
        inputs["g0c_soak_seal"] = {
            "path": soak_seal["path"],
            "sha256": soak_seal["sha256"],
        }
    plan = {
        "schema": PLAN_SCHEMA,
        "row_id": "G0c",
        "purpose": purpose,
        "evidence_status": "registered_runnable_no_result",
        "matched_contract": {
            **batch_contract,
            "global_batch": batch_contract["effective_global_batch"],
            "optimizer_updates": int(args.updates),
            "planned_micro_batches_without_amp_skips": planned_micro_batches,
            "train_rows": dataset_rows,
            "train_micro_batches_per_epoch": micro_batches_per_epoch,
            "expected_checkpoint_epoch": 0,
            "expected_checkpoint_iteration": planned_micro_batches,
            "required_amp_skips": 0,
            "seed": int(args.seed),
            "source_mix_weights": [1.0, 1.0, 1.0, 3.0],
            "horizontal_flip": False,
            "warmstart": "historical_gdino_stageb_data_ft_b58",
        },
        "inputs": inputs,
        "source_dependency_tree": _source_dependency_tree(),
        "runtime_evidence_required": purpose in {"formal", "soak"},
        "cuda_visible_devices": str(
            getattr(
                args,
                "cuda_visible_devices",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            )
        ),
        "soak_seal": (
            {
                "schema": SOAK_SEAL_SCHEMA,
                "path": soak_seal["path"],
                "sha256": soak_seal["sha256"],
            }
            if soak_seal is not None
            else None
        ),
        "output_dir": str(output_dir),
        "command": command,
        "evaluation": {
            "tool": str(REPO_ROOT / "tools/eval_text_groundingdino_refcoco_tn.py"),
            "required_ref_splits": "all",
            "required_strict_manifests": ["strict2031", "strict1607"],
            "candidate_count_control": 50,
            "candidate_count_repeats": 32,
            "note": "the 50-query subset is a multiplicity diagnostic, not G0c",
        },
    }
    if purpose == "formal":
        assert soak_seal is not None
        plan["soak_compatibility"] = _validate_formal_soak_compatibility(
            plan, soak_seal["plan"]
        )
    plan["plan_sha256"] = _plan_sha256(plan)
    return plan


def _validate_training_state(payload: Mapping[str, Any]) -> None:
    import torch

    model = payload.get("model")
    if not isinstance(model, Mapping) or len(model) < 100:
        raise ValueError("G0c checkpoint model state is empty or implausibly small")
    required_model_prefixes = ("transformer.", "bert.", "bbox_embed.", "backbone.")
    for prefix in required_model_prefixes:
        matches = [value for key, value in model.items() if str(key).startswith(prefix)]
        if not matches or not any(torch.is_tensor(value) for value in matches):
            raise ValueError(f"G0c checkpoint model lacks tensor prefix {prefix}")
    for key, value in model.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"G0c checkpoint model tensor is non-finite: {key}")

    criterion = payload.get("criterion")
    if not isinstance(criterion, Mapping):
        raise ValueError("G0c checkpoint criterion state must be a mapping")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("G0c checkpoint optimizer state must be a mapping")
    optimizer_state = optimizer.get("state")
    param_groups = optimizer.get("param_groups")
    if not isinstance(optimizer_state, Mapping) or not optimizer_state:
        raise ValueError("G0c checkpoint optimizer has no learned state")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("G0c checkpoint optimizer has no parameter groups")
    if not any(
        isinstance(group, Mapping) and bool(group.get("params"))
        for group in param_groups
    ):
        raise ValueError("G0c checkpoint optimizer parameter groups are all empty")

    scheduler = payload.get("lr_scheduler")
    if not isinstance(scheduler, Mapping):
        raise ValueError("G0c checkpoint scheduler state must be a mapping")
    if not isinstance(scheduler.get("base_lrs"), list) or not scheduler["base_lrs"]:
        raise ValueError("G0c checkpoint scheduler has no base learning rates")
    if not isinstance(scheduler.get("last_epoch"), int):
        raise ValueError("G0c checkpoint scheduler last_epoch is invalid")
    if int(scheduler.get("_step_count", 0)) < 1:
        raise ValueError("G0c checkpoint scheduler step count is invalid")

    scaler = payload.get("scaler")
    if not isinstance(scaler, Mapping) or not scaler:
        raise ValueError("G0c checkpoint AMP scaler state is empty")
    numeric_checks = {
        "scale": lambda value: math.isfinite(value) and value > 0.0,
        "growth_factor": lambda value: value > 1.0,
        "backoff_factor": lambda value: 0.0 < value < 1.0,
        "growth_interval": lambda value: value > 0.0,
        "_growth_tracker": lambda value: value >= 0.0,
    }
    for key, predicate in numeric_checks.items():
        value = scaler.get(key)
        if not isinstance(value, (int, float)) or not predicate(float(value)):
            raise ValueError(f"G0c checkpoint AMP scaler field is invalid: {key}")


def _evidence_runtime(plan: Mapping[str, Any]):
    from tools import run_stageb_paper_ablation_matrices as evidence

    python = Path(str(plan["command"][0])).resolve(strict=True)
    checkpoint = Path(str(plan["inputs"]["checkpoint"]["path"])).resolve(strict=True)
    output_dir = Path(str(plan["output_dir"])).resolve(strict=False)
    return evidence.Runtime(
        python=python,
        stage_a_init=checkpoint,
        scorer_warmstart=checkpoint,
        tn_output_root=output_dir.parent,
        score_output_root=output_dir.parent,
        data_root=Path(
            os.environ.get("DATA_ROOT", "/media/haoyi/T9/data")
        ).resolve(strict=False),
        batch_size=int(plan["matched_contract"]["micro_batch_size_per_rank"]),
        total_train_iters=int(plan["matched_contract"]["optimizer_updates"]),
        iter_checkpoint_interval=int(plan["matched_contract"]["optimizer_updates"]),
        num_workers=0,
        prefetch_factor=1,
        omp_num_threads=1,
        min_nofile=0,
        cuda_visible_devices=str(plan.get("cuda_visible_devices", "0")),
        mp_sharing_strategy="none",
        gradient_diagnostic_interval=1,
    )


def _stream_training(
    command: Sequence[str], *, environment: Mapping[str, str], log_path: Path
) -> int:
    with log_path.open("xb") as raw_log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=dict(environment),
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


def _runtime_evidence_from_output(plan: Mapping[str, Any]) -> Dict[str, Any]:
    from tools import run_stageb_paper_ablation_matrices as evidence

    output_dir = Path(str(plan["output_dir"])).resolve(strict=True)
    paths = {
        "native_info_log": output_dir / "info.txt",
        "train_console_log": output_dir / "train_console.log",
        "gpu_environment": output_dir / "gpu_environment.json",
        "gpu_telemetry": output_dir / "gpu_telemetry.csv",
        "gpu_telemetry_summary": output_dir / "gpu_telemetry_summary.json",
    }
    resolved = {key: path.resolve(strict=True) for key, path in paths.items()}
    gpu_environment = _read_json(
        resolved["gpu_environment"], label="G0c GPU environment"
    )
    gpu_summary = _read_json(
        resolved["gpu_telemetry_summary"], label="G0c GPU telemetry summary"
    )
    evidence._validate_gpu_telemetry_contract(gpu_environment, gpu_summary)
    numerical = evidence._training_numerical_status(
        resolved["native_info_log"], resolved["train_console_log"]
    )
    devices = gpu_summary.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError("G0c runtime evidence requires exactly one GPU")
    return {
        "numerical_status": numerical,
        "gpu_environment": dict(gpu_environment),
        "gpu_telemetry_summary": dict(gpu_summary),
        "runtime_artifacts": {
            key: _file_record(path) for key, path in resolved.items()
        },
    }


def verify_checkpoint(
    plan: Mapping[str, Any], *, write_postflight: bool = True
) -> Dict[str, Any]:
    import numpy as np
    import torch

    _validate_plan_identity(plan)
    if plan.get("row_id") != "G0c":
        raise ValueError("G0c postflight received the wrong paper row")
    purpose = str(plan.get("purpose", ""))
    contract = plan.get("matched_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("G0c plan has no matched contract")
    if purpose in {"formal", "soak"}:
        expected_updates = (
            FORMAL_OPTIMIZER_UPDATES if purpose == "formal" else SOAK_OPTIMIZER_UPDATES
        )
        expected_seeds = FORMAL_SEEDS if purpose == "formal" else (17,)
        expected_root = (
            formal_output_root(int(contract.get("seed", -1)))
            if purpose == "formal"
            else DEFAULT_SOAK_ROOT
        ).resolve(strict=False)
        observed_contract = {
            "micro_batch_size_per_rank": int(
                contract.get("micro_batch_size_per_rank", -1)
            ),
            "gradient_accumulation_steps": int(
                contract.get("gradient_accumulation_steps", -1)
            ),
            "effective_global_batch": int(contract.get("effective_global_batch", -1)),
            "optimizer_updates": int(contract.get("optimizer_updates", -1)),
        }
        expected_contract = {
            "micro_batch_size_per_rank": FORMAL_MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch": REQUIRED_EFFECTIVE_GLOBAL_BATCH,
            "optimizer_updates": expected_updates,
        }
        if observed_contract != expected_contract:
            raise ValueError(f"G0c {purpose} matched contract is not exact")
        if int(contract.get("seed", -1)) not in expected_seeds:
            raise ValueError(f"G0c {purpose} seed is not predeclared")
        if Path(str(plan.get("output_dir", ""))).resolve(strict=False) != expected_root:
            raise ValueError(f"G0c {purpose} output root is not canonical")
    elif purpose != "probe":
        raise ValueError("G0c plan purpose is invalid")
    if purpose == "formal":
        soak = plan.get("soak_seal")
        if not isinstance(soak, Mapping):
            raise ValueError("G0c formal plan has no soak seal binding")
        seal_path = Path(str(soak.get("path", ""))).resolve(strict=True)
        validated = _validate_soak_seal(seal_path)
        if validated["sha256"] != str(soak.get("sha256", "")):
            raise ValueError("G0c formal soak seal SHA-256 drifted")
        compatibility = _validate_formal_soak_compatibility(plan, validated["plan"])
        if plan.get("soak_compatibility") != compatibility:
            raise ValueError("G0c formal soak compatibility receipt drifted")
    verify_data(dataset=DATASET, audit_path=AUDIT)
    if plan.get("source_dependency_tree") != _source_dependency_tree():
        raise ValueError("G0c runtime source dependency tree changed after planning")
    for label, record in plan["inputs"].items():
        path = Path(record["path"]).resolve(strict=True)
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise ValueError(f"G0c registered input changed after planning: {label}")

    output_dir = Path(plan["output_dir"]).resolve()
    checkpoint = output_dir / "checkpoint_iter.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    numpy_core = getattr(np, "_core", np.core)
    safe_globals = [
        numpy_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("G0c checkpoint payload must be a mapping")
    _validate_training_state(payload)

    expected_updates = int(contract["optimizer_updates"])
    expected_micro_batches = int(
        contract["planned_micro_batches_without_amp_skips"]
    )
    if int(payload.get("epoch", -1)) != int(contract["expected_checkpoint_epoch"]):
        raise ValueError("G0c checkpoint epoch differs from the registered plan")
    if payload.get("checkpoint_reason") != "max_train_iters":
        raise ValueError("G0c checkpoint reason must be max_train_iters")
    if bool(payload.get("epoch_finished", True)):
        raise ValueError("G0c fixed-update checkpoint must be mid-epoch")
    if int(payload.get("optimizer_updates", -1)) != expected_updates:
        raise ValueError("G0c checkpoint optimizer-update count mismatch")
    if int(payload.get("iteration", -1)) != expected_micro_batches:
        raise ValueError(
            "G0c consumed micro-batch count differs from the zero-AMP-skip plan"
        )

    checkpoint_args = payload.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise TypeError("G0c checkpoint args must be a mapping")
    expected_args = {
        "batch_size": int(contract["micro_batch_size_per_rank"]),
        "gradient_accumulation_steps": int(
            contract["gradient_accumulation_steps"]
        ),
        "max_train_iters": expected_updates,
        "iter_checkpoint_interval": expected_updates,
        "seed": int(contract["seed"]),
        "amp": True,
        "amp_init_scale": 512.0,
        "amp_max_consecutive_skips": 8,
        "paper_table_a_id": "G0c",
        "patch_only": False,
        "stage_b": False,
        "enable_patch_branch": False,
        "data_aug_hflip_prob": 0.0,
        "skip_eval": True,
        "world_size": 1,
        "rank": 0,
        "local_rank": 0,
        "distributed": False,
    }
    for key, expected in expected_args.items():
        if checkpoint_args.get(key) != expected:
            raise ValueError(
                f"G0c checkpoint args mismatch at {key}: "
                f"{checkpoint_args.get(key)!r} != {expected!r}"
            )
    expected_paths = {
        "config_file": CONFIG,
        "datasets": DATASET,
        "output_dir": output_dir,
        "pretrain_model_path": Path(plan["inputs"]["checkpoint"]["path"]),
    }
    for key, expected in expected_paths.items():
        reported = checkpoint_args.get(key)
        if not reported or Path(str(reported)).expanduser().resolve() != expected.resolve():
            raise ValueError(f"G0c checkpoint path mismatch at {key}")
    if checkpoint_args.get("resume") not in (None, ""):
        raise ValueError("G0c must warm-start model weights, not resume optimizer state")

    result = {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "PASS",
        "row_id": "G0c",
        "purpose": purpose,
        "plan_sha256": str(plan["plan_sha256"]),
        "checkpoint": _file_record(checkpoint),
        "seed": int(contract["seed"]),
        "micro_batch_size_per_rank": int(contract["micro_batch_size_per_rank"]),
        "gradient_accumulation_steps": int(contract["gradient_accumulation_steps"]),
        "optimizer_updates": expected_updates,
        "consumed_micro_batches": expected_micro_batches,
        "amp_skips_inferred": 0,
        "effective_global_batch": int(contract["effective_global_batch"]),
        "source_dependency_tree_sha256": str(plan["source_dependency_tree"]["sha256"]),
        "validated_at_utc": _utc_now(),
    }
    if bool(plan.get("runtime_evidence_required")):
        result.update(_runtime_evidence_from_output(plan))
    postflight = output_dir / "postflight.json"
    if write_postflight:
        postflight.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def build_soak_seal(plan_path: Path) -> Dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    if plan_path != DEFAULT_SOAK_PLAN.resolve(strict=True):
        raise ValueError("G0c soak seal requires the canonical v2 soak plan path")
    plan = _read_json(plan_path, label="G0c soak plan")
    _validate_plan_identity(plan)
    if plan.get("purpose") != "soak":
        raise ValueError("G0c soak seal received a non-soak plan")
    postflight = verify_checkpoint(plan, write_postflight=True)
    if (
        postflight.get("schema") != POSTFLIGHT_SCHEMA
        or postflight.get("status") != "PASS"
    ):
        raise ValueError("G0c soak postflight did not pass")
    devices = postflight.get("gpu_telemetry_summary", {}).get("devices", [])
    if not isinstance(devices, list) or len(devices) != 1:
        raise ValueError("G0c soak telemetry must contain exactly one device")
    headroom = float(devices[0].get("min_free_memory_mib", -1.0))
    if not math.isfinite(headroom) or headroom < MINIMUM_SOAK_HEADROOM_MIB:
        raise ValueError(
            f"G0c soak has {headroom} MiB headroom; "
            f"requires {MINIMUM_SOAK_HEADROOM_MIB} MiB"
        )
    output_dir = Path(str(plan["output_dir"])).resolve(strict=True)
    runtime_artifacts = postflight.get("runtime_artifacts")
    if not isinstance(runtime_artifacts, Mapping):
        raise ValueError("G0c soak postflight has no runtime artifacts")
    artifacts = {
        "plan": _file_record(plan_path),
        "postflight": _file_record(output_dir / "postflight.json"),
        "checkpoint": _file_record(output_dir / "checkpoint_iter.pth"),
        **{key: dict(value) for key, value in runtime_artifacts.items()},
    }
    expected_artifacts = {
        "plan",
        "postflight",
        "checkpoint",
        "native_info_log",
        "train_console_log",
        "gpu_environment",
        "gpu_telemetry",
        "gpu_telemetry_summary",
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError("G0c soak evidence artifact set is incomplete")
    return {
        "schema": SOAK_SEAL_SCHEMA,
        "status": "sealed",
        "sealed_at_utc": _utc_now(),
        "contract": {
            "purpose": "soak",
            "seed": 17,
            "micro_batch_size_per_rank": FORMAL_MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
            "effective_global_batch": REQUIRED_EFFECTIVE_GLOBAL_BATCH,
            "optimizer_updates": SOAK_OPTIMIZER_UPDATES,
            "minimum_headroom_mib": MINIMUM_SOAK_HEADROOM_MIB,
            "required_amp_skips": 0,
            "finite_losses_required": True,
        },
        "observed_minimum_free_memory_mib": headroom,
        "gpu_identity": {
            key: devices[0][key]
            for key in ("uuid", "name", "driver_version", "total_memory_mib")
        },
        "plan_sha256": str(plan["plan_sha256"]),
        "artifacts": artifacts,
    }


def _write_json_fresh(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refuse to overwrite sealed artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "run", "verify", "seal-soak"])
    parser.add_argument(
        "--purpose", choices=["formal", "soak", "probe"], default="formal"
    )
    parser.add_argument(
        "--python",
        default=os.environ.get(
            "PIVOT_PYTHON", "/home/haoyi/miniconda/envs/gdino5090/bin/python"
        ),
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="physical per-rank micro-batch; 10 is below the historical GDINO batch-19 ceiling",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=4
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=REQUIRED_EFFECTIVE_GLOBAL_BATCH,
    )
    parser.add_argument("--updates", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir")
    parser.add_argument("--plan-json")
    parser.add_argument("--soak-seal", default=str(DEFAULT_SOAK_SEAL))
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    args = parser.parse_args()
    if args.updates is None:
        args.updates = (
            FORMAL_OPTIMIZER_UPDATES
            if args.purpose == "formal"
            else SOAK_OPTIMIZER_UPDATES
            if args.purpose == "soak"
            else 2
        )
    if args.output_dir is None:
        args.output_dir = str(
            formal_output_root(args.seed)
            if args.purpose == "formal"
            else DEFAULT_SOAK_ROOT
            if args.purpose == "soak"
            else REPO_ROOT
            / f"outputs/paper_cvpr_v1/memory_probes/table_a_g0c_probe_seed{args.seed}"
        )
    if args.plan_json is None:
        args.plan_json = str(
            formal_plan_path(args.seed)
            if args.purpose == "formal"
            else DEFAULT_SOAK_PLAN
            if args.purpose == "soak"
            else REPO_ROOT
            / f"outputs/paper_cvpr_v1/plans/table_a_g0c_probe_seed{args.seed}.json"
        )
    plan_path = Path(args.plan_json).expanduser().resolve()
    if args.purpose == "formal" and plan_path != formal_plan_path(args.seed).resolve(
        strict=False
    ):
        raise ValueError("G0c formal plan path must be canonical")
    if args.purpose == "soak" and plan_path != DEFAULT_SOAK_PLAN.resolve(strict=False):
        raise ValueError("G0c soak plan path must be canonical")
    if args.mode == "seal-soak":
        if args.purpose != "soak":
            raise ValueError("seal-soak requires --purpose soak")
        seal = build_soak_seal(plan_path)
        seal_path = Path(args.soak_seal).expanduser().resolve(strict=False)
        if seal_path != DEFAULT_SOAK_SEAL.resolve(strict=False):
            raise ValueError("G0c soak seal path must be canonical")
        _write_json_fresh(seal_path, seal)
        print(json.dumps({"status": "sealed", "soak_seal": str(seal_path)}))
        return
    if args.mode == "verify":
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print(json.dumps(verify_checkpoint(plan), sort_keys=True))
        return

    plan = build_plan(args)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if plan_path.exists() and plan_path.read_text(encoding="utf-8") != rendered:
        raise FileExistsError(f"refuse to replace immutable G0c plan: {plan_path}")
    if not plan_path.exists():
        plan_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({"plan": str(plan_path), "command": plan["command"]}))
    if args.mode == "run":
        output_dir = Path(plan["output_dir"])
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"G0c run output must be fresh, found existing files in {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        from tools import run_stageb_paper_ablation_matrices as evidence

        runtime = _evidence_runtime(plan)
        gpu_environment = evidence._capture_gpu_environment(runtime, output_dir)
        sampler = evidence._GpuTelemetrySampler(runtime, output_dir)
        try:
            environment = _single_process_environment()
            environment["CUDA_VISIBLE_DEVICES"] = str(plan["cuda_visible_devices"])
            returncode = _stream_training(
                plan["command"],
                environment=environment,
                log_path=output_dir / "train_console.log",
            )
        finally:
            gpu_summary = sampler.stop()
        evidence._validate_gpu_telemetry_contract(gpu_environment, gpu_summary)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, plan["command"])
    if args.mode == "run":
        print(json.dumps(verify_checkpoint(plan), sort_keys=True))


if __name__ == "__main__":
    main()
