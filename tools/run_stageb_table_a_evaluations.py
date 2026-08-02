#!/usr/bin/env python3
"""Plan, run, and verify the sealed Table-A evaluation protocols.

``candidate`` evaluates G1-G5, fixed-canonical TN edit sensitivity, and the
synchronized noun-category intervention in one model load. ``g0c`` evaluates
the continued pure-GDINO checkpoint on REF8 and both locked TN manifests.
Neither mode accepts a bare, unverified paper checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import eval_stageb_role_causal as role_eval  # noqa: E402
from tools import run_stageb_table_a_controls as g0c_controls  # noqa: E402
from tools import run_stageb_paper_evaluations as paper_eval  # noqa: E402
from tools.stageb_screen_calibration import (  # noqa: E402
    DEFAULT_AUDIT as SCREEN_CALIBRATION_AUDIT,
    DEFAULT_SOURCE as SCREEN_CALIBRATION_SOURCE,
)
from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT as TABLE_A_REF_SPLIT_CONTRACT,
)
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.table_a_evaluation_launch/v1"
POSTFLIGHT_SCHEMA = "pivot.stageb.table_a_evaluation_postflight/v1"
INPUT_REHASH_SCHEMA = "pivot.stageb.table_a_evaluation_input_rehash/v1"
EVAL_SEED = 42
ROLE_EVALUATOR = REPO_ROOT / "tools/eval_stageb_role_causal.py"
TEXT_EVALUATOR = REPO_ROOT / "tools/eval_text_groundingdino_refcoco_tn.py"
CATEGORY_ROOT = (
    REPO_ROOT / "data/ablations/stageb_table_a_category_intervention_20260717"
)
CATEGORY_JSONL = CATEGORY_ROOT / "category_intervention_pairs.jsonl"
CATEGORY_SUPPORT = CATEGORY_ROOT / "category_intervention_support.tsv"
CATEGORY_AUDIT = CATEGORY_ROOT / "audit.json"
REQUIRED_EDIT_TAXONOMIES = ("color", "size", "action", "spatial", "relation")
ROLE_CODE_PATHS = (
    ROLE_EVALUATOR,
    REPO_ROOT / "tools/eval_refcoco_stageb.py",
    REPO_ROOT / "tools/eval_stageb_tn_val.py",
    REPO_ROOT / "tools/build_stageb_table_a_category_interventions.py",
    Path(__file__).resolve(),
)
VALIDATION_PROFILE = "validation"
FINAL_PROFILE = "final"
PROFILES = (VALIDATION_PROFILE, FINAL_PROFILE)
FORMAL_SEEDS = (17, 42, 73)
FORMAL_EVAL_BATCH_SIZE = 16
FORMAL_EVAL_NUM_WORKERS = 8
FORMAL_EVAL_DEVICE = "cuda:0"
VALIDATION_REF_SPLITS = tuple(paper_eval.SCREEN_REF_SPLITS)
FINAL_GATE_SCHEMA = "pivot.stageb.table_a_final_evaluation_gate/v2"
FINAL_GATE_PATH = (
    REPO_ROOT / "outputs/paper_cvpr_v1/gates/table_a_final_evaluation_gate.json"
)
FINAL_CONSUMPTION_SCHEMA = "pivot.stageb.table_a_final_consumption/v1"
FINAL_CONSUMPTION_ROOT = (
    REPO_ROOT / "outputs/paper_cvpr_v1/gates/table_a_final_consumptions"
)
INSTANCE_SCHEMA = "pivot.stageb.table_a_evaluation_instance/v1"
RELEASE_PREREQUISITE_NAMES = (
    "headline_selection_receipt",
    "paper_ablation_completion_receipt",
)
VOLATILE_EVIDENCE_KEYS = {
    "claimed_at_utc",
    "created_at_utc",
    "finished_at_utc",
    "observed_at_utc",
    "started_at_utc",
    "updated_at_utc",
    "validated_at_utc",
    "verified_at_utc",
}
G0C_SOURCE_KIND = "table_a_g0c_formal_training_run"
G0C_SOURCE_FAMILY = "table_a_g0c"
G0C_TRAINING_PROVENANCE_SCHEMA = (
    "pivot.stageb.table_a_g0c_training_provenance/v1"
)
G0C_TN_AGGREGATE_METRICS = (
    "fpr95tpr",
    "fpr90tpr",
    "pair_win_rate",
    "pair_tie_rate",
    "pos_score_mean",
    "tn_score_mean",
    "score_gap_mean",
    "threshold_at_95tpr",
    "actual_tpr_at_95tpr",
)
LOCKED_CANDIDATE_QUEUES: Mapping[int, Mapping[str, str]] = {
    17: {
        "path": str(
            REPO_ROOT
            / "outputs/paper_cvpr_v1/queues/"
            "table_c_screen_l0_l4_seed17_b40_u1000_frozen_v2"
        ),
        "queue_id": "3e5a961a-f2da-45ba-8e44-94740f4baee9",
        "plan_sha256": "63619de10c9e41d2ecc5177242b4b3bbf175d57c3c9cdcd7013b1185a53e6cde",
    },
    42: {
        "path": str(
            REPO_ROOT
            / "outputs/paper_cvpr_v1/queues/"
            "table_c_remaining_28_b40_u1000_frozen_v2"
        ),
        "queue_id": "ffcc3e46-ca1d-45d0-9fbd-22e5db14ac9f",
        "plan_sha256": "b4b8ef280fcbd67dbf82fc59d6c90f63c9c3573976b8950c06f1e84dbb31c2cc",
    },
    73: {
        "path": str(
            REPO_ROOT
            / "outputs/paper_cvpr_v1/queues/"
            "table_c_remaining_28_b40_u1000_frozen_v2"
        ),
        "queue_id": "ffcc3e46-ca1d-45d0-9fbd-22e5db14ac9f",
        "plan_sha256": "b4b8ef280fcbd67dbf82fc59d6c90f63c9c3573976b8950c06f1e84dbb31c2cc",
    },
}


class TableAEvaluationError(RuntimeError):
    """Raised when a Table-A evidence contract cannot be proven."""


@dataclass(frozen=True)
class Runtime:
    python: Path
    data_root: Path
    device: str
    batch_size: int
    num_workers: int
    amp: bool


@dataclass(frozen=True)
class G0cEvaluationSource(paper_eval.EvaluationSource):
    """A completed continued-GDINO run, never a historical baseline alias."""

    source_family: str = G0C_SOURCE_FAMILY
    training_plan: Path | None = None
    training_plan_contract_sha256: str | None = None
    source_dependency_tree_sha256: str | None = None
    source_provenance_dependencies: tuple[Path, ...] = ()


class HashCache:
    def __init__(self) -> None:
        self._values: dict[tuple[str, int, int], str] = {}

    def digest(self, path: Path) -> str:
        path = path.resolve(strict=True)
        stat = path.stat()
        key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        value = self._values.get(key)
        if value is not None:
            return value
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._values[key] = value
        return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TableAEvaluationError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TableAEvaluationError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a fully written gate without replacing another one."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - required Linux runtime
        raise TableAEvaluationError(
            "atomic Table-A gate publication requires Linux renameat2"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            f"Table-A final gate appeared concurrently: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: Path, cache: HashCache, *roles: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise TableAEvaluationError(f"input is not a regular file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": cache.digest(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "roles": sorted(set(roles)),
    }


def _verify_file_record(
    value: Any,
    *,
    label: str,
    roles: Sequence[str],
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TableAEvaluationError(f"{label} is not a file record")
    try:
        path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    except OSError as exc:
        raise TableAEvaluationError(f"{label} is unavailable: {exc}") from exc
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise TableAEvaluationError(f"{label} path is not canonical")
    observed = _file_record(path, HashCache(), *roles)
    if dict(value) != observed:
        raise TableAEvaluationError(f"{label} changed after gate sealing")
    return observed


def _strip_volatile_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_volatile_evidence(item)
            for key, item in value.items()
            if key not in VOLATILE_EVIDENCE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile_evidence(item) for item in value]
    return value


def _release_prerequisite_records(
    *, replay_selection: bool = False
) -> dict[str, dict[str, Any]]:
    from tools import stageb_headline_release_contract as headline_release

    try:
        selection = headline_release.validate_selection_receipt(
            headline_release.SELECTION_RECEIPT_PATH,
            replay_validation=replay_selection,
        )
        completion = headline_release.validate_paper_ablation_completion_receipt(
            headline_release.PAPER_ABLATION_COMPLETION_RECEIPT_PATH
        )
        if selection.get("status") != "eligible":
            raise TableAEvaluationError(
                "headline selection is ineligible; Table-A final gate remains closed"
            )
        if (
            completion.get("status") != "completed"
            or completion.get("completed_before_final_gate") is not True
            or completion.get("all_training_validation_diagnostics_completed")
            is not True
        ):
            raise TableAEvaluationError(
                "paper ablation completion receipt is not final-gate eligible"
            )
        records = {
            "headline_selection_receipt": headline_release.file_record(
                headline_release.SELECTION_RECEIPT_PATH
            ),
            "paper_ablation_completion_receipt": headline_release.file_record(
                headline_release.PAPER_ABLATION_COMPLETION_RECEIPT_PATH
            ),
        }
    except TableAEvaluationError:
        raise
    except (headline_release.HeadlineReleaseError, OSError) as exc:
        raise TableAEvaluationError(
            f"Table-A final release prerequisite replay failed: {exc}"
        ) from exc
    if set(records) != set(RELEASE_PREREQUISITE_NAMES):
        raise TableAEvaluationError(
            "Table-A final release prerequisite set is not exact"
        )
    return records


def _merge_record(
    records: dict[str, dict[str, Any]],
    path: Path,
    cache: HashCache,
    *roles: str,
) -> None:
    record = _file_record(path, cache, *roles)
    key = record["path"]
    existing = records.get(key)
    if existing is None:
        records[key] = record
        return
    if existing["sha256"] != record["sha256"]:
        raise TableAEvaluationError(f"input hash changed during planning: {key}")
    existing["roles"] = sorted(set(existing["roles"]).union(roles))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


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


def _formal_runtime_contract(value: Runtime | Mapping[str, Any]) -> dict[str, Any]:
    raw = asdict(value) if isinstance(value, Runtime) else dict(value)
    try:
        contract = {
            "python": str(Path(str(raw["python"])).expanduser().resolve(strict=True)),
            "data_root": str(
                Path(str(raw["data_root"])).expanduser().resolve(strict=True)
            ),
            "device": str(raw["device"]),
            "batch_size": int(raw["batch_size"]),
            "num_workers": int(raw["num_workers"]),
            "amp": bool(raw["amp"]),
            "eval_seed": int(raw.get("eval_seed", EVAL_SEED)),
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise TableAEvaluationError("Table-A runtime contract is incomplete") from exc
    expected = {
        "device": FORMAL_EVAL_DEVICE,
        "batch_size": FORMAL_EVAL_BATCH_SIZE,
        "num_workers": FORMAL_EVAL_NUM_WORKERS,
        "amp": True,
        "eval_seed": EVAL_SEED,
    }
    for key, expected_value in expected.items():
        if contract[key] != expected_value:
            raise TableAEvaluationError(
                f"formal Table-A runtime drifted at {key}: "
                f"{contract[key]!r} != {expected_value!r}"
            )
    return contract


def _require_current_python_runtime(python: Path, *, label: str) -> Path:
    try:
        selected = python.expanduser().resolve(strict=True)
        current = Path(sys.executable).resolve(strict=True)
    except OSError as exc:
        raise TableAEvaluationError(f"{label} Python runtime is unavailable: {exc}") from exc
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise TableAEvaluationError(f"{label} Python runtime is not executable: {selected}")
    if selected != current:
        raise TableAEvaluationError(
            f"{label} must run under its selected Python runtime so native "
            f"dependencies are sealed correctly: caller={current}, selected={selected}"
        )
    return selected


def canonical_output_dir(kind: str, profile: str, seed: int) -> Path:
    if kind not in {"candidate", "g0c"} or profile not in PROFILES:
        raise TableAEvaluationError("invalid Table-A canonical output identity")
    row = "L4" if kind == "candidate" else "G0c"
    return (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/table_a/evaluations"
        / profile
        / kind
        / row
        / f"seed{int(seed)}"
    )


def _instance_payload(
    *,
    kind: str,
    profile: str,
    seed: int,
    output_dir: Path,
    source: Any,
    runtime: Runtime | Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": INSTANCE_SCHEMA,
        "kind": kind,
        "profile": profile,
        "seed": int(seed),
        "evaluation_seed": EVAL_SEED,
        "evaluation_id": str(source.evaluation_id),
        "training_run_id": source.training_run_id,
        "checkpoint_sha256": str(source.checkpoint_sha256),
        "output_dir": str(output_dir),
        "training_queue_id": source.training_queue_id,
        "training_queue_plan_sha256": source.training_queue_plan_sha256,
        "runtime_contract": _formal_runtime_contract(runtime),
    }
    payload["instance_id"] = (
        f"table_a:{profile}:{kind}:"
        f"{source.training_run_id or source.evaluation_id}"
    )
    payload["instance_sha256"] = _canonical_json_sha256(payload)
    return payload


def _validate_instance(plan: Mapping[str, Any]) -> None:
    instance = plan.get("instance")
    if not isinstance(instance, Mapping) or instance.get("schema") != INSTANCE_SCHEMA:
        raise TableAEvaluationError("Table-A evaluation instance contract is missing")
    payload = copy.deepcopy(dict(instance))
    expected = str(payload.pop("instance_sha256", ""))
    if len(expected) != 64 or expected != _canonical_json_sha256(payload):
        raise TableAEvaluationError("Table-A evaluation instance SHA-256 mismatch")
    source = plan.get("source")
    runtime = plan.get("runtime")
    if not isinstance(source, Mapping) or not isinstance(runtime, Mapping):
        raise TableAEvaluationError("Table-A plan source/runtime contract is missing")
    expected_fields = {
        "kind": plan.get("kind"),
        "profile": plan.get("profile"),
        "evaluation_seed": EVAL_SEED,
        "evaluation_id": plan.get("evaluation_id"),
        "training_run_id": source.get("training_run_id"),
        "checkpoint_sha256": source.get("checkpoint_sha256"),
        "output_dir": plan.get("output_dir"),
        "training_queue_id": source.get("training_queue_id"),
        "training_queue_plan_sha256": source.get("training_queue_plan_sha256"),
        "runtime_contract": _formal_runtime_contract(runtime),
    }
    for key, value in expected_fields.items():
        if instance.get(key) != value:
            raise TableAEvaluationError(f"Table-A instance drifted at {key}")
    seed = int(instance.get("seed", -1))
    if seed not in FORMAL_SEEDS:
        raise TableAEvaluationError("Table-A instance seed is not predeclared")
    expected_root = canonical_output_dir(
        str(plan.get("kind")), str(plan.get("profile")), seed
    ).resolve(strict=False)
    if Path(str(plan.get("output_dir", ""))).resolve(strict=False) != expected_root:
        raise TableAEvaluationError("Table-A instance output root is not canonical")


def _validate_command_surface(plan: Mapping[str, Any]) -> None:
    kind = str(plan.get("kind", ""))
    profile = str(plan.get("profile", ""))
    commands = plan.get("commands")
    expected_count = 2 if kind == "g0c" and profile == FINAL_PROFILE else 1
    if not isinstance(commands, list) or len(commands) != expected_count:
        raise TableAEvaluationError("Table-A command phase count drifted")
    flattened: list[str] = []
    for spec in commands:
        if not isinstance(spec, Mapping) or not isinstance(spec.get("command"), list):
            raise TableAEvaluationError("Table-A command specification is invalid")
        command = [str(value) for value in spec["command"]]
        if spec.get("command_shell") != shlex.join(command):
            raise TableAEvaluationError("Table-A command shell receipt drifted")
        flattened.extend(command)
    if "--candidate_count_control" in flattened or "--candidate_count_repeats" in flattened:
        raise TableAEvaluationError("diagnostic candidate-count flags are forbidden")
    if kind == "candidate":
        required = {"--formal_table_a", "--true_role_swap", "--evaluation_profile"}
        if not required.issubset(flattened):
            raise TableAEvaluationError("candidate command lacks formal G1-G5 flags")
        command = [str(value) for value in commands[0]["command"]]

        def option(flag: str) -> str:
            if command.count(flag) != 1:
                raise TableAEvaluationError(
                    f"candidate command requires exactly one {flag}"
                )
            index = command.index(flag)
            if index + 1 >= len(command):
                raise TableAEvaluationError(f"candidate command has no value for {flag}")
            return command[index + 1]

        expected_options = {
            "--config": str(Path(str(plan["source"]["config"])).resolve(strict=True)),
            "--checkpoint": str(
                Path(str(plan["source"]["checkpoint"])).resolve(strict=True)
            ),
            "--output_dir": str(
                Path(str(plan["output_dir"])).resolve(strict=False) / "role_causal"
            ),
            "--tn_jsonl": str(
                Path(str(plan["tn_manifest"]["path"])).resolve(strict=True)
            ),
            "--category_jsonl": str(CATEGORY_JSONL.resolve(strict=True)),
            "--category_support_tsv": str(CATEGORY_SUPPORT.resolve(strict=True)),
            "--category_audit": str(CATEGORY_AUDIT.resolve(strict=True)),
            "--evaluation_profile": profile,
        }
        for flag, expected in expected_options.items():
            if option(flag) != expected:
                raise TableAEvaluationError(f"candidate command drifted at {flag}")
    elif kind == "g0c":
        runtime_contract = _formal_runtime_contract(plan.get("runtime", {}))
        runtime = Runtime(
            python=Path(runtime_contract["python"]),
            data_root=Path(runtime_contract["data_root"]),
            device=str(runtime_contract["device"]),
            batch_size=int(runtime_contract["batch_size"]),
            num_workers=int(runtime_contract["num_workers"]),
            amp=bool(runtime_contract["amp"]),
        )
        source = plan.get("source")
        contract = plan.get("contract")
        if not isinstance(source, Mapping) or not isinstance(contract, Mapping):
            raise TableAEvaluationError("G0c command source/contract is missing")
        expected = _g0c_command_specs(
            runtime=runtime,
            source=SimpleNamespace(
                config=Path(str(source.get("config", ""))).resolve(strict=True),
                checkpoint=Path(str(source.get("checkpoint", ""))).resolve(
                    strict=True
                ),
            ),
            output_dir=Path(str(plan.get("output_dir", ""))).resolve(strict=False),
            profile=profile,
            ref_splits=tuple(contract.get("ref_splits", ())),
            tn_primary=plan.get("tn_manifest", {}),
            tn_inputs=plan.get("tn_inputs", {}),
        )
        if commands != expected:
            raise TableAEvaluationError(
                "G0c command surface differs from the formal source/runtime/profile"
            )
    else:
        raise TableAEvaluationError(f"unknown Table-A command kind: {kind!r}")
    if profile == VALIDATION_PROFILE:
        forbidden = {
            "refcoco_testA",
            "refcoco_testB",
            "refcocop_testA",
            "refcocop_testB",
            "refcocog_test",
            str(Path(paper_eval.STRICT_SPECS["strict2031"]["path"]).resolve()),
            str(Path(paper_eval.STRICT_SPECS["strict1607"]["path"]).resolve()),
        }
        if forbidden.intersection(flattened):
            raise TableAEvaluationError("validation command accesses a final surface")
        if "--screen_calibration_manifest" not in flattened:
            raise TableAEvaluationError("validation command lacks calibration binding")
    elif "--screen_calibration_manifest" in flattened:
        raise TableAEvaluationError("final command contains validation calibration mode")


def _validate_final_gate(path: Path, instance: Mapping[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if path != FINAL_GATE_PATH.resolve(strict=True):
        raise TableAEvaluationError("Table-A final gate path is not canonical")
    gate = dict(_read_json(path, label="Table-A final evaluation gate"))
    if (
        gate.get("schema") != FINAL_GATE_SCHEMA
        or gate.get("status") != "sealed"
        or gate.get("selection_frozen") is not True
        or gate.get("all_paper_ablations_completed") is not True
        or gate.get("created_before_first_final_evaluation") is not True
        or gate.get("selection_rule")
        != (
            "promote_all_predeclared_candidate_and_g0c_seeds_from_completed_"
            "validation_without_test_metric_selection"
        )
    ):
        raise TableAEvaluationError("Table-A final evaluation gate is not sealed")
    payload = copy.deepcopy(gate)
    expected_sha = str(payload.pop("gate_sha256", ""))
    if len(expected_sha) != 64 or expected_sha != _canonical_json_sha256(payload):
        raise TableAEvaluationError("Table-A final gate canonical SHA-256 mismatch")
    prerequisites = gate.get("release_prerequisites")
    if (
        not isinstance(prerequisites, Mapping)
        or set(prerequisites) != set(RELEASE_PREREQUISITE_NAMES)
        or dict(prerequisites) != _release_prerequisite_records(
            replay_selection=False
        )
    ):
        raise TableAEvaluationError(
            "Table-A final gate release prerequisites changed after sealing"
        )
    provenance = gate.get("validation_provenance")
    expected_pairs = [
        (kind, seed) for kind in ("candidate", "g0c") for seed in FORMAL_SEEDS
    ]
    if not isinstance(provenance, list) or len(provenance) != len(expected_pairs):
        raise TableAEvaluationError(
            "Table-A final gate validation provenance set is not exact"
        )
    reconstructed_instances: list[dict[str, Any]] = []
    for index, ((kind, seed), evidence) in enumerate(
        zip(expected_pairs, provenance)
    ):
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("kind") != kind
            or int(evidence.get("seed", -1)) != seed
        ):
            raise TableAEvaluationError(
                f"Table-A final gate validation provenance {index} drifted"
            )
        root = canonical_output_dir(kind, VALIDATION_PROFILE, seed).resolve(
            strict=True
        )
        launch_path = (root / "launch_manifest.json").resolve(strict=True)
        postflight_path = (root / "postflight.json").resolve(strict=True)
        _verify_file_record(
            evidence.get("validation_launch"),
            label=f"Table-A {kind}/seed{seed} validation launch",
            roles=("table_a_validation_launch",),
            expected_path=launch_path,
        )
        _verify_file_record(
            evidence.get("validation_postflight"),
            label=f"Table-A {kind}/seed{seed} validation postflight",
            roles=("table_a_validation_postflight",),
            expected_path=postflight_path,
        )
        launch = _read_json(
            launch_path, label=f"Table-A {kind}/seed{seed} validation launch"
        )
        if (
            launch.get("status") != "completed"
            or launch.get("kind") != kind
            or launch.get("profile") != VALIDATION_PROFILE
            or int(launch.get("instance", {}).get("seed", -1)) != seed
            or launch.get("instance", {}).get("instance_id")
            != evidence.get("validation_instance_id")
            or launch.get("instance", {}).get("instance_sha256")
            != evidence.get("validation_instance_sha256")
        ):
            raise TableAEvaluationError(
                f"Table-A {kind}/seed{seed} validation identity drifted"
            )
        source = launch.get("source")
        runtime = launch.get("runtime")
        if (
            not isinstance(source, Mapping)
            or not isinstance(runtime, Mapping)
            or not isinstance(launch.get("evaluation_id"), str)
        ):
            raise TableAEvaluationError(
                f"Table-A {kind}/seed{seed} validation source/runtime is missing"
            )
        reconstructed_instances.append(
            _instance_payload(
                kind=kind,
                profile=FINAL_PROFILE,
                seed=seed,
                output_dir=canonical_output_dir(kind, FINAL_PROFILE, seed).resolve(
                    strict=False
                ),
                source=SimpleNamespace(
                    evaluation_id=launch["evaluation_id"],
                    training_run_id=source.get("training_run_id"),
                    checkpoint_sha256=source.get("checkpoint_sha256"),
                    training_queue_id=source.get("training_queue_id"),
                    training_queue_plan_sha256=source.get(
                        "training_queue_plan_sha256"
                    ),
                ),
                runtime=runtime,
            )
        )
    instances = gate.get("instances")
    if (
        not isinstance(instances, list)
        or len(instances) != len(expected_pairs)
        or [
            (value.get("kind"), int(value.get("seed", -1)))
            for value in instances
            if isinstance(value, Mapping)
        ]
        != expected_pairs
    ):
        raise TableAEvaluationError(
            "Table-A final gate instance order/cardinality is not exact"
        )
    instance_digests: set[str] = set()
    for index, value in enumerate(instances):
        assert isinstance(value, Mapping)
        payload = copy.deepcopy(dict(value))
        digest = str(payload.pop("instance_sha256", ""))
        if (
            value.get("schema") != INSTANCE_SCHEMA
            or value.get("profile") != FINAL_PROFILE
            or len(digest) != 64
            or digest != _canonical_json_sha256(payload)
            or value.get("runtime_contract") != gate.get("runtime_contract")
        ):
            raise TableAEvaluationError(
                f"Table-A final gate instance {index} self binding failed"
            )
        instance_digests.add(digest)
    if len(instance_digests) != len(expected_pairs):
        raise TableAEvaluationError("Table-A final gate instances are not unique")
    if [dict(value) for value in instances] != reconstructed_instances:
        raise TableAEvaluationError(
            "Table-A final gate instances differ from validation reconstruction"
        )
    matches = (
        [
            value
            for value in instances
            if isinstance(value, Mapping)
            and value.get("instance_id") == instance.get("instance_id")
            and value.get("instance_sha256") == instance.get("instance_sha256")
        ]
        if isinstance(instances, list)
        else []
    )
    if len(matches) != 1:
        raise TableAEvaluationError(
            "Table-A final gate does not uniquely authorize this immutable instance"
        )
    if gate.get("runtime_contract") != instance.get("runtime_contract"):
        raise TableAEvaluationError("Table-A final gate runtime contract drifted")
    if dict(matches[0]) != dict(instance):
        raise TableAEvaluationError(
            "Table-A final gate authorized instance payload drifted"
        )
    return {"path": str(path), "sha256": HashCache().digest(path), "payload": gate}


def _final_consumption_path(instance: Mapping[str, Any]) -> Path:
    sha256 = str(instance.get("instance_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise TableAEvaluationError("final Table-A instance has no valid SHA-256")
    return FINAL_CONSUMPTION_ROOT / f"{sha256}.json"


def _consume_final_gate(plan: Mapping[str, Any]) -> dict[str, Any]:
    instance = plan.get("instance")
    gate = plan.get("final_gate")
    if not isinstance(instance, Mapping) or not isinstance(gate, Mapping):
        raise TableAEvaluationError("final Table-A consumption lacks instance/gate")
    validated = _validate_final_gate(Path(str(gate.get("path", ""))), instance)
    if validated["sha256"] != gate.get("sha256"):
        raise TableAEvaluationError("final Table-A gate changed before consumption")
    path = _final_consumption_path(instance).resolve(strict=False)
    payload = {
        "schema": FINAL_CONSUMPTION_SCHEMA,
        "status": "claimed",
        "instance_id": instance.get("instance_id"),
        "instance_sha256": instance.get("instance_sha256"),
        "gate_path": validated["path"],
        "gate_sha256": validated["sha256"],
        "output_dir": plan.get("output_dir"),
        "claimed_at_utc": _utc_now(),
    }
    payload["consumption_sha256"] = _canonical_json_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise TableAEvaluationError(
            "final Table-A instance was already consumed; rerun is forbidden"
        ) from exc
    return {"path": str(path), "sha256": HashCache().digest(path)}


def _validate_final_consumption(plan: Mapping[str, Any]) -> dict[str, Any]:
    instance = plan.get("instance")
    record = plan.get("final_consumption")
    gate = plan.get("final_gate")
    if not all(isinstance(value, Mapping) for value in (instance, record, gate)):
        raise TableAEvaluationError("final Table-A consumption receipt is missing")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    if path != _final_consumption_path(instance).resolve(strict=True):
        raise TableAEvaluationError("final Table-A consumption path is not canonical")
    payload = dict(_read_json(path, label="Table-A final consumption receipt"))
    expected_sha = str(payload.pop("consumption_sha256", ""))
    if expected_sha != _canonical_json_sha256(payload):
        raise TableAEvaluationError("final Table-A consumption self SHA-256 mismatch")
    if (
        payload.get("schema") != FINAL_CONSUMPTION_SCHEMA
        or payload.get("status") != "claimed"
        or payload.get("instance_id") != instance.get("instance_id")
        or payload.get("instance_sha256") != instance.get("instance_sha256")
        or payload.get("gate_sha256") != gate.get("sha256")
        or payload.get("output_dir") != plan.get("output_dir")
        or HashCache().digest(path) != record.get("sha256")
    ):
        raise TableAEvaluationError("final Table-A consumption receipt drifted")
    return {"path": str(path), "sha256": str(record["sha256"])}


def seal_final_gate(path: Path = FINAL_GATE_PATH) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=False)
    if path != FINAL_GATE_PATH.resolve(strict=False):
        raise TableAEvaluationError("Table-A final gate output path is not canonical")
    if path.exists():
        raise FileExistsError(f"Table-A final gate already exists: {path}")
    release_prerequisites = _release_prerequisite_records(
        replay_selection=True
    )
    instances: list[dict[str, Any]] = []
    validation_provenance: list[dict[str, Any]] = []
    runtime_contracts: list[dict[str, Any]] = []
    for kind in ("candidate", "g0c"):
        for seed in FORMAL_SEEDS:
            validation_root = canonical_output_dir(
                kind, VALIDATION_PROFILE, seed
            ).resolve(strict=True)
            final_root = canonical_output_dir(kind, FINAL_PROFILE, seed).resolve(
                strict=False
            )
            if final_root.exists():
                raise TableAEvaluationError(
                    f"final evaluation already started before gate sealing: {final_root}"
                )
            launch = _load_launch(validation_root)
            if (
                launch.get("status") != "completed"
                or launch.get("kind") != kind
                or launch.get("profile") != VALIDATION_PROFILE
                or int(launch.get("instance", {}).get("seed", -1)) != seed
            ):
                raise TableAEvaluationError(
                    f"validation instance is incomplete: {kind}/seed{seed}"
                )
            replay = postflight(launch)
            if replay.get("status") != "passed":
                raise TableAEvaluationError(
                    f"validation postflight replay failed: {kind}/seed{seed}"
                )
            persisted_path = (validation_root / "postflight.json").resolve(
                strict=True
            )
            persisted = _read_json(
                persisted_path, label="Table-A validation postflight"
            )
            if (
                persisted.get("status") != "passed"
                or _strip_volatile_evidence(persisted)
                != _strip_volatile_evidence(replay)
            ):
                raise TableAEvaluationError(
                    "validation postflight differs from fresh replay: "
                    f"{kind}/seed{seed}"
                )
            source = launch.get("source")
            if not isinstance(source, Mapping):
                raise TableAEvaluationError("validation launch source is missing")
            namespace = SimpleNamespace(
                evaluation_id=launch["evaluation_id"],
                training_run_id=source.get("training_run_id"),
                checkpoint_sha256=source.get("checkpoint_sha256"),
                training_queue_id=source.get("training_queue_id"),
                training_queue_plan_sha256=source.get(
                    "training_queue_plan_sha256"
                ),
            )
            final_instance = _instance_payload(
                kind=kind,
                profile=FINAL_PROFILE,
                seed=seed,
                output_dir=final_root,
                source=namespace,
                runtime=launch["runtime"],
            )
            if _final_consumption_path(final_instance).exists():
                raise TableAEvaluationError(
                    "final evaluation consumption predates gate sealing"
                )
            instances.append(final_instance)
            runtime_contracts.append(dict(final_instance["runtime_contract"]))
            validation_provenance.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "validation_instance_id": launch["instance"]["instance_id"],
                    "validation_instance_sha256": launch["instance"][
                        "instance_sha256"
                    ],
                    "validation_launch": _file_record(
                        validation_root / "launch_manifest.json",
                        HashCache(),
                        "table_a_validation_launch",
                    ),
                    "validation_postflight": _file_record(
                        persisted_path,
                        HashCache(),
                        "table_a_validation_postflight",
                    ),
                }
            )
    if len(instances) != 6 or len(
        {str(value["instance_sha256"]) for value in instances}
    ) != 6:
        raise TableAEvaluationError("final gate requires six unique instances")
    if len({_canonical_json_sha256(value) for value in runtime_contracts}) != 1:
        raise TableAEvaluationError(
            "Table-A validation instances used different formal runtimes"
        )
    if release_prerequisites != _release_prerequisite_records(
        replay_selection=True
    ):
        raise TableAEvaluationError(
            "Table-A release prerequisites changed while sealing the gate"
        )
    for evidence in validation_provenance:
        kind = str(evidence["kind"])
        seed = int(evidence["seed"])
        root = canonical_output_dir(kind, VALIDATION_PROFILE, seed)
        _verify_file_record(
            evidence["validation_launch"],
            label=f"Table-A {kind}/seed{seed} validation launch",
            roles=("table_a_validation_launch",),
            expected_path=root / "launch_manifest.json",
        )
        _verify_file_record(
            evidence["validation_postflight"],
            label=f"Table-A {kind}/seed{seed} validation postflight",
            roles=("table_a_validation_postflight",),
            expected_path=root / "postflight.json",
        )
    gate = {
        "schema": FINAL_GATE_SCHEMA,
        "status": "sealed",
        "selection_frozen": True,
        "all_paper_ablations_completed": True,
        "created_before_first_final_evaluation": True,
        "selection_rule": (
            "promote_all_predeclared_candidate_and_g0c_seeds_from_completed_"
            "validation_without_test_metric_selection"
        ),
        "instances": instances,
        "runtime_contract": runtime_contracts[0],
        "release_prerequisites": release_prerequisites,
        "validation_provenance": validation_provenance,
        "sealed_at_utc": _utc_now(),
    }
    gate["gate_sha256"] = _canonical_json_sha256(gate)
    _write_json_exclusive(path, gate)
    _validate_final_gate(path, instances[0])
    return gate


def _resolve_runtime(args: argparse.Namespace) -> Runtime:
    python = Path(args.python).expanduser().resolve(strict=True)
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise TableAEvaluationError(f"Python is not executable: {python}")
    if not data_root.is_dir():
        raise TableAEvaluationError(f"data root is not a directory: {data_root}")
    if int(args.batch_size) <= 0 or int(args.num_workers) < 0:
        raise TableAEvaluationError("batch size must be positive and workers nonnegative")
    return Runtime(
        python=python,
        data_root=data_root,
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=not bool(args.no_amp),
    )


def _category_rows() -> list[Mapping[str, Any]]:
    from tools.build_stageb_table_a_category_interventions import verify

    summary = verify(
        output=CATEGORY_JSONL,
        output_support_tsv=CATEGORY_SUPPORT,
        audit_path=CATEGORY_AUDIT,
        require_canonical=True,
    )
    if summary.get("pairs") != 512 or summary.get("rows") != 1024:
        raise TableAEvaluationError("category intervention is not the locked 512-pair set")
    rows: list[Mapping[str, Any]] = []
    with CATEGORY_JSONL.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TableAEvaluationError(
                    f"category JSONL row {line_number} is not an object"
                )
            rows.append(value)
    receipt = role_eval.verify_category_runtime_assets(rows, CATEGORY_SUPPORT)
    if (
        receipt.get("images_rehashed") != 512
        or receipt.get("supports_rehashed") != 318
    ):
        raise TableAEvaluationError("category runtime asset cardinality drifted")
    return rows


def _category_asset_paths(rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: set[Path] = set()
    for row in rows:
        intervention = row.get("category_intervention")
        if not isinstance(intervention, Mapping):
            raise TableAEvaluationError("category row has no intervention metadata")
        paths.add(Path(str(intervention["image_path"])).resolve(strict=True))
        for key in ("class_a", "class_b"):
            category = intervention.get(key)
            if not isinstance(category, Mapping):
                raise TableAEvaluationError(f"category row has no {key}")
            paths.add(Path(str(category["support_path"])).resolve(strict=True))
    return sorted(paths)


def _source_records(
    source: Any, cache: HashCache, records: dict[str, dict[str, Any]]
) -> None:
    _merge_record(records, source.checkpoint, cache, "evaluation_checkpoint")
    for path in paper_eval._config_paths(source.config):
        _merge_record(records, path, cache, "config_dependency")
    for attribute, role in (
        ("sequence_manifest", "training_sequence_manifest"),
        ("final_phase_manifest", "training_phase_manifest"),
        ("training_postflight", "training_postflight"),
        ("training_plan", "g0c_training_plan"),
        ("training_queue_manifest", "training_queue_manifest"),
        (
            "training_queue_detached_launch",
            "training_queue_detached_launch",
        ),
        (
            "training_queue_detached_status",
            "training_queue_detached_status",
        ),
    ):
        path = getattr(source, attribute, None)
        if path is not None:
            _merge_record(records, path, cache, role)
    for path in getattr(source, "training_data", ()):
        _merge_record(records, path, cache, "training_data")
    source_kind = str(getattr(source, "kind", ""))
    if source_kind == G0C_SOURCE_KIND:
        family = str(getattr(source, "source_family", ""))
        paths = tuple(getattr(source, "source_provenance_dependencies", ()))
        if family != G0C_SOURCE_FAMILY or not paths:
            raise TableAEvaluationError(
                "G0c source-family provenance dependencies are missing"
            )
    else:
        try:
            family = paper_eval.evaluation_source_family(source)
            paths = tuple(paper_eval._evaluation_source_provenance_paths(source))
        except (paper_eval.PaperEvaluationError, TypeError, ValueError) as exc:
            raise TableAEvaluationError(
                f"Table-A source-family provenance resolution failed: {exc}"
            ) from exc
    family_role = f"source_family_{family}_provenance_dependency"
    for path in paths:
        _merge_record(
            records,
            path,
            cache,
            "source_provenance_dependency",
            family_role,
        )


def _base_data_and_code_records(
    runtime: Runtime,
    cache: HashCache,
    records: dict[str, dict[str, Any]],
    *,
    role_mode: bool,
) -> None:
    selected_python = _require_current_python_runtime(
        runtime.python, label="Table-A evaluation"
    )
    _merge_record(
        records,
        selected_python,
        cache,
        "evaluation_python_runtime",
    )
    for path in paper_eval._data_input_paths(runtime.data_root):
        _merge_record(records, path, cache, "evaluation_data_dependency")
    common_code_paths = set(paper_eval.evaluation_common_code_paths())
    for path in sorted(common_code_paths):
        _merge_record(records, path, cache, "evaluation_code_dependency")
    if role_mode:
        table_a_code_paths = {
            path.resolve(strict=True) for path in ROLE_CODE_PATHS
        }
    else:
        table_a_code_paths = {
            TEXT_EVALUATOR.resolve(strict=True),
            Path(__file__).resolve(),
        }
    for path in sorted(table_a_code_paths - common_code_paths):
        _merge_record(records, path, cache, "table_a_evaluation_code_dependency")
    for path in g0c_controls._native_runtime_dependency_paths():
        _merge_record(records, path, cache, "evaluation_native_runtime_dependency")


def _strict_record(label: str, cache: HashCache) -> dict[str, Any]:
    record = paper_eval._strict_manifest_record(label, paper_eval.HashCache())
    path = Path(record["path"])
    observed = _file_record(path, cache, label, "locked_tn_source_manifest")
    observed["rows"] = int(record["rows"])
    observed["source_counts"] = dict(record["source_counts"])
    return observed


def _strict_taxonomy_counts(path: Path) -> dict[str, int]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TableAEvaluationError("strict manifest row is not an object")
                rows.append(value)
    metadata = role_eval._metadata_index(rows)
    counts: dict[str, int] = {}
    for row in metadata.values():
        taxonomy = role_eval.normalize_edit_taxonomy(
            role_eval._raw_edit_categories(row)
        )
        counts[taxonomy] = counts.get(taxonomy, 0) + 1
    if sum(counts.values()) != len(rows):
        raise TableAEvaluationError("strict taxonomy accounting is incomplete")
    return dict(sorted(counts.items()))


def _candidate_contract(source: Any) -> dict[str, Any]:
    cfg = SLConfig.fromfile(str(source.config))
    expected_flags = {
        "stage_b_v11_fixed_text": True,
        "stage_b_v15_patch_rank_fusion": True,
        "stage_b_v19_explicit_confidence_output_contract": True,
    }
    for key, expected in expected_flags.items():
        if bool(getattr(cfg, key, False)) is not expected:
            raise TableAEvaluationError(f"candidate config drifted at {key}")
    candidate_topk = int(getattr(cfg, "stage_b_v11_candidate_topk", 0) or 0)
    query_count = int(getattr(cfg, "num_queries", 0) or 0)
    if candidate_topk != 50 or query_count <= candidate_topk:
        raise TableAEvaluationError(
            "Table A requires patch Top-50 below a larger all-query domain"
        )
    return {
        "candidate_topk": candidate_topk,
        "all_query_count": query_count,
        "required_rows": ["G1", "G2", "G3", "G4", "G5"],
        "required_edit_taxonomies": list(REQUIRED_EDIT_TAXONOMIES),
        "category_pairs": 512,
        "category_arms": 1024,
        "ref_manifest_contract": dict(TABLE_A_REF_SPLIT_CONTRACT),
    }


def _validate_candidate_source(
    source: Any,
    *,
    training_run_root: Path,
    training_queue_dir: Path | None,
) -> int:
    try:
        seed = int(source.training_seed)
    except (TypeError, ValueError) as exc:
        raise TableAEvaluationError("candidate training seed is invalid") from exc
    if seed not in FORMAL_SEEDS:
        raise TableAEvaluationError("candidate seed is not a predeclared Table-A seed")
    locked = LOCKED_CANDIDATE_QUEUES[seed]
    if training_queue_dir is None:
        raise TableAEvaluationError(
            "candidate Table-A evaluation requires its predeclared training queue"
        )
    queue_dir = training_queue_dir.expanduser().resolve(strict=True)
    if queue_dir != Path(str(locked["path"])).resolve(strict=True):
        raise TableAEvaluationError("candidate training queue path is not predeclared")
    expected_root = (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/token_ablation_frozen_v2/L4"
        / f"seed{seed}"
    ).resolve(strict=True)
    if (
        training_run_root.expanduser().resolve(strict=True) != expected_root
        or source.training_run_root is None
        or Path(source.training_run_root).resolve(strict=True) != expected_root
        or source.training_run_id != f"L4:{seed}"
        or source.evaluation_id != f"L4_seed{seed}"
        or source.training_queue_id != locked["queue_id"]
        or source.training_queue_plan_sha256 != locked["plan_sha256"]
    ):
        raise TableAEvaluationError(
            "candidate source is not the locked formal L4 run/queue identity"
        )
    manifest = source.training_queue_manifest
    if manifest is None or Path(manifest).resolve(strict=True).parent != queue_dir:
        raise TableAEvaluationError("candidate queue manifest parent drifted")
    return seed


def _profile_surface(
    profile: str, cache: HashCache
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, dict[str, Any]]]:
    if profile == VALIDATION_PROFILE:
        try:
            screen = paper_eval._screen_calibration_contract(paper_eval.HashCache())
        except Exception as exc:
            raise TableAEvaluationError(
                f"screen calibration contract failed: {exc}"
            ) from exc
        source = _file_record(
            SCREEN_CALIBRATION_SOURCE, cache, "validation_tn_source_manifest"
        )
        source["rows"] = int(screen["source_manifest"]["rows"])
        source["profile"] = VALIDATION_PROFILE
        audit = _file_record(
            SCREEN_CALIBRATION_AUDIT, cache, "validation_tn_source_audit"
        )
        return VALIDATION_REF_SPLITS, source, {"screen_calibration_audit": audit}
    if profile == FINAL_PROFILE:
        strict = {
            label: _strict_record(label, cache)
            for label in ("strict2031", "strict1607")
        }
        return tuple(paper_eval.REF_SPLITS), strict["strict2031"], strict
    raise TableAEvaluationError(f"unsupported Table-A evaluation profile: {profile!r}")


def _bind_final_gate(
    *,
    profile: str,
    final_gate: Path | None,
    instance: Mapping[str, Any],
    cache: HashCache,
    records: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if profile == VALIDATION_PROFILE:
        if final_gate is not None:
            raise TableAEvaluationError("validation profile cannot bind a final gate")
        return None
    if final_gate is None:
        raise TableAEvaluationError("final profile requires the sealed one-time gate")
    gate = _validate_final_gate(final_gate, instance)
    consumption_path = _final_consumption_path(instance)
    if consumption_path.exists():
        raise TableAEvaluationError(
            "final Table-A instance was already consumed; rerun is forbidden"
        )
    _merge_record(
        records,
        Path(str(gate["path"])),
        cache,
        "table_a_final_evaluation_gate",
    )
    prerequisites = gate["payload"].get("release_prerequisites")
    if not isinstance(prerequisites, Mapping):
        raise TableAEvaluationError(
            "Table-A final gate lacks release prerequisite records"
        )
    for name in RELEASE_PREREQUISITE_NAMES:
        record = prerequisites.get(name)
        if not isinstance(record, Mapping):
            raise TableAEvaluationError(
                f"Table-A final gate lacks {name}"
            )
        _merge_record(
            records,
            Path(str(record.get("path", ""))),
            cache,
            f"table_a_final_{name}",
        )
    return {
        "path": gate["path"],
        "sha256": gate["sha256"],
        "consumption_path": str(consumption_path),
    }


def _common_text_command(runtime: Runtime, config: Path, checkpoint: Path) -> list[str]:
    command = [
        str(runtime.python),
        str(TEXT_EVALUATOR.resolve(strict=True)),
        "--config",
        str(config),
        "--ckpts",
        str(checkpoint),
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
        "5",
        "10",
        "50",
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
    ]
    if runtime.amp:
        command.append("--amp")
    return command


def _g0c_command_specs(
    *,
    runtime: Runtime,
    source: Any,
    output_dir: Path,
    profile: str,
    ref_splits: Sequence[str],
    tn_primary: Mapping[str, Any],
    tn_inputs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    common = _common_text_command(runtime, Path(source.config), Path(source.checkpoint))
    if profile == VALIDATION_PROFILE:
        primary = [
            *common,
            "--output_dir",
            str(output_dir / "validation_calibration"),
            "--ref_splits",
            *ref_splits,
            "--tn_jsonl",
            str(Path(str(tn_primary["path"])).resolve(strict=True)),
            "--screen_calibration_manifest",
            "--screen_calibration_audit",
            str(SCREEN_CALIBRATION_AUDIT.resolve(strict=True)),
        ]
        phases = [("validation_calibration", primary)]
    elif profile == FINAL_PROFILE:
        primary = [
            *common,
            "--output_dir",
            str(output_dir / "ref8_strict2031"),
            "--ref_splits",
            *ref_splits,
            "--tn_jsonl",
            str(Path(str(tn_primary["path"])).resolve(strict=True)),
            "--tn_splits",
            "refcocop_val",
            "refcocog_umd_val",
        ]
        strict1607 = tn_inputs["strict1607"]
        supplemental = [
            *common,
            "--output_dir",
            str(output_dir / "strict1607"),
            "--skip_ref",
            "--tn_jsonl",
            str(Path(str(strict1607["path"])).resolve(strict=True)),
            "--tn_splits",
            "refcocop_val",
            "refcocog_umd_val",
        ]
        phases = [
            ("ref8_strict2031", primary),
            ("strict1607", supplemental),
        ]
    else:
        raise TableAEvaluationError(f"unsupported G0c profile: {profile!r}")
    return [
        {
            "phase_id": phase_id,
            "command": command,
            "command_shell": shlex.join(command),
            "console_log": str(output_dir / f"{phase_id}_console.log"),
        }
        for phase_id, command in phases
    ]


def build_candidate_plan(
    runtime: Runtime,
    training_run_root: Path,
    output_dir: Path,
    *,
    profile: str = VALIDATION_PROFILE,
    training_queue_dir: Path | None = None,
    final_gate: Path | None = None,
) -> dict[str, Any]:
    cache = HashCache()
    source = paper_eval._resolve_pivot_source(
        training_run_root,
        paper_eval.HashCache(),
        training_queue_dir=training_queue_dir,
    )
    seed = _validate_candidate_source(
        source,
        training_run_root=training_run_root,
        training_queue_dir=training_queue_dir,
    )
    contract = _candidate_contract(source)
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir != canonical_output_dir("candidate", profile, seed).resolve(
        strict=False
    ):
        raise TableAEvaluationError("candidate evaluation output root is not canonical")
    if output_dir.exists():
        raise FileExistsError(f"Table-A output root must be fresh: {output_dir}")
    category_rows = _category_rows()
    records: dict[str, dict[str, Any]] = {}
    _source_records(source, cache, records)
    _base_data_and_code_records(runtime, cache, records, role_mode=True)
    ref_splits, tn_primary, tn_inputs = _profile_surface(profile, cache)
    taxonomy_counts = _strict_taxonomy_counts(Path(str(tn_primary["path"])))
    if sum(taxonomy_counts.values()) != int(tn_primary["rows"]):
        raise TableAEvaluationError("TN taxonomy count differs from row contract")
    missing_taxonomies = sorted(set(REQUIRED_EDIT_TAXONOMIES) - set(taxonomy_counts))
    if missing_taxonomies:
        raise TableAEvaluationError(
            f"TN surface lacks required edit taxonomies: {missing_taxonomies}"
        )
    contract = {
        **contract,
        "profile": profile,
        "ref_splits": list(ref_splits),
        "ref_manifest_contract": {
            split: dict(TABLE_A_REF_SPLIT_CONTRACT[split]) for split in ref_splits
        },
        "tn_taxonomy_counts": taxonomy_counts,
        "validation_has_no_ref_test_or_strict_tn": profile == VALIDATION_PROFILE,
        "final_surface_gate_required": profile == FINAL_PROFILE,
        "category_causal_claim": "joint_canonical_prompt_plus_support_route",
        "patch_only_category_claim_eligible": False,
    }
    records[str(Path(str(tn_primary["path"])).resolve())] = dict(tn_primary)
    for record in tn_inputs.values():
        records[str(Path(str(record["path"])).resolve())] = dict(record)
    for path, role in (
        (CATEGORY_JSONL, "category_intervention_jsonl"),
        (CATEGORY_SUPPORT, "category_support_tsv"),
        (CATEGORY_AUDIT, "category_intervention_audit"),
    ):
        _merge_record(records, path, cache, role)
    for path in _category_asset_paths(category_rows):
        _merge_record(records, path, cache, "category_runtime_asset")

    role_output = output_dir / "role_causal"
    command = [
        str(runtime.python),
        str(ROLE_EVALUATOR.resolve(strict=True)),
        "--config",
        str(source.config),
        "--checkpoint",
        str(source.checkpoint),
        "--output_dir",
        str(role_output),
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
        "--max_batches",
        "0",
        "--true_role_swap",
        "--formal_table_a",
        "--evaluation_profile",
        profile,
        "--ref_splits",
        *ref_splits,
        "--tn_jsonl",
        str(Path(str(tn_primary["path"])).resolve()),
        "--category_jsonl",
        str(CATEGORY_JSONL.resolve()),
        "--category_support_tsv",
        str(CATEGORY_SUPPORT.resolve()),
        "--category_audit",
        str(CATEGORY_AUDIT.resolve()),
    ]
    if profile == VALIDATION_PROFILE:
        command.extend(
            [
                "--screen_calibration_manifest",
                "--screen_calibration_audit",
                str(SCREEN_CALIBRATION_AUDIT.resolve(strict=True)),
            ]
        )
    if runtime.amp:
        command.append("--amp")
    instance = _instance_payload(
        kind="candidate",
        profile=profile,
        seed=seed,
        output_dir=output_dir,
        source=source,
        runtime=runtime,
    )
    gate = _bind_final_gate(
        profile=profile,
        final_gate=final_gate,
        instance=instance,
        cache=cache,
        records=records,
    )
    plan = {
        "schema": SCHEMA,
        "status": "planned",
        "kind": "candidate",
        "profile": profile,
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "evaluation_id": str(source.evaluation_id),
        "output_dir": str(output_dir),
        "source": _jsonable(asdict(source)),
        "runtime": {**_jsonable(asdict(runtime)), "eval_seed": EVAL_SEED},
        "instance": instance,
        "final_gate": gate,
        "contract": contract,
        "tn_manifest": tn_primary,
        "tn_inputs": tn_inputs,
        "commands": [
            {
                "phase_id": f"g1_g5_{profile}_tn_category",
                "command": command,
                "command_shell": shlex.join(command),
                "console_log": str(output_dir / "role_causal_console.log"),
            }
        ],
        "inputs": {
            "algorithm": "sha256",
            "records": sorted(records.values(), key=lambda row: row["path"]),
        },
    }
    _validate_instance(plan)
    _validate_command_surface(plan)
    return plan


def _verified_g0c_file_record(
    value: Any,
    *,
    label: str,
    cache: HashCache,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(value, Mapping):
        raise TableAEvaluationError(f"{label} record is missing")
    try:
        path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TableAEvaluationError(f"{label} path is unavailable: {exc}") from exc
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise TableAEvaluationError(f"{label} path is not canonical")
    if not path.is_file():
        raise TableAEvaluationError(f"{label} is not a regular file")
    expected_sha256 = str(value.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise TableAEvaluationError(f"{label} SHA-256 is invalid")
    if cache.digest(path) != expected_sha256:
        raise TableAEvaluationError(f"{label} SHA-256 changed")
    if "size_bytes" in value and int(value.get("size_bytes", -1)) != int(
        path.stat().st_size
    ):
        raise TableAEvaluationError(f"{label} size changed")
    return path


def _g0c_plan_input_path(
    plan: Mapping[str, Any],
    label: str,
    cache: HashCache,
    *,
    expected_path: Path | None = None,
) -> Path:
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TableAEvaluationError("G0c plan has no sealed inputs")
    return _verified_g0c_file_record(
        inputs.get(label),
        label=f"G0c training input {label}",
        cache=cache,
        expected_path=expected_path,
    )


def _g0c_source_dependency_paths(
    plan: Mapping[str, Any], cache: HashCache
) -> tuple[Path, ...]:
    tree = plan.get("source_dependency_tree")
    if not isinstance(tree, Mapping):
        raise TableAEvaluationError("G0c training source dependency tree is missing")
    try:
        live_tree = g0c_controls._source_dependency_tree()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise TableAEvaluationError(
            f"G0c live source dependency audit failed: {exc}"
        ) from exc
    if tree != live_tree:
        raise TableAEvaluationError(
            "G0c training source dependency tree differs from the formal controller"
        )
    records = tree.get("records")
    if not isinstance(records, list) or not records:
        raise TableAEvaluationError("G0c source dependency records are missing")
    if int(tree.get("file_count", -1)) != len(records):
        raise TableAEvaluationError("G0c source dependency cardinality drifted")
    digest = hashlib.sha256()
    paths: list[Path] = []
    seen_paths: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TableAEvaluationError(
                f"G0c source dependency record {index} is invalid"
            )
        relative_path = str(record.get("relative_path", ""))
        if not relative_path:
            raise TableAEvaluationError(
                f"G0c source dependency record {index} has no relative path"
            )
        path = _verified_g0c_file_record(
            record,
            label=f"G0c source dependency {relative_path}",
            cache=cache,
        )
        if path in seen_paths:
            raise TableAEvaluationError("G0c source dependency path is duplicated")
        seen_paths.add(path)
        paths.append(path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    if digest.hexdigest() != str(tree.get("sha256", "")):
        raise TableAEvaluationError("G0c source dependency tree SHA-256 is invalid")
    controller = Path(g0c_controls.__file__).resolve(strict=True)
    if controller not in seen_paths:
        raise TableAEvaluationError(
            "G0c source dependency tree omits its training controller"
        )
    return tuple(paths)


def _resolve_g0c_source(
    plan_path: Path,
    cache: HashCache,
    *,
    training_queue_dir: Path | None = None,
) -> tuple[G0cEvaluationSource, Mapping[str, Any], Path, int]:
    try:
        plan_path = plan_path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise TableAEvaluationError(
            "canonical G0c v3 formal plan is unavailable; the new telemetry-sealed "
            "u50 soak and formal training must complete first"
        ) from exc
    plan = _read_json(plan_path, label="G0c training plan")
    try:
        g0c_controls._validate_plan_identity(plan)
    except (TypeError, ValueError, KeyError) as exc:
        raise TableAEvaluationError(f"G0c training plan identity failed: {exc}") from exc
    if (
        plan.get("schema") != g0c_controls.PLAN_SCHEMA
        or plan.get("row_id") != "G0c"
        or plan.get("purpose") != "formal"
    ):
        raise TableAEvaluationError("G0c training plan schema/row mismatch")
    contract = plan.get("matched_contract")
    if not isinstance(contract, Mapping):
        raise TableAEvaluationError("G0c training plan has no matched contract")
    try:
        seed = int(contract.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise TableAEvaluationError("G0c training seed is invalid") from exc
    if seed not in FORMAL_SEEDS:
        raise TableAEvaluationError("G0c training seed is not predeclared")
    if plan_path != g0c_controls.formal_plan_path(seed).resolve(strict=True):
        raise TableAEvaluationError("G0c formal plan path is not canonical")
    canonical_output_dir = g0c_controls.formal_output_root(seed).resolve(
        strict=False
    )
    if Path(str(plan.get("output_dir", ""))).resolve(strict=False) != (
        canonical_output_dir
    ):
        raise TableAEvaluationError("G0c formal training root is not canonical")
    output_dir = Path(str(plan.get("output_dir", ""))).resolve(strict=True)
    config = _g0c_plan_input_path(
        plan, "config", cache, expected_path=g0c_controls.CONFIG
    )
    source_dependencies = _g0c_source_dependency_paths(plan, cache)
    if config not in set(source_dependencies):
        raise TableAEvaluationError(
            "G0c source dependency tree omits the selected training config"
        )
    try:
        config_dependencies = set(paper_eval._config_paths(config))
    except paper_eval.PaperEvaluationError as exc:
        raise TableAEvaluationError(
            f"G0c config dependency audit failed: {exc}"
        ) from exc
    if not config_dependencies.issubset(set(source_dependencies)):
        raise TableAEvaluationError(
            "G0c training source tree omits a selected-config dependency"
        )
    postflight_path = (output_dir / "postflight.json").resolve(strict=True)
    postflight = _read_json(postflight_path, label="G0c training postflight")
    if (
        postflight.get("schema") != g0c_controls.POSTFLIGHT_SCHEMA
        or postflight.get("status") != "PASS"
        or postflight.get("row_id") != "G0c"
        or postflight.get("purpose") != "formal"
        or postflight.get("plan_sha256") != plan.get("plan_sha256")
        or postflight.get("source_dependency_tree_sha256")
        != plan.get("source_dependency_tree", {}).get("sha256")
    ):
        raise TableAEvaluationError("G0c training postflight did not pass")
    try:
        observed_postflight = g0c_controls.verify_checkpoint(
            plan, write_postflight=False
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise TableAEvaluationError(
            f"G0c formal checkpoint revalidation failed: {exc}"
        ) from exc
    persisted_stable = dict(postflight)
    observed_stable = dict(observed_postflight)
    persisted_stable.pop("validated_at_utc", None)
    observed_stable.pop("validated_at_utc", None)
    if persisted_stable != observed_stable:
        raise TableAEvaluationError(
            "G0c persisted postflight differs from fresh formal revalidation"
        )
    checkpoint_record = observed_postflight.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise TableAEvaluationError("G0c postflight has no checkpoint record")
    checkpoint = _verified_g0c_file_record(
        checkpoint_record,
        label="G0c trained checkpoint",
        cache=cache,
        expected_path=output_dir / "checkpoint_iter.pth",
    )
    training_data = tuple(
        _g0c_plan_input_path(plan, label, cache)
        for label in sorted(plan["inputs"])
        if label == "dataset" or str(label).startswith("training_jsonl_")
    )
    if len(training_data) != 5:
        raise TableAEvaluationError(
            "G0c formal source requires one dataset and four training JSONLs"
        )
    source = G0cEvaluationSource(
        kind=G0C_SOURCE_KIND,
        evaluation_id=f"G0c_seed{seed}",
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=str(checkpoint_record["sha256"]),
        training_run_id=f"G0c:{seed}",
        training_seed=seed,
        training_run_root=output_dir,
        training_phase="final",
        diagnostic_only=False,
        final_phase_id="formal",
        training_postflight=postflight_path,
        selected_phase_id="formal",
        selected_training_postflight=postflight_path,
        training_data=training_data,
        training_plan=plan_path,
        training_plan_contract_sha256=str(plan["plan_sha256"]),
        source_dependency_tree_sha256=str(plan["source_dependency_tree"]["sha256"]),
        source_provenance_dependencies=source_dependencies,
    )
    if training_queue_dir is not None:
        from tools import run_stageb_table_a_g0c_queues as g0c_queue

        try:
            attestation = g0c_queue.verify_training_run(
                training_queue_dir, seed, require_canonical_path=True
            )
        except (
            g0c_queue.G0cQueueError,
            FileNotFoundError,
            NotADirectoryError,
            OSError,
            ValueError,
        ) as exc:
            raise TableAEvaluationError(
                f"G0c dedicated training queue attestation failed: {exc}"
            ) from exc
        expected_artifacts = {
            "formal_plan": plan_path,
            "postflight": postflight_path,
            "checkpoint": checkpoint,
        }
        for key, expected_path in expected_artifacts.items():
            record = attestation.get(key)
            if (
                not isinstance(record, Mapping)
                or Path(str(record.get("path", ""))).resolve(strict=True)
                != expected_path
                or str(record.get("sha256", "")) != cache.digest(expected_path)
            ):
                raise TableAEvaluationError(
                    f"G0c training queue {key} differs from resolved source"
                )
        source = G0cEvaluationSource(
            **{
                **source.__dict__,
                "training_queue_manifest": Path(
                    str(attestation["queue_manifest"])
                ).resolve(strict=True),
                "training_queue_detached_launch": Path(
                    str(attestation["job_launch"])
                ).resolve(strict=True),
                "training_queue_detached_status": Path(
                    str(attestation["job_status"])
                ).resolve(strict=True),
                "training_queue_id": str(attestation["queue_id"]),
                "training_queue_plan_sha256": str(
                    attestation["queue_plan_sha256"]
                ),
            }
        )
    return source, plan, postflight_path, seed


def _g0c_training_provenance_contract(
    source: G0cEvaluationSource,
    training_plan: Mapping[str, Any],
    cache: HashCache,
) -> dict[str, Any]:
    if (
        not isinstance(source, G0cEvaluationSource)
        or source.kind != G0C_SOURCE_KIND
        or source.source_family != G0C_SOURCE_FAMILY
        or source.training_plan is None
        or source.training_postflight is None
        or source.training_queue_manifest is None
        or source.training_queue_detached_launch is None
        or source.training_queue_detached_status is None
        or not source.training_queue_id
        or re.fullmatch(
            r"[0-9a-f]{64}", str(source.training_queue_plan_sha256)
        )
        is None
        or source.training_seed not in FORMAL_SEEDS
        or not source.source_provenance_dependencies
        or source.training_plan_contract_sha256
        != training_plan.get("plan_sha256")
        or re.fullmatch(
            r"[0-9a-f]{64}", str(source.training_plan_contract_sha256)
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(source.source_dependency_tree_sha256)
        )
        is None
    ):
        raise TableAEvaluationError("resolved G0c source provenance is incomplete")

    def identity(path: Path) -> dict[str, Any]:
        path = path.resolve(strict=True)
        return {
            "path": str(path),
            "sha256": cache.digest(path),
            "size_bytes": int(path.stat().st_size),
        }

    return {
        "schema": G0C_TRAINING_PROVENANCE_SCHEMA,
        "source_kind": G0C_SOURCE_KIND,
        "source_family": G0C_SOURCE_FAMILY,
        "training_run_id": source.training_run_id,
        "training_seed": source.training_seed,
        "training_plan": {
            **identity(source.training_plan),
            "contract_sha256": str(training_plan.get("plan_sha256", "")),
        },
        "training_postflight": identity(source.training_postflight),
        "checkpoint": identity(source.checkpoint),
        "config": identity(source.config),
        "source_dependency_tree_sha256": source.source_dependency_tree_sha256,
        "source_dependency_count": len(source.source_provenance_dependencies),
        "training_queue": {
            "queue_id": source.training_queue_id,
            "plan_sha256": source.training_queue_plan_sha256,
            "manifest": identity(source.training_queue_manifest),
            "job_launch": identity(source.training_queue_detached_launch),
            "job_status": identity(source.training_queue_detached_status),
        },
    }


def _validate_g0c_launch_provenance(plan: Mapping[str, Any]) -> None:
    if plan.get("kind") != "g0c":
        return
    source = plan.get("source")
    contract = plan.get("contract")
    provenance = (
        contract.get("training_provenance")
        if isinstance(contract, Mapping)
        else None
    )
    if not isinstance(source, Mapping) or not isinstance(provenance, Mapping):
        raise TableAEvaluationError("G0c launch training provenance is missing")
    try:
        seed = int(source.get("training_seed", -1))
    except (TypeError, ValueError) as exc:
        raise TableAEvaluationError("G0c launch training seed is invalid") from exc
    if seed not in FORMAL_SEEDS:
        raise TableAEvaluationError("G0c launch training seed is not predeclared")
    source_dependencies = source.get("source_provenance_dependencies")
    if not isinstance(source_dependencies, list) or not source_dependencies:
        raise TableAEvaluationError(
            "G0c launch source dependency paths are missing"
        )
    canonical_root = g0c_controls.formal_output_root(seed).resolve(strict=False)
    canonical_plan = g0c_controls.formal_plan_path(seed).resolve(strict=False)
    expected_source = {
        "kind": G0C_SOURCE_KIND,
        "source_family": G0C_SOURCE_FAMILY,
        "evaluation_id": f"G0c_seed{seed}",
        "training_run_id": f"G0c:{seed}",
        "training_seed": seed,
        "training_run_root": str(canonical_root),
        "training_plan": str(canonical_plan),
        "training_postflight": str(canonical_root / "postflight.json"),
        "selected_training_postflight": str(
            canonical_root / "postflight.json"
        ),
        "checkpoint": str(canonical_root / "checkpoint_iter.pth"),
        "config": str(g0c_controls.CONFIG.resolve(strict=False)),
        "training_phase": "final",
        "diagnostic_only": False,
        "final_phase_id": "formal",
        "selected_phase_id": "formal",
    }
    for key, expected in expected_source.items():
        observed = source.get(key)
        if key in {
            "training_run_root",
            "training_plan",
            "training_postflight",
            "checkpoint",
            "config",
        }:
            observed = str(Path(str(observed)).expanduser().resolve(strict=False))
        if observed != expected:
            raise TableAEvaluationError(f"G0c launch source drifted at {key}")
    if plan.get("evaluation_id") != source.get("evaluation_id"):
        raise TableAEvaluationError("G0c launch evaluation identity drifted")
    for key in (
        "checkpoint_sha256",
        "training_plan_contract_sha256",
        "source_dependency_tree_sha256",
        "training_queue_plan_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(source.get(key, ""))) is None:
            raise TableAEvaluationError(f"G0c launch source {key} is invalid")
    if not isinstance(source.get("training_queue_id"), str) or not source.get(
        "training_queue_id"
    ):
        raise TableAEvaluationError("G0c launch training queue ID is invalid")
    for key in (
        "training_queue_manifest",
        "training_queue_detached_launch",
        "training_queue_detached_status",
    ):
        value = source.get(key)
        if not isinstance(value, str) or not value:
            raise TableAEvaluationError(f"G0c launch source {key} is missing")
    expected_provenance = {
        "schema": G0C_TRAINING_PROVENANCE_SCHEMA,
        "source_kind": G0C_SOURCE_KIND,
        "source_family": G0C_SOURCE_FAMILY,
        "training_run_id": f"G0c:{seed}",
        "training_seed": seed,
        "source_dependency_tree_sha256": source.get(
            "source_dependency_tree_sha256"
        ),
        "source_dependency_count": len(source_dependencies),
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise TableAEvaluationError(
                f"G0c training provenance drifted at {key}"
            )
    queue_provenance = provenance.get("training_queue")
    if (
        not isinstance(queue_provenance, Mapping)
        or queue_provenance.get("queue_id") != source.get("training_queue_id")
        or queue_provenance.get("plan_sha256")
        != source.get("training_queue_plan_sha256")
    ):
        raise TableAEvaluationError("G0c training queue provenance drifted")
    training_plan_identity = provenance.get("training_plan")
    if not isinstance(training_plan_identity, Mapping):
        raise TableAEvaluationError("G0c training plan identity is missing")
    if source.get("training_plan_contract_sha256") != training_plan_identity.get(
        "contract_sha256"
    ):
        raise TableAEvaluationError("G0c training plan contract SHA-256 drifted")
    checkpoint_identity = provenance.get("checkpoint")
    if (
        not isinstance(checkpoint_identity, Mapping)
        or source.get("checkpoint_sha256") != checkpoint_identity.get("sha256")
    ):
        raise TableAEvaluationError("G0c trained checkpoint identity drifted")

    records = plan.get("inputs", {}).get("records")
    if not isinstance(records, list):
        raise TableAEvaluationError("G0c launch input records are missing")
    records_by_path = {
        str(Path(str(record.get("path", ""))).expanduser().resolve(strict=False)): record
        for record in records
        if isinstance(record, Mapping)
    }
    if len(records_by_path) != len(records):
        raise TableAEvaluationError("G0c launch input paths are invalid or duplicated")

    def require_record(
        identity: Any, *required_roles: str
    ) -> Mapping[str, Any]:
        if not isinstance(identity, Mapping):
            raise TableAEvaluationError("G0c provenance artifact identity is missing")
        path = str(Path(str(identity.get("path", ""))).resolve(strict=False))
        record = records_by_path.get(path)
        if not isinstance(record, Mapping):
            raise TableAEvaluationError(f"G0c provenance input is missing: {path}")
        try:
            sizes_match = int(record.get("size_bytes", -1)) == int(
                identity.get("size_bytes", -2)
            )
        except (TypeError, ValueError) as exc:
            raise TableAEvaluationError(
                f"G0c provenance input size is invalid: {path}"
            ) from exc
        if record.get("sha256") != identity.get("sha256") or not sizes_match:
            raise TableAEvaluationError(f"G0c provenance input drifted: {path}")
        roles = record.get("roles")
        if not isinstance(roles, list) or not set(required_roles).issubset(roles):
            raise TableAEvaluationError(
                f"G0c provenance roles are missing for {path}"
            )
        return record

    require_record(
        provenance.get("training_plan"), "g0c_training_plan"
    )
    require_record(
        provenance.get("training_postflight"), "g0c_training_postflight"
    )
    require_record(provenance.get("checkpoint"), "evaluation_checkpoint")
    require_record(provenance.get("config"), "config_dependency")
    require_record(
        queue_provenance.get("manifest"), "training_queue_manifest"
    )
    require_record(
        queue_provenance.get("job_launch"), "training_queue_detached_launch"
    )
    require_record(
        queue_provenance.get("job_status"), "training_queue_detached_status"
    )
    family_role = f"source_family_{G0C_SOURCE_FAMILY}_provenance_dependency"
    source_paths = {
        str(Path(str(path)).expanduser().resolve(strict=False))
        for path in source_dependencies
    }
    family_paths = {
        path
        for path, record in records_by_path.items()
        if family_role in record.get("roles", ())
    }
    try:
        expected_dependency_count = int(
            provenance.get("source_dependency_count", -1)
        )
    except (TypeError, ValueError) as exc:
        raise TableAEvaluationError(
            "G0c source dependency count is invalid"
        ) from exc
    if len(source_paths) != expected_dependency_count or family_paths != source_paths:
        raise TableAEvaluationError(
            "G0c source-family provenance input surface is incomplete"
        )
    for path in source_paths:
        roles = records_by_path[path].get("roles", ())
        if "source_provenance_dependency" not in roles:
            raise TableAEvaluationError(
                f"G0c generic source provenance role is missing for {path}"
            )


def _replay_g0c_training_provenance(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    if plan.get("kind") != "g0c":
        return None
    source = plan.get("source")
    contract = plan.get("contract")
    if not isinstance(source, Mapping) or not isinstance(contract, Mapping):
        raise TableAEvaluationError("G0c provenance replay lacks source/contract")
    training_plan_value = source.get("training_plan")
    if not isinstance(training_plan_value, str) or not training_plan_value:
        raise TableAEvaluationError("G0c provenance replay has no training plan")
    queue_manifest_value = source.get("training_queue_manifest")
    if not isinstance(queue_manifest_value, str) or not queue_manifest_value:
        raise TableAEvaluationError("G0c provenance replay has no training queue")
    training_queue_dir = Path(queue_manifest_value).expanduser().resolve(
        strict=True
    ).parent
    cache = HashCache()
    resolved, training_plan, training_postflight, seed = _resolve_g0c_source(
        Path(training_plan_value),
        cache,
        training_queue_dir=training_queue_dir,
    )
    expected_source = _jsonable(asdict(resolved))
    if dict(source) != expected_source:
        differing = sorted(
            key
            for key in set(source).union(expected_source)
            if source.get(key) != expected_source.get(key)
        )
        raise TableAEvaluationError(
            "G0c launch source differs from fresh canonical resolution: "
            + ", ".join(differing)
        )
    expected_provenance = _g0c_training_provenance_contract(
        resolved, training_plan, cache
    )
    if contract.get("training_provenance") != expected_provenance:
        raise TableAEvaluationError(
            "G0c launch training provenance differs from fresh canonical resolution"
        )
    matched_contract = training_plan.get("matched_contract")
    if not isinstance(matched_contract, Mapping) or contract.get(
        "training_contract"
    ) != dict(matched_contract):
        raise TableAEvaluationError(
            "G0c launch training contract differs from fresh canonical resolution"
        )
    return {
        "status": "passed",
        "training_seed": seed,
        "training_run_id": resolved.training_run_id,
        "training_plan": {
            "path": str(resolved.training_plan),
            "sha256": cache.digest(Path(resolved.training_plan)),
            "contract_sha256": resolved.training_plan_contract_sha256,
        },
        "training_postflight": {
            "path": str(training_postflight),
            "sha256": cache.digest(training_postflight),
        },
        "checkpoint": {
            "path": str(resolved.checkpoint),
            "sha256": resolved.checkpoint_sha256,
        },
        "training_contract_sha256": _canonical_json_sha256(matched_contract),
        "source_dependency_tree_sha256": resolved.source_dependency_tree_sha256,
    }


def build_g0c_plan(
    runtime: Runtime,
    training_plan_path: Path,
    output_dir: Path,
    *,
    profile: str = VALIDATION_PROFILE,
    training_queue_dir: Path | None = None,
    final_gate: Path | None = None,
) -> dict[str, Any]:
    if training_queue_dir is None:
        raise TableAEvaluationError(
            "G0c evaluation requires its dedicated completed training queue"
        )
    cache = HashCache()
    source, training_plan, training_postflight, seed = _resolve_g0c_source(
        training_plan_path,
        cache,
        training_queue_dir=training_queue_dir,
    )
    output_dir = output_dir.expanduser().resolve(strict=False)
    if output_dir != canonical_output_dir("g0c", profile, seed).resolve(strict=False):
        raise TableAEvaluationError("G0c evaluation output root is not canonical")
    if output_dir.exists():
        raise FileExistsError(f"Table-A output root must be fresh: {output_dir}")
    records: dict[str, dict[str, Any]] = {}
    _source_records(source, cache, records)
    _base_data_and_code_records(runtime, cache, records, role_mode=False)
    _merge_record(records, training_plan_path, cache, "g0c_training_plan")
    _merge_record(records, training_postflight, cache, "g0c_training_postflight")
    training_provenance = _g0c_training_provenance_contract(
        source, training_plan, cache
    )
    ref_splits, tn_primary, tn_inputs = _profile_surface(profile, cache)
    records[str(Path(str(tn_primary["path"])).resolve())] = dict(tn_primary)
    for record in tn_inputs.values():
        records[str(Path(record["path"]).resolve())] = record
    commands = _g0c_command_specs(
        runtime=runtime,
        source=source,
        output_dir=output_dir,
        profile=profile,
        ref_splits=ref_splits,
        tn_primary=tn_primary,
        tn_inputs=tn_inputs,
    )
    instance = _instance_payload(
        kind="g0c",
        profile=profile,
        seed=seed,
        output_dir=output_dir,
        source=source,
        runtime=runtime,
    )
    gate = _bind_final_gate(
        profile=profile,
        final_gate=final_gate,
        instance=instance,
        cache=cache,
        records=records,
    )
    plan = {
        "schema": SCHEMA,
        "status": "planned",
        "kind": "g0c",
        "profile": profile,
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "evaluation_id": str(source.evaluation_id),
        "output_dir": str(output_dir),
        "source": _jsonable(asdict(source)),
        "runtime": {**_jsonable(asdict(runtime)), "eval_seed": EVAL_SEED},
        "instance": instance,
        "final_gate": gate,
        "contract": {
            "table_a_row": "G0c",
            "profile": profile,
            "ref_splits": list(ref_splits),
            "ref_manifest_contract": {
                split: dict(TABLE_A_REF_SPLIT_CONTRACT[split]) for split in ref_splits
            },
            "topks": [1, 5, 10, 50, "all"],
            "validation_has_no_ref_test_or_strict_tn": profile == VALIDATION_PROFILE,
            "final_surface_gate_required": profile == FINAL_PROFILE,
            "training_contract": dict(training_plan["matched_contract"]),
            "training_provenance": training_provenance,
        },
        "tn_manifest": tn_primary,
        "tn_inputs": tn_inputs,
        "commands": commands,
        "inputs": {
            "algorithm": "sha256",
            "records": sorted(records.values(), key=lambda row: row["path"]),
        },
    }
    _validate_instance(plan)
    _validate_command_surface(plan)
    _validate_g0c_launch_provenance(plan)
    return plan


def _verify_inputs(plan: Mapping[str, Any], *, hash_content: bool) -> dict[str, Any]:
    cache = HashCache()
    rows = []
    records = plan.get("inputs", {}).get("records")
    if not isinstance(records, list) or not records:
        raise TableAEvaluationError("evaluation plan has no input records")
    for record in records:
        if not isinstance(record, Mapping):
            raise TableAEvaluationError("evaluation input record is invalid")
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        stat = path.stat()
        passed = (
            int(record.get("size_bytes", -1)) == int(stat.st_size)
            and int(record.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
        )
        observed_sha = None
        if hash_content:
            observed_sha = cache.digest(path)
            passed = passed and observed_sha == str(record.get("sha256", ""))
        rows.append(
            {
                "path": str(path),
                "expected_sha256": str(record.get("sha256", "")),
                "observed_sha256": observed_sha,
                "passed": bool(passed),
            }
        )
    failed = [row["path"] for row in rows if not row["passed"]]
    if failed:
        raise TableAEvaluationError(f"evaluation inputs drifted: {failed[:5]}")
    return {
        "schema": INPUT_REHASH_SCHEMA,
        "status": "passed",
        "verified_at_utc": _utc_now(),
        "records": rows,
    }


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TableAEvaluationError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TableAEvaluationError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TableAEvaluationError(f"{label} must be exact")
    return result


def _finite_metric(value: Any, label: str, *, probability: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TableAEvaluationError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise TableAEvaluationError(f"{label} is non-finite")
    if probability and not 0.0 <= result <= 1.0:
        raise TableAEvaluationError(f"{label} is outside [0,1]")
    return result


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _validate_g0c_summary_provenance(
    row: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_seed: int,
) -> None:
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise TableAEvaluationError("G0c plan source is missing")
    runtime = _formal_runtime_contract(plan.get("runtime", {}))
    config = Path(str(source.get("config", ""))).resolve(strict=True)
    checkpoint = Path(str(source.get("checkpoint", ""))).resolve(strict=True)
    expected = {
        "config": str(config),
        "config_sha256": HashCache().digest(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(source.get("checkpoint_sha256", "")),
        "checkpoint_name": checkpoint.name,
        "run_id": paper_eval._checkpoint_run_id(checkpoint),
        "seed": int(expected_seed),
        "batch_size": int(runtime["batch_size"]),
        "num_workers": int(runtime["num_workers"]),
        "amp": bool(runtime["amp"]),
        "device": str(runtime["device"]),
        "data_root": str(runtime["data_root"]),
        "max_batches": 0,
    }
    for key, value in expected.items():
        observed = row.get(key)
        if key in {"seed", "batch_size", "num_workers", "max_batches"}:
            observed = _required_int(observed, f"G0c summary {key}")
        if key == "amp" and type(observed) is not bool:
            raise TableAEvaluationError("G0c summary amp must be an exact boolean")
        if observed != value:
            raise TableAEvaluationError(
                f"G0c summary provenance/runtime mismatch at {key}"
            )


def _replay_g0c_ref_records(
    path: Path,
    *,
    row: Mapping[str, Any],
    split: str,
) -> dict[str, float]:
    expected_topks = ("1", "5", "10", "50")
    expected_n = _required_int(row.get("manifest_n"), f"G0c {split} manifest_n")
    expected_sha = str(row.get("manifest_sha256", ""))
    expected_run_id = str(row.get("run_id", ""))
    topk_values = {key: [] for key in expected_topks}
    all_query: list[float] = []
    sample_ids: set[str] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(line for line in handle if line.strip()):
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TableAEvaluationError(
                    f"G0c {split} record {index} is invalid JSON"
                ) from exc
            if not isinstance(record, Mapping):
                raise TableAEvaluationError(f"G0c {split} record {index} is invalid")
            identity_ok = (
                record.get("schema") == "stageb-eval-record-v1"
                and record.get("task") == "ref"
                and record.get("split") == split
                and record.get("manifest_key") == f"ref:{split}"
                and str(record.get("manifest_sha256", "")) == expected_sha
                and _required_int(
                    record.get("manifest_n"), f"G0c {split} record manifest_n"
                )
                == expected_n
                and _required_int(
                    record.get("manifest_index"),
                    f"G0c {split} record manifest_index",
                )
                == index
                and record.get("run_id") == expected_run_id
                and type(record.get("valid")) is bool
                and record.get("valid") is True
            )
            if not identity_ok:
                raise TableAEvaluationError(
                    f"G0c {split} record {index} identity/completeness mismatch"
                )
            sample_id = str(record.get("sample_id", ""))
            if not sample_id or sample_id in sample_ids:
                raise TableAEvaluationError(
                    f"G0c {split} record sample IDs are missing or duplicated"
                )
            sample_ids.add(sample_id)
            ranked = record.get("ranked_best_iou")
            if not isinstance(ranked, Mapping) or set(ranked) != set(expected_topks):
                raise TableAEvaluationError(
                    f"G0c {split} record {index} lacks exact Top-K IoUs"
                )
            previous = -1.0
            for key in expected_topks:
                value = _finite_metric(
                    ranked.get(key),
                    f"G0c {split} record {index} Top-{key} IoU",
                    probability=True,
                )
                if value + 1e-12 < previous:
                    raise TableAEvaluationError(
                        f"G0c {split} record {index} Top-K IoUs are not monotonic"
                    )
                previous = value
                topk_values[key].append(value)
            top1 = _finite_metric(
                record.get("top1_iou"),
                f"G0c {split} record {index} top1_iou",
                probability=True,
            )
            oracle = _finite_metric(
                record.get("all_query_best_iou"),
                f"G0c {split} record {index} all-query IoU",
                probability=True,
            )
            if not math.isclose(top1, topk_values["1"][-1], abs_tol=1e-12):
                raise TableAEvaluationError(f"G0c {split} Top-1 record drifted")
            if oracle + 1e-12 < topk_values["50"][-1]:
                raise TableAEvaluationError(f"G0c {split} oracle is below Top-50")
            if record.get("correct50") is not bool(top1 >= 0.5):
                raise TableAEvaluationError(f"G0c {split} correct50 drifted")
            all_query.append(oracle)
    if count != expected_n or len(sample_ids) != expected_n:
        raise TableAEvaluationError(f"G0c {split} full record count mismatch")
    if _required_int(row.get("num_expressions"), f"G0c {split} num_expressions") != count:
        raise TableAEvaluationError(f"G0c {split} summary expression count mismatch")
    if _required_int(row.get("valid_mask_expressions"), f"G0c {split} valid") != count:
        raise TableAEvaluationError(f"G0c {split} valid expression count mismatch")
    if _required_int(row.get("invalid_mask_expressions"), f"G0c {split} invalid") != 0:
        raise TableAEvaluationError(f"G0c {split} contains invalid expressions")
    measured: dict[str, float] = {}
    for key in expected_topks:
        suffix = "" if key == "1" else f"@{key}"
        values = topk_values[key]
        measured[f"acc50{suffix}"] = float(sum(value >= 0.5 for value in values) / count)
        measured[f"mean_iou{suffix}"] = float(sum(values) / count)
    measured["recall50@all_queries"] = float(
        sum(value >= 0.5 for value in all_query) / count
    )
    measured["mean_best_iou@all_queries"] = float(sum(all_query) / count)
    for key, value in measured.items():
        reported = _finite_metric(row.get(key), f"G0c {split}.{key}", probability=True)
        if not math.isclose(value, reported, rel_tol=0.0, abs_tol=1e-12):
            raise TableAEvaluationError(
                f"G0c {split} summary {key} differs from record replay"
            )
    return measured


def _load_candidate_record_groups(
    path: Path,
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {
        "tn_positive": [],
        "tn_counterfactual": [],
        "category_intervention": [],
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TableAEvaluationError(
                    f"candidate records line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise TableAEvaluationError(
                    f"candidate records line {line_number} is not an object"
                )
            task = str(row.get("task", ""))
            if task == "ref":
                key = f"ref:{row.get('dataset', '')}"
            else:
                key = task
            groups.setdefault(key, []).append(row)
    return groups


def _validate_candidate_records(
    path: Path, *, expected_ref: Mapping[str, int], expected_tn: int
) -> dict[str, Any]:
    task_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TableAEvaluationError(
                    f"candidate records line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, Mapping):
                raise TableAEvaluationError(
                    f"candidate records line {line_number} is not an object"
                )
            task = str(row.get("task", ""))
            task_counts[task] = task_counts.get(task, 0) + 1
            if task in {"ref", "tn_positive"}:
                true_swap = row.get("true_role_swap")
                routes = row.get("routes")
                if (
                    row.get("candidate_source") != "patch_topk"
                    or not isinstance(true_swap, Mapping)
                    or true_swap.get("supported") is not True
                    or not isinstance(routes, Mapping)
                    or role_eval.TRUE_ROLE_SWAP_ROUTE not in routes
                ):
                    raise TableAEvaluationError(
                        f"candidate record {line_number} lacks the proven G5 route"
                    )
            if task == "ref":
                split = str(row.get("dataset", ""))
                ref_counts[split] = ref_counts.get(split, 0) + 1
            elif task == "tn_counterfactual":
                if row.get("causal_comparison_supported") is not True:
                    raise TableAEvaluationError(
                        f"TN causal record {line_number} is unsupported"
                    )
                patch = row.get("surfaces", {}).get("patch")
                if (
                    not isinstance(patch, Mapping)
                    or float(patch.get("delta_max_logit_negative_minus_positive", math.nan))
                    != 0.0
                    or patch.get("top1_changed") is not False
                ):
                    raise TableAEvaluationError(
                        f"TN causal record {line_number} violated patch invariance"
                    )
            elif task == "category_intervention":
                if (
                    row.get("category_causal_evidence_eligible") is not True
                    or row.get("category_causal_route")
                    != "joint_canonical_prompt_plus_support_patch"
                    or row.get("patch_only_category_causal_claim_eligible")
                    is not False
                    or row.get("prompt_and_support_changed_together") is not True
                ):
                    raise TableAEvaluationError(
                        f"category record {line_number} has an invalid causal claim"
                    )
    if ref_counts != dict(expected_ref):
        raise TableAEvaluationError(
            f"candidate Ref per-example split counts mismatch: {ref_counts}"
        )
    expected_tasks = {
        "ref": sum(expected_ref.values()),
        "tn_positive": int(expected_tn),
        "tn_counterfactual": int(expected_tn),
        "category_intervention": 1024,
    }
    if task_counts != expected_tasks:
        raise TableAEvaluationError(
            f"candidate per-example task counts mismatch: {task_counts}"
        )
    return {"task_counts": task_counts, "ref_split_counts": ref_counts}


def _verify_candidate_outputs(plan: Mapping[str, Any], cache: HashCache) -> dict[str, Any]:
    output_dir = Path(str(plan["output_dir"])).resolve(strict=True)
    summary_path = (output_dir / "role_causal/role_causal.summary.json").resolve(
        strict=True
    )
    records_path = (output_dir / "role_causal/role_causal.records.jsonl").resolve(
        strict=True
    )
    summary = _read_json(summary_path, label="candidate role summary")
    if summary.get("schema") != role_eval.SCHEMA:
        raise TableAEvaluationError("candidate role summary schema mismatch")
    profile = str(plan.get("profile", ""))
    expected_ref_splits = tuple(plan["contract"]["ref_splits"])
    if (
        summary.get("diagnostic_only") is not False
        or summary.get("formal_gate_eligible") is not True
        or summary.get("formal_table_a") is not True
        or summary.get("evaluation_profile") != profile
    ):
        raise TableAEvaluationError(
            "candidate output retained diagnostic or non-formal flags"
        )
    expected_formal_contract = {
        "full_dataset": True,
        "fixed_eval_seed_42": True,
        "true_role_swap_required": True,
        "profile_surface_isolated": True,
        "category_claim": "joint_canonical_prompt_plus_support_route",
        "patch_only_category_claim_eligible": False,
    }
    if summary.get("formal_contract") != expected_formal_contract:
        raise TableAEvaluationError("candidate formal summary contract drifted")
    source = plan["source"]
    for label, key in (("config", "config"), ("checkpoint", "checkpoint")):
        record = summary.get(label)
        if not isinstance(record, Mapping):
            raise TableAEvaluationError(f"candidate summary has no {label} record")
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        expected = Path(str(source[key])).resolve(strict=True)
        if path != expected or cache.digest(path) != str(record.get("sha256", "")):
            raise TableAEvaluationError(f"candidate summary {label} binding mismatch")
    if (
        _required_int(summary.get("seed"), "candidate seed") != EVAL_SEED
        or _required_int(summary.get("max_batches"), "candidate max_batches") != 0
        or bool(summary.get("amp")) != bool(plan["runtime"]["amp"])
        or _required_int(summary.get("batch_size"), "candidate batch size")
        != int(plan["runtime"]["batch_size"])
        or _required_int(summary.get("num_workers"), "candidate workers")
        != int(plan["runtime"]["num_workers"])
    ):
        raise TableAEvaluationError("candidate runtime contract mismatch")
    if summary.get("true_role_swap") != role_eval.TRUE_ROLE_SWAP_STATUS:
        raise TableAEvaluationError("candidate summary does not prove the G5 route")
    ref = summary.get("ref")
    if not isinstance(ref, Mapping) or set(ref) != set(expected_ref_splits):
        raise TableAEvaluationError("candidate summary Ref surface differs from profile")
    expected_rows = set(plan["contract"]["required_rows"])
    expected_oracle = {"1", "5", "10", "50", "all"}
    for split in expected_ref_splits:
        split_summary = ref[split]
        if not isinstance(split_summary, Mapping):
            raise TableAEvaluationError(f"{split} role summary is invalid")
        expected_n = int(TABLE_A_REF_SPLIT_CONTRACT[split]["rows"])
        if _required_int(split_summary.get("num_expressions"), split) != expected_n:
            raise TableAEvaluationError(f"{split} expression count mismatch")
        rows = split_summary.get("table_a_rows")
        if not isinstance(rows, Mapping) or set(rows) != expected_rows:
            raise TableAEvaluationError(f"{split} does not expose G1-G5 exactly")
        for row_id, row in rows.items():
            if not isinstance(row, Mapping):
                raise TableAEvaluationError(f"{split}.{row_id} is invalid")
            expected_candidates = (
                int(plan["contract"]["all_query_count"])
                if row_id == "G1"
                else int(plan["contract"]["candidate_topk"])
            )
            if _required_int(row.get("candidate_count"), f"{split}.{row_id}") != expected_candidates:
                raise TableAEvaluationError(f"{split}.{row_id} candidate count mismatch")
            _finite_metric(row.get("acc50"), f"{split}.{row_id}.acc50", probability=True)
            _finite_metric(row.get("mean_selected_iou"), f"{split}.{row_id}.iou", probability=True)
            _finite_metric(
                row.get("top1_query_churn_vs_patch_only"),
                f"{split}.{row_id}.churn",
                probability=True,
            )
            if row_id == "G5":
                _finite_metric(
                    row.get("top1_query_churn_vs_patch_admission_text_rank"),
                    f"{split}.G5.fused_churn",
                    probability=True,
                )
            oracle = row.get("ranked_oracle")
            if not isinstance(oracle, Mapping) or set(oracle) != expected_oracle:
                raise TableAEvaluationError(f"{split}.{row_id} oracle keys mismatch")
            for key, values in oracle.items():
                _finite_metric(
                    values.get("recall_iou50"),
                    f"{split}.{row_id}.recall@{key}",
                    probability=True,
                )
                _finite_metric(
                    values.get("mean_best_iou"),
                    f"{split}.{row_id}.best_iou@{key}",
                    probability=True,
                )
    manifests = summary.get("input_manifests")
    if not isinstance(manifests, Mapping):
        raise TableAEvaluationError("candidate summary has no input manifest bindings")
    for label, expected_path in (
        ("category", CATEGORY_JSONL),
        ("category_support", CATEGORY_SUPPORT),
        ("category_audit", CATEGORY_AUDIT),
    ):
        record = manifests.get(label)
        expected_path = expected_path.resolve(strict=True)
        if (
            not isinstance(record, Mapping)
            or Path(str(record.get("path", ""))).resolve(strict=True) != expected_path
            or str(record.get("sha256", "")) != cache.digest(expected_path)
        ):
            raise TableAEvaluationError(
                f"candidate category input binding drifted: {label}"
            )
    for split in expected_ref_splits:
        record = manifests.get(f"ref:{split}")
        expected = TABLE_A_REF_SPLIT_CONTRACT[split]
        if (
            not isinstance(record, Mapping)
            or _required_int(record.get("rows"), f"{split} manifest rows")
            != int(expected["rows"])
            or str(record.get("sha256")) != str(expected["sha256"])
        ):
            raise TableAEvaluationError(f"{split} generated manifest binding mismatch")
    tn_record = manifests.get("tn")
    tn_manifest = plan["tn_manifest"]
    if (
        not isinstance(tn_record, Mapping)
        or Path(str(tn_record.get("path", ""))).resolve(strict=True)
        != Path(str(tn_manifest["path"])).resolve(strict=True)
        or str(tn_record.get("sha256")) != str(tn_manifest["sha256"])
        or _required_int(tn_record.get("rows"), "TN rows")
        != int(tn_manifest["rows"])
    ):
        raise TableAEvaluationError("candidate TN causal input binding mismatch")
    counterfactual = summary.get("tn_counterfactual")
    groups = counterfactual.get("by_edit_taxonomy") if isinstance(counterfactual, Mapping) else None
    if not isinstance(groups, Mapping):
        raise TableAEvaluationError("candidate TN taxonomy summary is missing")
    expected_taxonomy_counts = plan["contract"].get("tn_taxonomy_counts")
    if not isinstance(expected_taxonomy_counts, Mapping) or set(groups) != set(
        expected_taxonomy_counts
    ):
        raise TableAEvaluationError("candidate TN taxonomy key set drifted")
    for taxonomy, expected_count in expected_taxonomy_counts.items():
        if int(groups[taxonomy].get("num_pairs", -1)) != int(expected_count):
            raise TableAEvaluationError(
                f"candidate TN taxonomy count drifted: {taxonomy}"
            )
    for taxonomy in REQUIRED_EDIT_TAXONOMIES:
        group = groups.get(taxonomy)
        if not isinstance(group, Mapping) or int(group.get("num_supported", 0)) <= 0:
            raise TableAEvaluationError(f"required edit taxonomy is absent: {taxonomy}")
        patch = group.get("surfaces", {}).get("patch")
        if not isinstance(patch, Mapping):
            raise TableAEvaluationError(f"{taxonomy} patch surface is missing")
        if (
            _finite_metric(
                patch.get("invariant_rate_at_1e-8"),
                f"{taxonomy}.patch_invariance",
                probability=True,
            )
            != 1.0
            or _finite_metric(
                patch.get("top1_change_rate"),
                f"{taxonomy}.patch_top1_change",
                probability=True,
            )
            != 0.0
        ):
            raise TableAEvaluationError(f"{taxonomy} violated fixed-canonical patch invariance")
    expected_surfaces = {
        "patch",
        "fulltext",
        "patch_admission_text_rank",
        role_eval.TRUE_ROLE_SWAP_ROUTE,
    }
    for group_name, group in {
        "overall": counterfactual.get("overall"),
        **dict(groups),
    }.items():
        if not isinstance(group, Mapping):
            raise TableAEvaluationError(f"candidate TN group {group_name} is invalid")
        if int(group.get("num_pairs", -1)) <= 0 or int(
            group.get("num_supported", -1)
        ) != int(group.get("num_pairs", -2)):
            raise TableAEvaluationError(
                f"candidate TN group {group_name} is incomplete/unsupported"
            )
        _finite_metric(
            group.get("candidate_admission_change_rate"),
            f"{group_name}.candidate_admission_change_rate",
            probability=True,
        )
        surfaces = group.get("surfaces")
        if not isinstance(surfaces, Mapping) or set(surfaces) != expected_surfaces:
            raise TableAEvaluationError(
                f"candidate TN group {group_name} lacks all formal score surfaces"
            )
        for surface_name, surface in surfaces.items():
            if not isinstance(surface, Mapping):
                raise TableAEvaluationError(
                    f"candidate TN surface {group_name}.{surface_name} is invalid"
                )
            for key in (
                "delta_max_logit_mean",
                "delta_max_logit_median",
                "mean_absolute_rank_change",
            ):
                _finite_metric(surface.get(key), f"{group_name}.{surface_name}.{key}")
            for key in ("top1_change_rate", "invariant_rate_at_1e-8"):
                _finite_metric(
                    surface.get(key),
                    f"{group_name}.{surface_name}.{key}",
                    probability=True,
                )
    category = summary.get("category_intervention")
    if not isinstance(category, Mapping):
        raise TableAEvaluationError("category intervention summary is missing")
    assets = category.get("asset_rehash")
    if (
        int(category.get("num_pairs", -1)) != 512
        or int(category.get("num_arms", -1)) != 1024
        or not isinstance(assets, Mapping)
        or assets.get("status") != "passed"
        or int(assets.get("images_rehashed", -1)) != 512
        or int(assets.get("supports_rehashed", -1)) != 318
        or category.get("category_causal_route")
        != "joint_canonical_prompt_plus_support_patch"
        or category.get("patch_only_category_causal_claim_eligible") is not False
        or category.get("intervention_contract", {}).get(
            "patch_only_attribution_forbidden"
        )
        is not True
    ):
        raise TableAEvaluationError("category intervention completeness/asset rehash failed")
    admission = category.get("candidate_admission")
    if not isinstance(admission, Mapping) or set(admission) != expected_oracle:
        raise TableAEvaluationError("category admission surface is incomplete")
    for topk, row in admission.items():
        if not isinstance(row, Mapping):
            raise TableAEvaluationError(f"category admission @{topk} is invalid")
        for key in (
            "matched_active_recall_iou50",
            "counterfactual_category_recall_iou50",
            "matched_active_mean_best_iou",
            "counterfactual_mean_best_iou",
        ):
            _finite_metric(row.get(key), f"category.{topk}.{key}", probability=True)
    for key in (
        "top1_both_match_active_rate",
        "top1_query_change_rate",
        "mean_top1_box_iou_between_arms",
    ):
        _finite_metric(category.get(key), f"category.{key}", probability=True)
    equality = summary.get("g5_tensor_equality_receipt")
    equality_count = (
        int(equality.get("forced_forward_count", 0))
        if isinstance(equality, Mapping)
        else 0
    )
    equality_fields = (
        "selector_no_grad_count",
        "hs_bitwise_equal_count",
        "boxes_bitwise_equal_count",
        "patch_logits_bitwise_equal_count",
        "candidate_order_bitwise_equal_count",
    )
    if (
        not isinstance(equality, Mapping)
        or equality.get("status") != "passed"
        or equality.get("comparison") != "torch.equal_bitwise"
        or equality.get("no_grad_required") is not True
        or equality_count <= 0
        or any(int(equality.get(key, -1)) != equality_count for key in equality_fields)
    ):
        raise TableAEvaluationError("candidate G5 equality receipt is incomplete")
    output_record = summary.get("outputs", {}).get("records")
    if (
        not isinstance(output_record, Mapping)
        or Path(str(output_record.get("path", ""))).resolve(strict=True) != records_path
        or str(output_record.get("sha256")) != cache.digest(records_path)
        or int(output_record.get("rows", -1)) != _line_count(records_path)
    ):
        raise TableAEvaluationError("candidate per-example record binding mismatch")
    expected_record_rows = (
        sum(int(TABLE_A_REF_SPLIT_CONTRACT[split]["rows"]) for split in expected_ref_splits)
        + 2 * int(tn_manifest["rows"])
        + 1024
    )
    if _line_count(records_path) != expected_record_rows:
        raise TableAEvaluationError("candidate per-example record count mismatch")
    record_validation = _validate_candidate_records(
        records_path,
        expected_ref={
            split: int(TABLE_A_REF_SPLIT_CONTRACT[split]["rows"])
            for split in expected_ref_splits
        },
        expected_tn=int(tn_manifest["rows"]),
    )
    grouped = _load_candidate_record_groups(records_path)
    for split in expected_ref_splits:
        recomputed = role_eval.aggregate_role_records(grouped[f"ref:{split}"])
        if recomputed != ref[split]:
            raise TableAEvaluationError(
                f"candidate Ref summary differs from record replay: {split}"
            )
    recomputed_tn_roles = role_eval.aggregate_role_records(grouped["tn_positive"])
    if recomputed_tn_roles != summary.get("tn_positive_roles"):
        raise TableAEvaluationError("candidate TN role summary differs from records")
    recomputed_counterfactual = role_eval.aggregate_counterfactual_records(
        grouped["tn_counterfactual"]
    )
    if recomputed_counterfactual != counterfactual:
        raise TableAEvaluationError(
            "candidate TN sensitivity summary differs from records"
        )
    recomputed_category = role_eval.aggregate_category_intervention_records(
        grouped["category_intervention"]
    )
    reported_category = dict(category)
    reported_category.pop("asset_rehash", None)
    if recomputed_category != reported_category:
        raise TableAEvaluationError(
            "candidate category summary differs from per-arm records"
        )
    screen_artifacts = None
    if profile == VALIDATION_PROFILE:
        from tools.stageb_screen_calibration import load_binding

        screen = summary.get("screen_calibration")
        if not isinstance(screen, Mapping):
            raise TableAEvaluationError("validation candidate has no screen binding")
        derived = Path(str(screen.get("screen_calibration_derived_path", ""))).resolve(
            strict=True
        )
        binding = load_binding(
            Path(str(screen.get("screen_calibration_binding_path", ""))),
            expected_derived=derived,
        )
        audit = plan["tn_inputs"]["screen_calibration_audit"]
        if (
            str(binding.source_manifest["sha256"]) != str(tn_manifest["sha256"])
            or str(binding.source_audit["sha256"]) != str(audit["sha256"])
            or int(binding.source_manifest["rows"]) != int(tn_manifest["rows"])
        ):
            raise TableAEvaluationError("validation screen derivation binding drifted")
        screen_artifacts = {
            "derived": _file_record(derived, cache, "screen_calibration_derived"),
            "binding": _file_record(
                binding.path, cache, "screen_calibration_binding"
            ),
        }
    elif summary.get("screen_calibration") is not None:
        raise TableAEvaluationError("final candidate contains validation calibration")
    return {
        "summary": _file_record(summary_path, cache, "table_a_candidate_summary"),
        "records": _file_record(records_path, cache, "table_a_candidate_records"),
        "ref_expressions": sum(
            int(TABLE_A_REF_SPLIT_CONTRACT[split]["rows"])
            for split in expected_ref_splits
        ),
        "tn_pairs": int(tn_manifest["rows"]),
        "category_pairs": 512,
        "record_validation": record_validation,
        "metric_replay": {
            "ref_splits": list(expected_ref_splits),
            "tn_roles": True,
            "tn_sensitivity": True,
            "category": True,
        },
        "g5_tensor_equality_receipt": dict(equality),
        "screen_calibration": screen_artifacts,
    }


def _summary_rows(path: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    value = _read_json(path, label=f"evaluation summary {path}")
    ref = value.get("refcoco")
    tn = value.get("tn")
    if not isinstance(ref, list) or not isinstance(tn, list):
        raise TableAEvaluationError("text evaluation summary sections are invalid")
    if not all(isinstance(row, Mapping) for row in ref + tn):
        raise TableAEvaluationError("text evaluation summary contains non-object rows")
    return ref, tn


def _resolve_records_path(value: Any, *, section_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TableAEvaluationError("summary row has no records_jsonl")
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        REPO_ROOT / raw,
        section_dir / raw,
    ]
    existing = {path.resolve(strict=True) for path in candidates if path.is_file()}
    if len(existing) != 1:
        raise TableAEvaluationError(
            f"records_jsonl must resolve to one file, found {len(existing)}"
        )
    path = next(iter(existing))
    try:
        path.relative_to(section_dir.resolve(strict=True))
    except ValueError as exc:
        raise TableAEvaluationError("records_jsonl escapes its fresh output section") from exc
    return path


def _tn_record_metrics(
    path: Path, source_manifest_path: Path, *, expected_run_id: str | None = None
) -> dict[str, Any]:
    from tools.compare_stageb_fpr95_records import (
        exact_binary_auroc,
        exact_fpr_at_tpr,
        load_manifest,
        load_tn_records,
    )

    manifest = load_manifest(source_manifest_path)
    loaded = load_tn_records(path, manifest, label="G0c Table-A TN records")
    if not bool(loaded.valid.all()):
        raise TableAEvaluationError("G0c TN records contain invalid pairs")
    if expected_run_id is not None and loaded.run_ids != (expected_run_id,):
        raise TableAEvaluationError("G0c TN record run_id differs from the summary")
    positive_array = loaded.positive
    negative_array = loaded.negative
    positive = positive_array.tolist()
    negative = negative_array.tolist()
    fpr90 = exact_fpr_at_tpr(positive, negative, target_tpr=0.90)
    fpr95 = exact_fpr_at_tpr(positive, negative, target_tpr=0.95)
    gaps = positive_array - negative_array
    return {
        "fpr90tpr": float(fpr90["fpr"]),
        "fpr95tpr": float(fpr95["fpr"]),
        "threshold_at_95tpr": float(fpr95["threshold"]),
        "actual_tpr_at_95tpr": float(fpr95["actual_tpr"]),
        "pair_win_rate": float(
            (positive_array > negative_array).mean()
        ),
        "pair_tie_rate": float(
            (positive_array == negative_array).mean()
        ),
        "pos_score_mean": float(positive_array.mean()),
        "tn_score_mean": float(negative_array.mean()),
        "score_gap_mean": float(gaps.mean()),
        "roc_auc": float(exact_binary_auroc(positive, negative)),
        "manifest_binding_mode": loaded.manifest_binding_mode,
    }


def _validate_tn_metric_replay(
    row: Mapping[str, Any], replay: Mapping[str, Any], *, label: str
) -> None:
    for key in G0C_TN_AGGREGATE_METRICS:
        reported = _finite_metric(row.get(key), f"G0c {label}.{key}")
        observed = _finite_metric(replay.get(key), f"G0c {label} replay.{key}")
        if not math.isclose(reported, observed, rel_tol=0.0, abs_tol=1e-12):
            raise TableAEvaluationError(
                f"G0c {label} summary {key} differs from records"
            )


def _verify_g0c_outputs(plan: Mapping[str, Any], cache: HashCache) -> dict[str, Any]:
    output_dir = Path(str(plan["output_dir"])).resolve(strict=True)
    profile = str(plan.get("profile", ""))
    if profile == VALIDATION_PROFILE:
        screen_contract = paper_eval._screen_calibration_contract(
            paper_eval.HashCache()
        )
        paper_plan = {
            "output_dir": str(output_dir),
            "evaluation_id": plan["evaluation_id"],
            "source": plan["source"],
            "protocol": {
                "profile": paper_eval.MATRIX_PROFILE,
                "screen_calibration": screen_contract,
            },
        }
        try:
            receipt = paper_eval._postflight_screen(paper_plan, {})
        except Exception as exc:
            raise TableAEvaluationError(
                f"G0c validation output revalidation failed: {exc}"
            ) from exc
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise TableAEvaluationError("G0c validation receipt has no artifacts")
        summary_path = (
            output_dir / "validation_calibration/summary.json"
        ).resolve(strict=True)
        ref_rows, tn_rows = _summary_rows(summary_path)
        by_split = {str(row.get("dataset", "")): row for row in ref_rows}
        if len(ref_rows) != len(VALIDATION_REF_SPLITS) or set(by_split) != set(
            VALIDATION_REF_SPLITS
        ) or len(tn_rows) != 1:
            raise TableAEvaluationError(
                "G0c validation summary surface is not exact"
            )
        replay = {}
        for split_index, split in enumerate(VALIDATION_REF_SPLITS):
            row = by_split[split]
            _validate_g0c_summary_provenance(
                row, plan=plan, expected_seed=EVAL_SEED + split_index * 100000
            )
            records_path = _resolve_records_path(
                row.get("records_jsonl"), section_dir=summary_path.parent
            )
            replay[split] = _replay_g0c_ref_records(
                records_path, row=row, split=split
            )
        _validate_g0c_summary_provenance(
            tn_rows[0], plan=plan, expected_seed=EVAL_SEED
        )
        validation_records = _resolve_records_path(
            tn_rows[0].get("records_jsonl"), section_dir=summary_path.parent
        )
        derived_manifest = Path(
            str(tn_rows[0].get("screen_calibration_derived_path", ""))
        ).expanduser().resolve(strict=True)
        try:
            derived_manifest.relative_to(summary_path.parent)
        except ValueError as exc:
            raise TableAEvaluationError(
                "G0c validation calibration manifest escapes its fresh output"
            ) from exc
        validation_tn_metrics = _tn_record_metrics(
            validation_records,
            derived_manifest,
            expected_run_id=str(tn_rows[0].get("run_id", "")),
        )
        _validate_tn_metric_replay(
            tn_rows[0], validation_tn_metrics, label="calibration"
        )
        return {
            "validation_calibration": dict(artifacts),
            "ref_splits": len(VALIDATION_REF_SPLITS),
            "tn_manifests": 1,
            "profile": profile,
            "ref_metric_replay": replay,
            "tn_metrics_recomputed": {
                "calibration": validation_tn_metrics,
            },
        }
    if profile != FINAL_PROFILE:
        raise TableAEvaluationError("G0c output profile is invalid")
    primary_path = (output_dir / "ref8_strict2031/summary.json").resolve(strict=True)
    supplemental_path = (output_dir / "strict1607/summary.json").resolve(strict=True)
    primary_ref, primary_tn = _summary_rows(primary_path)
    supplemental_ref, supplemental_tn = _summary_rows(supplemental_path)
    if supplemental_ref or len(primary_tn) != 1 or len(supplemental_tn) != 1:
        raise TableAEvaluationError("G0c primary/supplemental process split is invalid")
    by_split = {str(row.get("dataset")): row for row in primary_ref}
    if len(primary_ref) != len(paper_eval.REF_SPLITS) or set(by_split) != set(
        paper_eval.REF_SPLITS
    ):
        raise TableAEvaluationError("G0c summary does not cover REF8 exactly")
    metric_keys = (
        "acc50",
        "acc50@5",
        "acc50@10",
        "acc50@50",
        "recall50@all_queries",
        "mean_iou",
        "mean_iou@5",
        "mean_iou@10",
        "mean_iou@50",
        "mean_best_iou@all_queries",
    )
    record_artifacts: dict[str, Any] = {}
    tn_metrics: dict[str, Any] = {}
    ref_replay: dict[str, Any] = {}
    for split_index, split in enumerate(paper_eval.REF_SPLITS):
        row = by_split[split]
        expected = TABLE_A_REF_SPLIT_CONTRACT[split]
        if (
            int(row.get("manifest_n", -1)) != int(expected["rows"])
            or str(row.get("manifest_sha256")) != str(expected["sha256"])
            or int(row.get("invalid_records", -1)) != 0
            or int(row.get("max_batches", -1)) != 0
        ):
            raise TableAEvaluationError(f"G0c {split} manifest/completeness mismatch")
        _validate_g0c_summary_provenance(
            row, plan=plan, expected_seed=EVAL_SEED + split_index * 100000
        )
        for key in metric_keys:
            _finite_metric(row.get(key), f"G0c.{split}.{key}", probability=True)
        records_path = _resolve_records_path(
            row.get("records_jsonl"), section_dir=primary_path.parent
        )
        if _line_count(records_path) != int(expected["rows"]):
            raise TableAEvaluationError(f"G0c {split} record count mismatch")
        ref_replay[split] = _replay_g0c_ref_records(
            records_path, row=row, split=split
        )
        record_artifacts[split] = _file_record(
            records_path, cache, "g0c_ref_records"
        )
    for label, row in zip(("strict2031", "strict1607"), (primary_tn[0], supplemental_tn[0])):
        expected = (
            plan["tn_manifest"]
            if label == "strict2031"
            else plan["tn_inputs"]["strict1607"]
        )
        if (
            int(row.get("source_manifest_n", -1)) != int(expected["rows"])
            or str(row.get("source_manifest_sha256")) != str(expected["sha256"])
            or int(row.get("manifest_n", -1)) != int(expected["rows"])
            or int(row.get("invalid_records", -1)) != 0
            or int(row.get("max_batches", -1)) != 0
        ):
            raise TableAEvaluationError(f"G0c {label} manifest/completeness mismatch")
        _validate_g0c_summary_provenance(
            row, plan=plan, expected_seed=EVAL_SEED
        )
        for key in ("fpr95tpr", "fpr90tpr", "pair_win_rate"):
            _finite_metric(row.get(key), f"G0c.{label}.{key}", probability=True)
        section = primary_path.parent if label == "strict2031" else supplemental_path.parent
        records_path = _resolve_records_path(row.get("records_jsonl"), section_dir=section)
        if _line_count(records_path) != int(expected["rows"]):
            raise TableAEvaluationError(f"G0c {label} record count mismatch")
        record_artifacts[label] = _file_record(
            records_path, cache, "g0c_tn_records"
        )
        metrics = _tn_record_metrics(
            records_path,
            Path(str(expected["path"])).resolve(strict=True),
            expected_run_id=str(row.get("run_id", "")),
        )
        _validate_tn_metric_replay(row, metrics, label=label)
        tn_metrics[label] = metrics
    return {
        "ref8_strict2031_summary": _file_record(primary_path, cache, "g0c_summary"),
        "strict1607_summary": _file_record(supplemental_path, cache, "g0c_summary"),
        "ref_splits": len(primary_ref),
        "strict_manifests": 2,
        "records": record_artifacts,
        "ref_metric_replay": ref_replay,
        "tn_metrics_recomputed": tn_metrics,
    }


def postflight(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA or plan.get("status") not in {
        "running",
        "completed",
        "planned",
    }:
        raise TableAEvaluationError("Table-A launch manifest schema/status mismatch")
    _validate_instance(plan)
    _validate_command_surface(plan)
    _validate_g0c_launch_provenance(plan)
    training_provenance_replay = _replay_g0c_training_provenance(plan)
    profile = str(plan.get("profile", ""))
    if profile not in PROFILES:
        raise TableAEvaluationError("Table-A evaluation profile is invalid")
    if profile == FINAL_PROFILE:
        gate = plan.get("final_gate")
        if not isinstance(gate, Mapping):
            raise TableAEvaluationError("final Table-A evaluation has no gate binding")
        observed_gate = _validate_final_gate(
            Path(str(gate.get("path", ""))), plan["instance"]
        )
        if observed_gate["sha256"] != gate.get("sha256"):
            raise TableAEvaluationError("final Table-A gate SHA-256 drifted")
        final_consumption = _validate_final_consumption(plan)
    elif plan.get("final_gate") is not None:
        raise TableAEvaluationError("validation Table-A evaluation bound a final gate")
    else:
        final_consumption = None
    input_rehash = _verify_inputs(plan, hash_content=True)
    cache = HashCache()
    kind = str(plan.get("kind"))
    if kind == "candidate":
        artifacts = _verify_candidate_outputs(plan, cache)
    elif kind == "g0c":
        artifacts = _verify_g0c_outputs(plan, cache)
    else:
        raise TableAEvaluationError(f"unknown Table-A evaluation kind {kind!r}")
    result = {
        "schema": POSTFLIGHT_SCHEMA,
        "status": "passed",
        "kind": kind,
        "profile": profile,
        "evaluation_id": plan["evaluation_id"],
        "instance": dict(plan["instance"]),
        "verified_at_utc": _utc_now(),
        "input_rehash": input_rehash,
        "artifacts": artifacts,
        "final_consumption": final_consumption,
        "invariants": {
            "full_batches": True,
            "fixed_eval_seed_42": True,
            "checkpoint_and_config_sha_bound": True,
            "per_example_records_sha_bound": True,
            "g5_all_query_tensor_equality_fail_closed": kind == "candidate",
            "g2_g4_fixed_patch_candidate_tensor": kind == "candidate",
            "fixed_canonical_patch_invariance_exact": kind == "candidate",
            "category_prompt_support_asset_swap_sha_bound": kind == "candidate",
        },
    }
    if training_provenance_replay is not None:
        result["training_provenance_replay"] = training_provenance_replay
        result["invariants"]["g0c_training_provenance_freshly_replayed"] = True
    return result


def _subprocess_environment(runtime: Runtime) -> dict[str, str]:
    environment = dict(os.environ)
    environment["DATA_ROOT"] = str(runtime.data_root)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment.setdefault("OMP_NUM_THREADS", "1")
    pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not pythonpath
        else str(REPO_ROOT) + os.pathsep + pythonpath
    )
    return environment


def _stream(command: Sequence[str], runtime: Runtime, log_path: Path) -> int:
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


def execute(plan: dict[str, Any], runtime: Runtime) -> int:
    _validate_instance(plan)
    _validate_command_surface(plan)
    _validate_g0c_launch_provenance(plan)
    if plan.get("profile") == FINAL_PROFILE:
        plan["final_consumption"] = _consume_final_gate(plan)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    launch_path = output_dir / "launch_manifest.json"
    plan["status"] = "running"
    plan["started_at_utc"] = _utc_now()
    plan["completed_phases"] = []
    _write_json(launch_path, plan)
    try:
        for command_spec in plan["commands"]:
            _validate_instance(plan)
            _validate_command_surface(plan)
            _validate_g0c_launch_provenance(plan)
            plan["current_phase"] = command_spec["phase_id"]
            _write_json(launch_path, plan)
            _verify_inputs(plan, hash_content=False)
            returncode = _stream(
                command_spec["command"], runtime, Path(command_spec["console_log"])
            )
            if returncode:
                raise TableAEvaluationError(
                    f"phase {command_spec['phase_id']} exited with code {returncode}"
                )
            plan["completed_phases"].append(
                {"phase_id": command_spec["phase_id"], "returncode": 0}
            )
        result = postflight(plan)
        postflight_path = output_dir / "postflight.json"
        _write_json(postflight_path, result)
        plan["postflight"] = result
        plan["status"] = "completed"
        plan["current_phase"] = None
        plan["finished_at_utc"] = _utc_now()
        _write_json(launch_path, plan)
        print(json.dumps({"status": "completed", "output_dir": str(output_dir)}))
        return 0
    except BaseException as exc:
        plan["status"] = "failed"
        plan["failure_phase"] = plan.get("current_phase") or "postflight"
        plan["error"] = f"{type(exc).__name__}: {exc}"
        plan["finished_at_utc"] = _utc_now()
        _write_json(launch_path, plan)
        raise


def _load_launch(output_dir: Path) -> dict[str, Any]:
    path = (output_dir.expanduser().resolve(strict=True) / "launch_manifest.json").resolve(
        strict=True
    )
    value = dict(_read_json(path, label="Table-A launch manifest"))
    if value.get("schema") != SCHEMA:
        raise TableAEvaluationError("Table-A launch manifest schema mismatch")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("list", "dry-run", "run", "verify", "seal-final-gate")
    )
    parser.add_argument("--kind", choices=("candidate", "g0c"))
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--training-run-root")
    parser.add_argument(
        "--training-queue-dir",
        type=Path,
        help=(
            "completed serial queue attesting a formal token run outside the "
            "default canonical output root"
        ),
    )
    parser.add_argument("--g0c-training-plan")
    parser.add_argument("--final-gate", type=Path)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--python",
        default=os.environ.get(
            "PIVOT_PYTHON", "/home/haoyi/miniconda/envs/gdino5090/bin/python"
        ),
    )
    parser.add_argument(
        "--data-root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--plan-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "list":
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "kinds": ["candidate", "g0c"],
                    "profiles": list(PROFILES),
                    "eval_seed": EVAL_SEED,
                    "validation_ref_splits": list(VALIDATION_REF_SPLITS),
                    "final_ref_splits": list(paper_eval.REF_SPLITS),
                    "candidate_rows": ["G1", "G2", "G3", "G4", "G5"],
                    "required_edit_taxonomies": list(REQUIRED_EDIT_TAXONOMIES),
                    "category_pairs": 512,
                    "g0c_topks": [1, 5, 10, 50, "all"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "seal-final-gate":
        incompatible = any(
            value is not None
            for value in (
                args.kind,
                args.profile,
                args.training_run_root,
                args.training_queue_dir,
                args.g0c_training_plan,
                args.output_dir,
                args.final_gate,
                args.plan_json,
            )
        )
        if incompatible:
            parser.error("seal-final-gate accepts no evaluation/source/output arguments")
        gate = seal_final_gate()
        print(
            json.dumps(
                {
                    "status": "sealed",
                    "path": str(FINAL_GATE_PATH),
                    "gate_sha256": gate["gate_sha256"],
                    "instances": len(gate["instances"]),
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required")
    output_dir = Path(args.output_dir)
    if args.mode == "verify":
        plan = _load_launch(output_dir)
        result = postflight(plan)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.kind is None:
        parser.error("--kind is required for dry-run/run")
    if args.profile is None:
        parser.error("--profile is required for dry-run/run")
    if args.profile == FINAL_PROFILE and args.final_gate is None:
        parser.error("final profile requires --final-gate")
    if args.profile == VALIDATION_PROFILE and args.final_gate is not None:
        parser.error("validation profile cannot accept --final-gate")
    if args.kind == "candidate":
        if not args.training_run_root or args.g0c_training_plan:
            parser.error("candidate requires only --training-run-root")
    else:
        if not args.g0c_training_plan or args.training_run_root:
            parser.error("g0c requires --g0c-training-plan and no training root")
        if args.training_queue_dir is None:
            parser.error("g0c requires --training-queue-dir")
    runtime = _resolve_runtime(args)
    if args.kind == "candidate":
        plan = build_candidate_plan(
            runtime,
            Path(args.training_run_root),
            output_dir,
            profile=args.profile,
            training_queue_dir=args.training_queue_dir,
            final_gate=args.final_gate,
        )
    else:
        plan = build_g0c_plan(
            runtime,
            Path(args.g0c_training_plan),
            output_dir,
            profile=args.profile,
            training_queue_dir=args.training_queue_dir,
            final_gate=args.final_gate,
        )
    if args.plan_json:
        _write_json(Path(args.plan_json).expanduser().resolve(), plan)
    if args.mode == "dry-run":
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    return execute(plan, runtime)


if __name__ == "__main__":
    raise SystemExit(main())
