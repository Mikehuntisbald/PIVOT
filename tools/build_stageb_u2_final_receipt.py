#!/usr/bin/env python3
"""Build the fail-closed final receipt for the formal Stage-B U2 gap-3 run."""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_ref_split_contract import REF_SPLIT_CONTRACT, REF_SPLITS  # noqa: E402
from tools.select_stageb_u2_category_gate import (  # noqa: E402
    SelectionError,
    build_selection_receipt as replay_selection_receipt,
)
from tools.verify_stageb_dual_gate import (  # noqa: E402
    GateValidationError,
    evaluate_dual_gate,
    load_record_set_with_inputs,
)


SCHEMA = "pivot.stageb.u2_final_result_receipt/v1"
TRAINING_SCHEMA = "pivot.stageb.u2_training_receipt/v1"
SELECTION_SCHEMA = "pivot.stageb.u2_category_gate_selection/v2"
RECORD_GATE_SCHEMA = "stageb-dual-gate-v1"
SWEEP_CONTRACT = "stageb-u2-category-gate-sweep-lexicographic-v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "44e3d70b164eff2bcefacc37081b7cbab184a9373720ef69713d47949d449b90"
)
EXPECTED_GAP = 3.0
EXPECTED_TPR = 0.95
FORMAL_RUNTIME = {"batch_size": 16, "num_workers": 4, "max_batches": 0}
CANONICAL_SEEDS = {
    split: 42 + 100000 * index for index, split in enumerate(REF_SPLITS)
}
VAL_SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")
TEST_SPLITS = tuple(split for split in REF_SPLITS if split not in VAL_SPLITS)
STRICT_SIZES = (2031, 1607)
INPUT_ROLES = (
    "training_receipt",
    "selection_receipt",
    "config",
    "checkpoint",
    "val_sweep_summary",
    "final_val3_summary",
    "test5_summary",
    "strict2031_summary",
    "strict1607_summary",
    "strict2031_record_gate",
    "strict1607_record_gate",
    "baseline_ref8_summary",
    "baseline_strict2031_summary",
    "baseline_strict1607_summary",
)


class FinalReceiptError(ValueError):
    """An input does not prove the sealed final-result contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FinalReceiptError(f"value is not canonical JSON: {error}") from error
    return rendered.encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalReceiptError(f"{label}: cannot resolve {path}: {error}") from error
    if not resolved.is_file():
        raise FinalReceiptError(f"{label}: not a file: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise FinalReceiptError(f"{label}: file changed while hashing")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _assert_unchanged(record: Mapping[str, Any], *, label: str) -> None:
    observed = _file_record(Path(str(record["path"])), label=label)
    if observed != dict(record):
        raise FinalReceiptError(f"{label}: file changed before final sealing")


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FinalReceiptError(f"{label}: cannot parse JSON: {error}") from error
    return _mapping(value, label=label)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalReceiptError(f"{label}: expected an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FinalReceiptError(f"{label}: expected a list")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalReceiptError(f"{label}: expected an exact integer")
    return value


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReceiptError(f"{label}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FinalReceiptError(f"{label}: expected a finite number")
    return result


def _require_true(value: Any, *, label: str) -> None:
    if value is not True:
        raise FinalReceiptError(f"{label}: must be true")


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    reported = payload.get(field)
    if not isinstance(reported, str) or len(reported) != 64:
        raise FinalReceiptError(f"{label}.{field}: missing SHA-256")
    unhashed = dict(payload)
    del unhashed[field]
    observed = canonical_json_sha256(unhashed)
    if reported != observed:
        raise FinalReceiptError(
            f"{label}.{field}: self-hash mismatch ({reported} != {observed})"
        )
    return reported


def _assert_file_record(
    reported: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    value = _mapping(reported, label=label)
    if set(value) != {"path", "size_bytes", "sha256"}:
        raise FinalReceiptError(f"{label}: malformed file record")
    try:
        reported_path = Path(str(value["path"])).expanduser().resolve(strict=True)
    except OSError as error:
        raise FinalReceiptError(f"{label}: reported path cannot resolve: {error}") from error
    if (
        reported_path != Path(str(expected["path"]))
        or _integer(value["size_bytes"], label=f"{label}.size_bytes")
        != int(expected["size_bytes"])
        or value["sha256"] != expected["sha256"]
    ):
        raise FinalReceiptError(f"{label}: file identity mismatch")


def _resolve_reported_path(
    reported: Any, *, relative_to: Path, repo_root: Path, label: str
) -> Path:
    if not isinstance(reported, str) or not reported.strip():
        raise FinalReceiptError(f"{label}: missing path")
    raw = Path(reported).expanduser()
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, relative_to / raw]
    matches: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and resolved not in matches:
            matches.append(resolved)
    if len(matches) != 1:
        raise FinalReceiptError(
            f"{label}: path must resolve to exactly one file, found {matches}"
        )
    return matches[0]


def _validate_training(
    payload: Mapping[str, Any], *, checkpoint_record: Mapping[str, Any]
) -> None:
    if payload.get("schema") != TRAINING_SCHEMA:
        raise FinalReceiptError("training receipt: schema drifted")
    _verify_self_hash(payload, field="receipt_sha256", label="training receipt")
    checkpoint = _mapping(payload.get("checkpoint"), label="training checkpoint")
    _assert_file_record(
        checkpoint.get("file"), checkpoint_record, label="training checkpoint.file"
    )
    if _integer(
        checkpoint.get("optimizer_updates"), label="training optimizer_updates"
    ) != 100:
        raise FinalReceiptError("training receipt: optimizer_updates must be 100")
    args = _mapping(checkpoint.get("args"), label="training checkpoint.args")
    expected_args = {
        "batch_size": 56,
        "seed": 42,
        "max_train_iters": 100,
        "stage_b_u0_patch_rank": True,
        "stage_b_u2_category_complete_supervision": True,
        "stage_b_gdino_score_adapter": True,
    }
    for field, expected in expected_args.items():
        if args.get(field) != expected:
            raise FinalReceiptError(
                f"training checkpoint.args.{field}: expected {expected!r}"
            )
    lineage = _mapping(payload.get("lineage"), label="training lineage")
    _assert_file_record(
        lineage.get("durable_checkpoint"),
        checkpoint_record,
        label="training lineage.durable_checkpoint",
    )
    invariants = _mapping(payload.get("invariants"), label="training invariants")
    for field in (
        "formal_checkpoint_sha256_exact",
        "effective_args_equal_checkpoint_args",
        "frozen_tensor_hash_equal_initializer_to_u100",
        "merged_r100_p50_teacher_frozen",
        "shared_patch_backbone_frozen",
        "transition_audit_recomputed_equal",
        "category_complete_data_receipt_replayed",
    ):
        _require_true(invariants.get(field), label=f"training invariants.{field}")


def _validate_selection(
    payload: Mapping[str, Any],
    *,
    sweep_record: Mapping[str, Any],
    baseline_record: Mapping[str, Any],
    canonical_seeds: Mapping[str, int],
) -> str:
    if payload.get("schema") != SELECTION_SCHEMA:
        raise FinalReceiptError("selection receipt: schema must be v2")
    receipt_hash = _verify_self_hash(
        payload, field="payload_sha256", label="selection receipt"
    )
    _require_true(payload.get("selection_frozen"), label="selection_frozen")
    selection = _mapping(payload.get("selection"), label="selection")
    if _number(selection.get("max_gap"), label="selection.max_gap") != EXPECTED_GAP:
        raise FinalReceiptError("selection.max_gap: expected 3.0")
    contract = _mapping(payload.get("contract"), label="selection.contract")
    runtime = _mapping(
        contract.get("evaluation_runtime"), label="selection evaluation_runtime"
    )
    if dict(runtime) != FORMAL_RUNTIME:
        raise FinalReceiptError("selection receipt: runtime is not B16/W4/full")
    if tuple(contract.get("val_splits", ())) != VAL_SPLITS:
        raise FinalReceiptError("selection receipt: val split contract drifted")
    expected_val_seeds = {split: canonical_seeds[split] for split in VAL_SPLITS}
    if contract.get("canonical_seeds") != expected_val_seeds:
        raise FinalReceiptError("selection receipt: canonical val seeds drifted")
    if contract.get("sweep_contract") != SWEEP_CONTRACT:
        raise FinalReceiptError("selection receipt: sweep contract drifted")
    exact_gaps = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
    if contract.get("exact_gaps") != exact_gaps:
        raise FinalReceiptError("selection receipt: exact gap grid drifted")
    if _integer(
        contract.get("expected_query_count"),
        label="selection contract.expected_query_count",
    ) != 900:
        raise FinalReceiptError("selection receipt: expected query count must be 900")
    policy = _mapping(payload.get("selection_policy"), label="selection_policy")
    if payload.get("selection_policy_sha256") != canonical_json_sha256(policy):
        raise FinalReceiptError("selection receipt: policy hash mismatch")
    inputs = _mapping(payload.get("inputs"), label="selection inputs")
    _assert_file_record(
        inputs.get("sweep_summary"), sweep_record, label="selection sweep_summary"
    )
    _assert_file_record(
        inputs.get("baseline_summary"),
        baseline_record,
        label="selection baseline_summary",
    )
    return receipt_hash


def _resolved_static_config(
    path: Path, *, repo_root: Path, visiting: set[Path] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.resolve(strict=True)
    visiting = set() if visiting is None else set(visiting)
    if path in visiting:
        raise FinalReceiptError(f"config: cyclic import chain at {path}")
    visiting.add(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise FinalReceiptError(f"config: cannot parse Python source: {error}") from error
    values: dict[str, Any] = {}
    chain: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if (
                node.level != 0
                or not node.module
                or not node.module.startswith("config.")
                or [alias.name for alias in node.names] != ["*"]
            ):
                continue
            imported = repo_root / (node.module.replace(".", "/") + ".py")
            imported_values, imported_chain = _resolved_static_config(
                imported, repo_root=repo_root, visiting=visiting
            )
            values.update(imported_values)
            for record in imported_chain:
                if record not in chain:
                    chain.append(record)
            continue
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            # Unrelated derived configuration values are not needed for this
            # receipt. Required fields are checked for static values below.
            values.pop(target.id, None)
    record = _file_record(path, label=f"config import {path}")
    if record not in chain:
        chain.append(record)
    return values, chain


def _validate_config(
    path: Path,
    *,
    selection_path: Path,
    selection_payload_sha256: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    values, chain = _resolved_static_config(path, repo_root=repo_root)
    required = {
        "stage_b_u0_category_gate_max_gap",
        "stage_b_u0_category_gate_selection_receipt",
        "stage_b_u0_category_gate_selection_payload_sha256",
    }
    if not required <= set(values):
        raise FinalReceiptError(
            f"config: missing sealed constants {sorted(required - set(values))}"
        )
    _require_true(
        values.get("stage_b_u0_category_preserving_patch_gate"),
        label="resolved config stage_b_u0_category_preserving_patch_gate",
    )
    for field in (
        "stage_b_u0_patch_rank",
        "stage_b_gdino_score_adapter",
        "enable_patch_branch",
        "stage_b_u2_category_complete_supervision",
    ):
        _require_true(values.get(field), label=f"resolved config {field}")
    if _number(
        values["stage_b_u0_category_gate_max_gap"], label="config max_gap"
    ) != EXPECTED_GAP:
        raise FinalReceiptError("config: max_gap must equal selected gap 3.0")
    bound_receipt = _resolve_reported_path(
        values["stage_b_u0_category_gate_selection_receipt"],
        relative_to=path.parent,
        repo_root=repo_root,
        label="config selection receipt",
    )
    if bound_receipt != selection_path:
        raise FinalReceiptError("config: selection receipt path does not match input")
    if (
        values["stage_b_u0_category_gate_selection_payload_sha256"]
        != selection_payload_sha256
    ):
        raise FinalReceiptError("config: selection payload SHA-256 drifted")
    if values.get("stage_b_row_text_arbiter", False) is not False:
        raise FinalReceiptError("resolved config: stage_b_row_text_arbiter must be false/absent")
    if values.get("stage_b_gdino_adapter_merged_eval_only", False) is not False:
        raise FinalReceiptError(
            "resolved config: stage_b_gdino_adapter_merged_eval_only must be false/absent"
        )
    return chain


def _summary(payload: Mapping[str, Any], *, label: str) -> tuple[list[Any], list[Any]]:
    if set(payload) != {"refcoco", "tn"}:
        raise FinalReceiptError(f"{label}: expected exact refcoco/tn keys")
    return (
        _list(payload["refcoco"], label=f"{label}.refcoco"),
        _list(payload["tn"], label=f"{label}.tn"),
    )


def _validate_ref_row(
    row: Any,
    *,
    split: str,
    label: str,
    split_contract: Mapping[str, Mapping[str, Any]],
    canonical_seeds: Mapping[str, int],
    candidate: bool,
) -> tuple[float, Path]:
    value = _mapping(row, label=label)
    if value.get("dataset") != split:
        raise FinalReceiptError(f"{label}: dataset drifted")
    expected = split_contract[split]
    integer_fields = {
        "seed": canonical_seeds[split],
        **FORMAL_RUNTIME,
        "manifest_n": int(expected["rows"]),
        "num_expressions": int(expected["rows"]),
        "valid_mask_expressions": int(expected["rows"]),
        "invalid_records": 0,
        "invalid_mask_expressions": 0,
    }
    for field, wanted in integer_fields.items():
        if _integer(value.get(field), label=f"{label}.{field}") != wanted:
            raise FinalReceiptError(
                f"{label}.{field}: expected {wanted}; protocol is not B16/W4/full/canonical"
            )
    if value.get("manifest_sha256") != expected["sha256"]:
        raise FinalReceiptError(f"{label}: official manifest SHA-256 drifted")
    metric = _number(value.get("acc50"), label=f"{label}.acc50")
    if not 0.0 <= metric <= 1.0:
        raise FinalReceiptError(f"{label}.acc50: outside [0, 1]")
    if candidate:
        _require_true(value.get("amp"), label=f"{label}.amp")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise FinalReceiptError(f"{label}: run_id is missing")
    record_path = _resolve_reported_path(
        value.get("records_jsonl"),
        relative_to=Path(str(value.get("_summary_dir", REPO_ROOT))),
        repo_root=REPO_ROOT,
        label=f"{label}.records_jsonl",
    )
    return metric, record_path


def _bind_candidate(
    row: Mapping[str, Any],
    *,
    label: str,
    summary_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    config_path: Path | None,
    config_sha256: str | None,
    repo_root: Path,
) -> None:
    if row.get("checkpoint_sha256") != checkpoint_sha256:
        raise FinalReceiptError(f"{label}: checkpoint SHA-256 drifted")
    observed_checkpoint = _resolve_reported_path(
        row.get("checkpoint"),
        relative_to=summary_dir,
        repo_root=repo_root,
        label=f"{label}.checkpoint",
    )
    if observed_checkpoint != checkpoint_path:
        raise FinalReceiptError(f"{label}: checkpoint path drifted")
    if config_path is not None:
        if row.get("config_sha256") != config_sha256:
            raise FinalReceiptError(f"{label}: config SHA-256 drifted")
        observed_config = _resolve_reported_path(
            row.get("config"),
            relative_to=summary_dir,
            repo_root=repo_root,
            label=f"{label}.config",
        )
        if observed_config != config_path:
            raise FinalReceiptError(f"{label}: config path drifted")


def _with_summary_dir(row: Any, summary_path: Path) -> Mapping[str, Any]:
    value = dict(_mapping(row, label=f"{summary_path.name} row"))
    value["_summary_dir"] = str(summary_path.parent)
    return value


def _validate_ref_summaries(
    *,
    sweep: Mapping[str, Any],
    sweep_path: Path,
    final_val3: Mapping[str, Any],
    final_val3_path: Path,
    test5: Mapping[str, Any],
    test5_path: Path,
    baseline: Mapping[str, Any],
    baseline_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    config_path: Path,
    config_sha256: str,
    split_contract: Mapping[str, Mapping[str, Any]],
    canonical_seeds: Mapping[str, int],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Path], dict[str, Any]]:
    sweep_rows, sweep_tn = _summary(sweep, label="selection val sweep summary")
    final_val_rows, final_val_tn = _summary(
        final_val3, label="final gap3 val3 summary"
    )
    test_rows, test_tn = _summary(test5, label="gap3 test5 summary")
    baseline_rows, baseline_tn = _summary(baseline, label="baseline Ref8 summary")
    if sweep_tn or final_val_tn or test_tn or baseline_tn:
        raise FinalReceiptError("Ref summaries must contain no TN rows")

    exact_gaps = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
    sweep_by_key: dict[tuple[str, float], Mapping[str, Any]] = {}
    sweep_config_record: dict[str, Any] | None = None
    for index, raw in enumerate(sweep_rows):
        row = _with_summary_dir(raw, sweep_path)
        split = row.get("dataset")
        gap = _number(
            row.get("category_gate_max_gap"), label=f"val sweep row {index}.max_gap"
        )
        key = (str(split), gap)
        if split not in VAL_SPLITS or gap not in exact_gaps or key in sweep_by_key:
            raise FinalReceiptError("val sweep summary: duplicate/unexpected split-gap row")
        if row.get("category_gate_sweep_contract") != SWEEP_CONTRACT:
            raise FinalReceiptError("val sweep summary: sweep contract drifted")
        if _integer(
            row.get("category_gate_single_forward_gap_count"),
            label=f"val sweep row {index}.gap_count",
        ) != len(exact_gaps):
            raise FinalReceiptError("val sweep summary: incomplete single-forward grid")
        _validate_ref_row(
            row,
            split=str(split),
            label=f"val sweep {split} gap={gap:g}",
            split_contract=split_contract,
            canonical_seeds=canonical_seeds,
            candidate=True,
        )
        _bind_candidate(
            row,
            label=f"val sweep {split} gap={gap:g}",
            summary_dir=sweep_path.parent,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            config_path=None,
            config_sha256=None,
            repo_root=repo_root,
        )
        reported_config = _resolve_reported_path(
            row.get("config"),
            relative_to=sweep_path.parent,
            repo_root=repo_root,
            label=f"val sweep {split} gap={gap:g}.config",
        )
        if sweep_config_record is None:
            sweep_config_record = _file_record(
                reported_config, label="val sweep base config"
            )
        elif reported_config != Path(sweep_config_record["path"]):
            raise FinalReceiptError("val sweep summary: multiple base configs")
        if row.get("config_sha256") != sweep_config_record["sha256"]:
            raise FinalReceiptError("val sweep summary: base config SHA-256 drifted")
        sweep_by_key[key] = row
    expected_keys = {(split, gap) for split in VAL_SPLITS for gap in exact_gaps}
    if set(sweep_by_key) != expected_keys:
        raise FinalReceiptError("val sweep summary: not the exact 3x11 grid")

    final_val_by_split: dict[str, Mapping[str, Any]] = {}
    for raw in final_val_rows:
        row = _with_summary_dir(raw, final_val3_path)
        split = str(row.get("dataset", ""))
        if split not in VAL_SPLITS or split in final_val_by_split:
            raise FinalReceiptError("final gap3 val3 summary: duplicate/unexpected split")
        _validate_ref_row(
            row,
            split=split,
            label=f"final gap3 val3 {split}",
            split_contract=split_contract,
            canonical_seeds=canonical_seeds,
            candidate=True,
        )
        _bind_candidate(
            row,
            label=f"final gap3 val3 {split}",
            summary_dir=final_val3_path.parent,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            config_path=config_path,
            config_sha256=config_sha256,
            repo_root=repo_root,
        )
        final_val_by_split[split] = row
    if set(final_val_by_split) != set(VAL_SPLITS):
        raise FinalReceiptError("final gap3 val3 summary: expected exactly three val splits")

    test_by_split: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(test_rows):
        row = _with_summary_dir(raw, test5_path)
        split = str(row.get("dataset", ""))
        if split not in TEST_SPLITS or split in test_by_split:
            raise FinalReceiptError("gap3 test5 summary: duplicate/unexpected split")
        _validate_ref_row(
            row,
            split=split,
            label=f"gap3 test5 {split}",
            split_contract=split_contract,
            canonical_seeds=canonical_seeds,
            candidate=True,
        )
        _bind_candidate(
            row,
            label=f"gap3 test5 {split}",
            summary_dir=test5_path.parent,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            config_path=config_path,
            config_sha256=config_sha256,
            repo_root=repo_root,
        )
        test_by_split[split] = row
    if set(test_by_split) != set(TEST_SPLITS):
        raise FinalReceiptError("gap3 test5 summary: expected exactly five test splits")

    baseline_by_split: dict[str, Mapping[str, Any]] = {}
    for raw in baseline_rows:
        row = _with_summary_dir(raw, baseline_path)
        split = str(row.get("dataset", ""))
        if split not in REF_SPLITS or split in baseline_by_split:
            raise FinalReceiptError("baseline Ref8 summary: duplicate/unexpected split")
        _validate_ref_row(
            row,
            split=split,
            label=f"baseline Ref8 {split}",
            split_contract=split_contract,
            canonical_seeds=canonical_seeds,
            candidate=False,
        )
        baseline_by_split[split] = row
    if set(baseline_by_split) != set(REF_SPLITS):
        raise FinalReceiptError("baseline Ref8 summary: expected exactly eight splits")

    results: dict[str, Any] = {}
    candidate_records: dict[str, Path] = {}
    baseline_records: dict[str, Path] = {}
    for split in REF_SPLITS:
        candidate_row = (
            final_val_by_split[split]
            if split in VAL_SPLITS
            else test_by_split[split]
        )
        baseline_row = baseline_by_split[split]
        candidate_metric, candidate_record = _validate_ref_row(
            candidate_row,
            split=split,
            label=f"final candidate {split}",
            split_contract=split_contract,
            canonical_seeds=canonical_seeds,
            candidate=True,
        )
        baseline_metric, baseline_record = _validate_ref_row(
            baseline_row,
            split=split,
            label=f"final baseline {split}",
            split_contract=split_contract,
            canonical_seeds=canonical_seeds,
            candidate=False,
        )
        if not candidate_metric > baseline_metric:
            raise FinalReceiptError(
                f"Ref8 gate failed for {split}: {candidate_metric} <= {baseline_metric}"
            )
        results[split] = {
            "baseline_acc50": baseline_metric,
            "candidate_acc50": candidate_metric,
            "candidate_minus_baseline_acc50": candidate_metric - baseline_metric,
            "improved": True,
            "n": int(split_contract[split]["rows"]),
            "manifest_sha256": str(split_contract[split]["sha256"]),
        }
        candidate_records[split] = candidate_record
        baseline_records[split] = baseline_record
    assert sweep_config_record is not None
    return results, candidate_records, baseline_records, sweep_config_record


def _validate_strict_row(
    payload: Mapping[str, Any],
    *,
    summary_path: Path,
    label: str,
    expected_n: int,
    candidate: bool,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    config_path: Path,
    config_sha256: str,
    repo_root: Path,
) -> tuple[Mapping[str, Any], float, Path]:
    ref_rows, tn_rows = _summary(payload, label=label)
    if ref_rows or len(tn_rows) != 1:
        raise FinalReceiptError(f"{label}: expected zero Ref rows and exactly one TN row")
    row = _mapping(tn_rows[0], label=f"{label}.tn[0]")
    expected_integers = {
        "seed": 42,
        **FORMAL_RUNTIME,
        "manifest_n": expected_n,
        "num_pairs": expected_n,
        "invalid_records": 0,
    }
    for field, wanted in expected_integers.items():
        if _integer(row.get(field), label=f"{label}.{field}") != wanted:
            raise FinalReceiptError(
                f"{label}.{field}: expected {wanted}; protocol is not B16/W4/full"
            )
    if candidate:
        _require_true(row.get("amp"), label=f"{label}.amp")
        _bind_candidate(
            row,
            label=label,
            summary_dir=summary_path.parent,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            config_path=config_path,
            config_sha256=config_sha256,
            repo_root=repo_root,
        )
    fpr = _number(row.get("fpr95tpr"), label=f"{label}.fpr95tpr")
    if not 0.0 <= fpr <= 1.0:
        raise FinalReceiptError(f"{label}.fpr95tpr: outside [0, 1]")
    actual_tpr = _number(
        row.get("actual_tpr_at_95tpr"), label=f"{label}.actual_tpr_at_95tpr"
    )
    if actual_tpr < EXPECTED_TPR:
        raise FinalReceiptError(f"{label}: actual TPR is below 0.95")
    record_path = _resolve_reported_path(
        row.get("records_jsonl"),
        relative_to=summary_path.parent,
        repo_root=repo_root,
        label=f"{label}.records_jsonl",
    )
    return row, fpr, record_path


def _validate_nested_file_record(
    value: Any,
    *,
    label: str,
    cache: dict[Path, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    reported = _mapping(value, label=label)
    if set(reported) != {"path", "size_bytes", "sha256"}:
        raise FinalReceiptError(f"{label}: malformed nested file record")
    path = Path(str(reported.get("path"))).expanduser().resolve(strict=True)
    if path not in cache:
        cache[path] = _file_record(path, label=label)
    _assert_file_record(reported, cache[path], label=label)
    return path, cache[path]


def _validate_record_gate(
    report: Mapping[str, Any],
    *,
    label: str,
    expected_n: int,
    ref_results: Mapping[str, Mapping[str, Any]],
    candidate_ref_records: Mapping[str, Path],
    baseline_ref_records: Mapping[str, Path],
    candidate_tn_record: Path,
    baseline_tn_record: Path,
    candidate_tn_row: Mapping[str, Any],
    baseline_tn_row: Mapping[str, Any],
    nested_cache: dict[Path, dict[str, Any]],
) -> None:
    if report.get("schema") != RECORD_GATE_SCHEMA:
        raise FinalReceiptError(f"{label}: record gate schema drifted")
    if _number(report.get("target_tpr"), label=f"{label}.target_tpr") != EXPECTED_TPR:
        raise FinalReceiptError(f"{label}: target TPR drifted")
    required = _list(report.get("required_ref_splits"), label=f"{label}.required")
    evaluated = _list(report.get("evaluated_ref_splits"), label=f"{label}.evaluated")
    if len(required) != len(REF_SPLITS) or set(required) != set(REF_SPLITS):
        raise FinalReceiptError(f"{label}: required Ref split set drifted")
    if len(evaluated) != len(REF_SPLITS) or set(evaluated) != set(REF_SPLITS):
        raise FinalReceiptError(f"{label}: evaluated Ref split set drifted")
    gate = _mapping(report.get("gate"), label=f"{label}.gate")
    for field in (
        "pass",
        "every_required_ref_split_acc50_higher",
        "global_fpr95_lower",
        "bootstrap_ci_is_informational_not_a_gate",
    ):
        _require_true(gate.get(field), label=f"{label}.gate.{field}")
    validation = _mapping(report.get("validation"), label=f"{label}.validation")
    _require_true(validation.get("pass"), label=f"{label}.validation.pass")
    if validation.get("errors") != []:
        raise FinalReceiptError(f"{label}: validation errors must be empty")

    groups = _mapping(validation.get("groups"), label=f"{label}.validation.groups")
    expected_groups = {"tn_global", *(f"ref:{split}" for split in REF_SPLITS)}
    if set(groups) != expected_groups:
        raise FinalReceiptError(f"{label}: validation group set drifted")
    for group_name, raw_group in groups.items():
        group = _mapping(raw_group, label=f"{label}.{group_name}")
        expected_group_n = (
            expected_n
            if group_name == "tn_global"
            else int(ref_results[group_name.removeprefix("ref:")]["n"])
        )
        expected_group_manifest = (
            str(candidate_tn_row.get("manifest_sha256"))
            if group_name == "tn_global"
            else str(
                ref_results[group_name.removeprefix("ref:")]["manifest_sha256"]
            )
        )
        side_manifests: list[str] = []
        for side in ("baseline", "candidate"):
            side_value = _mapping(group.get(side), label=f"{label}.{group_name}.{side}")
            for field in ("exact_manifest_index_order",):
                _require_true(
                    side_value.get(field), label=f"{label}.{group_name}.{side}.{field}"
                )
            for field in ("duplicates", "invalid"):
                if _integer(
                    side_value.get(field), label=f"{label}.{group_name}.{side}.{field}"
                ) != 0:
                    raise FinalReceiptError(f"{label}.{group_name}: invalid/duplicate records")
            manifest_n = _integer(
                side_value.get("manifest_n"),
                label=f"{label}.{group_name}.{side}.manifest_n",
            )
            if manifest_n != expected_group_n:
                raise FinalReceiptError(f"{label}.{group_name}: manifest size drifted")
            if _integer(
                side_value.get("records_n"),
                label=f"{label}.{group_name}.{side}.records_n",
            ) != manifest_n:
                raise FinalReceiptError(f"{label}.{group_name}: incomplete records")
            manifest_sha = side_value.get("manifest_sha256")
            if manifest_sha != expected_group_manifest:
                raise FinalReceiptError(f"{label}.{group_name}: manifest hash drifted")
            side_manifests.append(manifest_sha)
        if side_manifests[0] != side_manifests[1]:
            raise FinalReceiptError(f"{label}.{group_name}: manifest hash mismatch")
        paired = _mapping(group.get("paired"), label=f"{label}.{group_name}.paired")
        for field in (
            "n_match",
            "manifest_hash_match",
            "sample_id_order_match",
            "image_id_order_match",
        ):
            _require_true(paired.get(field), label=f"{label}.{group_name}.{field}")
        if group_name != "tn_global":
            _require_true(
                paired.get("all_query_best_iou_exact_match"),
                label=f"{label}.{group_name}.all_query_best_iou_exact_match",
            )

    input_files = _mapping(report.get("input_files"), label=f"{label}.input_files")
    _require_true(
        input_files.get("identity_is_from_the_same_bytes_used_for_metrics"),
        label=f"{label}.input byte identity",
    )
    path_sets: dict[str, set[Path]] = {}
    for side in ("baseline", "candidate"):
        records = _list(input_files.get(side), label=f"{label}.input_files.{side}")
        if len(records) != len(REF_SPLITS) + 1:
            raise FinalReceiptError(f"{label}: {side} must contain exact Ref8+TN records")
        paths: set[Path] = set()
        for index, record in enumerate(records):
            path, _ = _validate_nested_file_record(
                record,
                label=f"{label}.input_files.{side}[{index}]",
                cache=nested_cache,
            )
            if path in paths:
                raise FinalReceiptError(f"{label}: duplicate {side} input record")
            paths.add(path)
        path_sets[side] = paths
    expected_candidate_refs = set(candidate_ref_records.values())
    expected_baseline_refs = set(baseline_ref_records.values())
    if not expected_candidate_refs < path_sets["candidate"]:
        raise FinalReceiptError(f"{label}: candidate Ref8 record paths drifted")
    if not expected_baseline_refs < path_sets["baseline"]:
        raise FinalReceiptError(f"{label}: baseline Ref8 record paths drifted")
    candidate_extra = path_sets["candidate"] - expected_candidate_refs
    if candidate_extra != {candidate_tn_record}:
        raise FinalReceiptError(f"{label}: candidate TN record path drifted")
    baseline_extra = path_sets["baseline"] - expected_baseline_refs
    if baseline_extra != {baseline_tn_record}:
        raise FinalReceiptError(f"{label}: baseline TN record path drifted")

    ref_report = _mapping(report.get("refcoco"), label=f"{label}.refcoco")
    if set(ref_report) != set(REF_SPLITS):
        raise FinalReceiptError(f"{label}: Ref8 metric set drifted")
    for split in REF_SPLITS:
        observed = _mapping(ref_report[split], label=f"{label}.{split}")
        expected = ref_results[split]
        _require_true(observed.get("improved"), label=f"{label}.{split}.improved")
        if (
            _number(observed.get("baseline_acc50"), label=f"{label}.{split}.baseline")
            != expected["baseline_acc50"]
            or _number(
                observed.get("candidate_acc50"), label=f"{label}.{split}.candidate"
            )
            != expected["candidate_acc50"]
            or _integer(observed.get("n"), label=f"{label}.{split}.n")
            != expected["n"]
        ):
            raise FinalReceiptError(f"{label}: {split} metrics do not bind summaries")

    tn_report = _mapping(report.get("tn_global"), label=f"{label}.tn_global")
    _require_true(tn_report.get("improved"), label=f"{label}.tn_global.improved")
    if _integer(tn_report.get("n"), label=f"{label}.tn_global.n") != expected_n:
        raise FinalReceiptError(f"{label}: TN size drifted")
    baseline_result = _mapping(
        tn_report.get("baseline"), label=f"{label}.tn_global.baseline"
    )
    candidate_result = _mapping(
        tn_report.get("candidate"), label=f"{label}.tn_global.candidate"
    )
    comparisons = (
        (baseline_result, baseline_tn_row, "baseline"),
        (candidate_result, candidate_tn_row, "candidate"),
    )
    for report_row, summary_row, side in comparisons:
        if (
            _number(report_row.get("fpr"), label=f"{label}.{side}.fpr")
            != _number(summary_row.get("fpr95tpr"), label=f"{label}.{side}.summary_fpr")
            or _number(report_row.get("actual_tpr"), label=f"{label}.{side}.actual_tpr")
            != _number(
                summary_row.get("actual_tpr_at_95tpr"),
                label=f"{label}.{side}.summary_tpr",
            )
        ):
            raise FinalReceiptError(f"{label}: {side} TN metrics do not bind summary")


def _rebuild_record_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    input_files = _mapping(report.get("input_files"), label="record gate input_files")
    paths: dict[str, list[str]] = {}
    for side in ("baseline", "candidate"):
        records = _list(input_files.get(side), label=f"record gate {side} inputs")
        paths[side] = [
            str(_mapping(record, label=f"record gate {side} record")["path"])
            for record in records
        ]
    bootstrap = _mapping(
        _mapping(report.get("tn_global"), label="record gate tn_global").get(
            "bootstrap"
        ),
        label="record gate bootstrap",
    )
    iterations = _integer(bootstrap.get("iterations"), label="bootstrap iterations")
    confidence = _number(bootstrap.get("confidence"), label="bootstrap confidence")
    if iterations != 5000 or confidence != 0.95:
        raise FinalReceiptError("record gate: expected canonical 5000/.95 bootstrap")
    try:
        baseline, baseline_inputs = load_record_set_with_inputs(paths["baseline"])
        candidate, candidate_inputs = load_record_set_with_inputs(paths["candidate"])
        replayed = evaluate_dual_gate(
            baseline,
            candidate,
            required_ref_splits=_list(
                report.get("required_ref_splits"), label="record gate required splits"
            ),
            target_tpr=EXPECTED_TPR,
            bootstrap_iterations=iterations,
            confidence=confidence,
            seed=20260711,
        )
    except (OSError, GateValidationError, ValueError) as error:
        raise FinalReceiptError(f"record gate: exact replay failed: {error}") from error
    replayed["input_files"] = {
        "baseline": baseline_inputs,
        "candidate": candidate_inputs,
        "identity_is_from_the_same_bytes_used_for_metrics": True,
    }
    return replayed


def build_final_receipt_payload(
    *,
    training_receipt: Path,
    selection_receipt: Path,
    config: Path,
    checkpoint: Path,
    val_sweep_summary: Path,
    final_val3_summary: Path,
    test5_summary: Path,
    strict2031_summary: Path,
    strict1607_summary: Path,
    strict2031_record_gate: Path,
    strict1607_record_gate: Path,
    baseline_ref8_summary: Path,
    baseline_strict2031_summary: Path,
    baseline_strict1607_summary: Path,
    expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    split_contract: Mapping[str, Mapping[str, Any]] = REF_SPLIT_CONTRACT,
    canonical_seeds: Mapping[str, int] = CANONICAL_SEEDS,
    repo_root: Path = REPO_ROOT,
    _selection_rebuilder: Any = replay_selection_receipt,
    _selection_expected_query_count: int = 900,
    _record_gate_rebuilder: Any = _rebuild_record_gate,
) -> dict[str, Any]:
    raw_paths = {
        "training_receipt": Path(training_receipt),
        "selection_receipt": Path(selection_receipt),
        "config": Path(config),
        "checkpoint": Path(checkpoint),
        "val_sweep_summary": Path(val_sweep_summary),
        "final_val3_summary": Path(final_val3_summary),
        "test5_summary": Path(test5_summary),
        "strict2031_summary": Path(strict2031_summary),
        "strict1607_summary": Path(strict1607_summary),
        "strict2031_record_gate": Path(strict2031_record_gate),
        "strict1607_record_gate": Path(strict1607_record_gate),
        "baseline_ref8_summary": Path(baseline_ref8_summary),
        "baseline_strict2031_summary": Path(baseline_strict2031_summary),
        "baseline_strict1607_summary": Path(baseline_strict1607_summary),
    }
    input_records = {
        role: _file_record(path, label=role) for role, path in raw_paths.items()
    }
    resolved_paths = {
        role: Path(record["path"]) for role, record in input_records.items()
    }
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise FinalReceiptError("direct input paths must be distinct")
    if input_records["checkpoint"]["sha256"] != expected_checkpoint_sha256:
        raise FinalReceiptError(
            "checkpoint SHA-256 drifted: expected "
            f"{expected_checkpoint_sha256}, got {input_records['checkpoint']['sha256']}"
        )
    json_roles = tuple(role for role in INPUT_ROLES if role not in {"config", "checkpoint"})
    payloads = {
        role: _strict_json(resolved_paths[role], label=role) for role in json_roles
    }

    _validate_training(
        payloads["training_receipt"],
        checkpoint_record=input_records["checkpoint"],
    )
    selection_hash = _validate_selection(
        payloads["selection_receipt"],
        sweep_record=input_records["val_sweep_summary"],
        baseline_record=input_records["baseline_ref8_summary"],
        canonical_seeds=canonical_seeds,
    )
    try:
        replayed_selection = _selection_rebuilder(
            sweep_summary=resolved_paths["val_sweep_summary"],
            baseline_summary=resolved_paths["baseline_ref8_summary"],
            split_contract=split_contract,
            baseline_splits=REF_SPLITS,
            canonical_seeds=canonical_seeds,
            expected_query_count=_selection_expected_query_count,
        )
    except (OSError, SelectionError, ValueError) as error:
        raise FinalReceiptError(f"selection receipt: v2 replay failed: {error}") from error
    if dict(replayed_selection) != dict(payloads["selection_receipt"]):
        raise FinalReceiptError(
            "selection receipt: input is not exactly reproducible by the v2 selector"
        )
    config_chain = _validate_config(
        resolved_paths["config"],
        selection_path=resolved_paths["selection_receipt"],
        selection_payload_sha256=selection_hash,
        repo_root=repo_root,
    )
    if not config_chain or config_chain[-1] != input_records["config"]:
        raise FinalReceiptError("final config import chain does not bind direct config bytes")

    ref_results, candidate_ref_records, baseline_ref_records, sweep_config = (
        _validate_ref_summaries(
            sweep=payloads["val_sweep_summary"],
            sweep_path=resolved_paths["val_sweep_summary"],
            final_val3=payloads["final_val3_summary"],
            final_val3_path=resolved_paths["final_val3_summary"],
            test5=payloads["test5_summary"],
            test5_path=resolved_paths["test5_summary"],
            baseline=payloads["baseline_ref8_summary"],
            baseline_path=resolved_paths["baseline_ref8_summary"],
            checkpoint_path=resolved_paths["checkpoint"],
            checkpoint_sha256=expected_checkpoint_sha256,
            config_path=resolved_paths["config"],
            config_sha256=input_records["config"]["sha256"],
            split_contract=split_contract,
            canonical_seeds=canonical_seeds,
            repo_root=repo_root,
        )
    )

    strict_results: dict[str, Any] = {}
    strict_rows: dict[
        int, tuple[Mapping[str, Any], Mapping[str, Any], Path, Path]
    ] = {}
    for expected_n in STRICT_SIZES:
        suffix = str(expected_n)
        candidate_row, candidate_fpr, candidate_record = _validate_strict_row(
            payloads[f"strict{suffix}_summary"],
            summary_path=resolved_paths[f"strict{suffix}_summary"],
            label=f"candidate strict{suffix}",
            expected_n=expected_n,
            candidate=True,
            checkpoint_path=resolved_paths["checkpoint"],
            checkpoint_sha256=expected_checkpoint_sha256,
            config_path=resolved_paths["config"],
            config_sha256=input_records["config"]["sha256"],
            repo_root=repo_root,
        )
        baseline_row, baseline_fpr, baseline_record = _validate_strict_row(
            payloads[f"baseline_strict{suffix}_summary"],
            summary_path=resolved_paths[f"baseline_strict{suffix}_summary"],
            label=f"baseline strict{suffix}",
            expected_n=expected_n,
            candidate=False,
            checkpoint_path=resolved_paths["checkpoint"],
            checkpoint_sha256=expected_checkpoint_sha256,
            config_path=resolved_paths["config"],
            config_sha256=input_records["config"]["sha256"],
            repo_root=repo_root,
        )
        if candidate_row.get("manifest_sha256") != baseline_row.get("manifest_sha256"):
            raise FinalReceiptError(f"strict{suffix}: candidate/baseline manifest mismatch")
        if not candidate_fpr < baseline_fpr:
            raise FinalReceiptError(
                f"strict{suffix} FPR95 gate failed: {candidate_fpr} >= {baseline_fpr}"
            )
        strict_results[f"strict{suffix}"] = {
            "n": expected_n,
            "baseline_fpr95": baseline_fpr,
            "candidate_fpr95": candidate_fpr,
            "candidate_minus_baseline_fpr95": candidate_fpr - baseline_fpr,
            "improved": True,
        }
        strict_rows[expected_n] = (
            candidate_row,
            baseline_row,
            candidate_record,
            baseline_record,
        )

    nested_cache: dict[Path, dict[str, Any]] = {}
    for expected_n in STRICT_SIZES:
        suffix = str(expected_n)
        candidate_row, baseline_row, candidate_record, baseline_record = strict_rows[
            expected_n
        ]
        _validate_record_gate(
            payloads[f"strict{suffix}_record_gate"],
            label=f"strict{suffix} record gate",
            expected_n=expected_n,
            ref_results=ref_results,
            candidate_ref_records=candidate_ref_records,
            baseline_ref_records=baseline_ref_records,
            candidate_tn_record=candidate_record,
            baseline_tn_record=baseline_record,
            candidate_tn_row=candidate_row,
            baseline_tn_row=baseline_row,
            nested_cache=nested_cache,
        )
        try:
            replayed_gate = _record_gate_rebuilder(
                payloads[f"strict{suffix}_record_gate"]
            )
        except FinalReceiptError:
            raise
        except Exception as error:
            raise FinalReceiptError(
                f"strict{suffix} record gate: exact replay failed: {error}"
            ) from error
        if dict(replayed_gate) != dict(payloads[f"strict{suffix}_record_gate"]):
            raise FinalReceiptError(
                f"strict{suffix} record gate: report is not exactly reproducible"
            )

    nested_cache[Path(sweep_config["path"])] = sweep_config
    for record in config_chain:
        nested_cache[Path(record["path"])] = record
    for role, record in input_records.items():
        _assert_unchanged(record, label=f"final input {role}")
    for index, record in enumerate(nested_cache.values()):
        _assert_unchanged(record, label=f"record-gate dependency {index}")

    body = {
        "schema": SCHEMA,
        "status": "pass",
        "checkpoint": input_records["checkpoint"],
        "selection": {
            "schema": SELECTION_SCHEMA,
            "selected_max_gap": EXPECTED_GAP,
            "payload_sha256": selection_hash,
            "selection_frozen": True,
        },
        "protocol": {
            "evaluation_runtime": FORMAL_RUNTIME,
            "canonical_ref_seeds": {
                split: canonical_seeds[split] for split in REF_SPLITS
            },
            "ref_splits": list(REF_SPLITS),
            "full_official_manifests": True,
            "target_tpr": EXPECTED_TPR,
        },
        "results": {
            "ref8_acc50": ref_results,
            "strict_fpr95": strict_results,
        },
        "record_gates": {
            "strict2031": {"validation_pass": True, "gate_pass": True},
            "strict1607": {"validation_pass": True, "gate_pass": True},
        },
        "inputs": input_records,
        "verified_nested_inputs": {
            "sweep_config": sweep_config,
            "final_config_import_chain": config_chain,
            "record_gate_input_files": [
                nested_cache[path] for path in sorted(nested_cache, key=str)
                if path != Path(sweep_config["path"])
                and path not in {Path(record["path"]) for record in config_chain}
            ],
        },
        "invariants": {
            "single_checkpoint_sha256_exact": True,
            "trained_u2_receipt_self_hash_valid": True,
            "selection_v2_frozen_at_gap3": True,
            "gap3_config_binds_selection_payload": True,
            "resolved_config_enables_only_the_gap3_category_gate": True,
            "b16_w4_full_canonical_protocol": True,
            "all_eight_ref_acc50_strictly_higher": True,
            "both_strict_fpr95_strictly_lower": True,
            "both_record_validation_gates_pass": True,
            "all_direct_and_record_gate_inputs_hash_verified": True,
        },
    }
    receipt = dict(body)
    receipt["payload_sha256"] = canonical_json_sha256(body)
    return receipt


def render_markdown(receipt: Mapping[str, Any]) -> str:
    results = _mapping(receipt["results"], label="receipt.results")
    ref = _mapping(results["ref8_acc50"], label="receipt Ref8")
    strict = _mapping(results["strict_fpr95"], label="receipt strict")
    lines = [
        "# Stage-B U2 Final Result",
        "",
        "Status: PASS",
        "",
        f"Checkpoint SHA-256: `{receipt['checkpoint']['sha256']}`",
        f"Selected category-gate max gap: `{EXPECTED_GAP:g}`",
        "Protocol: B16, W4, full official manifests, canonical split seeds.",
        "",
        "## RefCOCO Acc50",
        "",
        "| Split | GDINO Stage-B data-ft | U2 gap3 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for split in REF_SPLITS:
        row = ref[split]
        lines.append(
            f"| {split} | {row['baseline_acc50']:.6f} | "
            f"{row['candidate_acc50']:.6f} | "
            f"{row['candidate_minus_baseline_acc50']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Strict FPR95",
            "",
            "| Set | GDINO Stage-B data-ft | U2 | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("strict2031", "strict1607"):
        row = strict[name]
        lines.append(
            f"| {name} | {row['baseline_fpr95']:.6f} | "
            f"{row['candidate_fpr95']:.6f} | "
            f"{row['candidate_minus_baseline_fpr95']:+.6f} |"
        )
    lines.extend(["", f"Payload SHA-256: `{receipt['payload_sha256']}`", ""])
    return "\n".join(lines)


def _publish_identical_or_new(path: Path, content: bytes, *, label: str) -> str:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_file() or output.read_bytes() != content:
            raise FinalReceiptError(f"{label}: refusing to overwrite different file: {output}")
        return "already_identical"
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if not _rename_noreplace(temporary, output):
            if not output.is_file() or output.read_bytes() != content:
                raise FinalReceiptError(
                    f"{label}: refusing to overwrite different file: {output}"
                )
            return "already_identical"
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return "created"
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> bool:
    """Atomically publish source, returning false when destination exists."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return True
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            return False
        if error_number not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(
                error_number,
                os.strerror(error_number),
                str(destination),
            )
    try:
        os.link(source, destination)
    except FileExistsError:
        return False
    return True


def build_and_publish(
    *, output_json: Path, output_md: Path | None = None, **inputs: Any
) -> tuple[dict[str, Any], dict[str, str]]:
    receipt = build_final_receipt_payload(**inputs)
    json_bytes = (
        json.dumps(
            receipt,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    markdown_bytes = render_markdown(receipt).encode("ascii")
    outputs = {"json": Path(output_json).expanduser().resolve()}
    if output_md is not None:
        outputs["markdown"] = Path(output_md).expanduser().resolve()
    input_paths = {Path(record["path"]) for record in receipt["inputs"].values()}
    if any(path in input_paths for path in outputs.values()):
        raise FinalReceiptError("output path collides with an input artifact")
    # Check all destinations before publishing either output.
    for name, path in outputs.items():
        expected = json_bytes if name == "json" else markdown_bytes
        if path.exists() and (not path.is_file() or path.read_bytes() != expected):
            raise FinalReceiptError(
                f"{name}: refusing to overwrite different file: {path}"
            )
    statuses = {
        "json": _publish_identical_or_new(outputs["json"], json_bytes, label="json")
    }
    if "markdown" in outputs:
        statuses["markdown"] = _publish_identical_or_new(
            outputs["markdown"], markdown_bytes, label="markdown"
        )
    return receipt, statuses


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in INPUT_ROLES:
        parser.add_argument("--" + role.replace("_", "-"), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = {role: Path(getattr(args, role)) for role in INPUT_ROLES}
    try:
        receipt, statuses = build_and_publish(
            output_json=Path(args.output_json),
            output_md=Path(args.output_md) if args.output_md else None,
            **inputs,
        )
    except (OSError, FinalReceiptError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "pass",
                "publish": statuses,
                "payload_sha256": receipt["payload_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
