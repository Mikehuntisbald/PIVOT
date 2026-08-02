#!/usr/bin/env python3
"""Select the preregistered new-head LR from three sealed dev summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


RECEIPT_SCHEMA = "pivot.stageb.data_driven.new_head_lr_selection_receipt/v1"
PREREGISTRATION_SCHEMA = (
    "pivot.stageb.data_driven.new_head_lr_preregistration/v1"
)
EVALUATION_SUMMARY_SCHEMA = (
    "pivot.stageb.data_driven.new_head_dev_evaluation/v1"
)
EVALUATION_SCOPE = "new_head_only_common_frozen_b58_not_end_to_end_unseen_v1"
EVALUATION_VARIANT = "d1_category_complete"
EVALUATION_MANIFEST_VARIANT = "d0_ordinary_primary"
EXECUTION_SCOPE = "fresh_a0_new_head_lr_probe_u1000_v1"
EXECUTION_CONTRACT = "diagnostic_lr_probe_u1000_v1"
CANDIDATE_RANK_LRS = (3e-5, 1e-4, 3e-4)
OPTIMIZER_UPDATES_PER_CANDIDATE = 1000
SELECTION_PARTITION = "dev_screen"
SELECTION_METRIC = "macro_ref3_acc50"
SECONDARY_SELECTION_METRIC = "macro_ref3_mean_listwise_nll"
TIE_BREAK_RULE = (
    "max_macro_ref3_acc50",
    "min_macro_ref3_mean_listwise_nll",
    "min_rank_lr",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class NewHeadLRSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    rank_lr: float
    summary_file: dict[str, Any]
    checkpoint_file: dict[str, Any]
    config_file: dict[str, Any]
    metrics: dict[str, float]
    protocol: dict[str, Any]
    evaluation_inputs: dict[str, Any]
    initializer_provenance: dict[str, Any]
    shared_training_provenance: dict[str, Any]
    paired_training_contract_without_rank_lr: dict[str, Any]
    evaluation_config_contract_without_rank_lr: dict[str, Any]


def _reject_json_constant(value: str) -> None:
    raise NewHeadLRSelectionError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NewHeadLRSelectionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise NewHeadLRSelectionError("value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    try:
        path = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise NewHeadLRSelectionError(f"could not resolve file: {path}") from error
    if not path.is_file():
        raise NewHeadLRSelectionError(f"not a regular file: {path}")
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
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
        raise NewHeadLRSelectionError(f"file changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    try:
        value = json.loads(
            Path(record["path"]).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except NewHeadLRSelectionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NewHeadLRSelectionError(
            f"could not parse {label}: {record['path']}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NewHeadLRSelectionError(f"{label} must be a JSON object")
    if _file_record(Path(record["path"])) != record:
        raise NewHeadLRSelectionError(f"{label} changed while it was read")
    return value, record


def _require_canonical_payload(value: Mapping[str, Any], *, label: str) -> str:
    digest = value.get("canonical_payload_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise NewHeadLRSelectionError(f"{label} has no lowercase canonical hash")
    payload = dict(value)
    del payload["canonical_payload_sha256"]
    if _sha256_bytes(_canonical_bytes(payload)) != digest:
        raise NewHeadLRSelectionError(f"{label} canonical hash drifted")
    return digest


def _validate_bound_file_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise NewHeadLRSelectionError(f"{label} file record drifted")
    observed = _file_record(Path(str(value.get("path", ""))))
    if dict(value) != observed:
        raise NewHeadLRSelectionError(f"{label} file identity drifted")
    return observed


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NewHeadLRSelectionError(f"{label} must be a mapping")
    result = dict(value)
    _canonical_bytes(result)
    return result


def _require_metric(value: Any, *, label: str, unit_interval: bool) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise NewHeadLRSelectionError(f"{label} must be one finite JSON float")
    if unit_interval and not 0.0 <= value <= 1.0:
        raise NewHeadLRSelectionError(f"{label} must be in [0, 1]")
    if not unit_interval and value < 0.0:
        raise NewHeadLRSelectionError(f"{label} must be non-negative")
    return float(value)


def _without_rank_lr(value: Any, *, label: str) -> dict[str, Any]:
    contract = _require_mapping(value, label=label)
    if "stage_b_data_driven_rank_lr" not in contract:
        raise NewHeadLRSelectionError(f"{label} has no rank LR")
    del contract["stage_b_data_driven_rank_lr"]
    return contract


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value, record = _load_json(path, label="LR preregistration")
    _require_canonical_payload(value, label="LR preregistration")
    invariants = value.get("invariants")
    if not isinstance(invariants, Mapping) or not invariants or any(
        item is not True for item in invariants.values()
    ):
        raise NewHeadLRSelectionError(
            "LR preregistration invariants are not all exactly true"
        )
    expected = {
        "schema": PREREGISTRATION_SCHEMA,
        "status": "preregistered",
        "candidate_rank_lrs": list(CANDIDATE_RANK_LRS),
        "optimizer_updates_per_candidate": OPTIMIZER_UPDATES_PER_CANDIDATE,
        "selection_partition": SELECTION_PARTITION,
        "selection_metric": SELECTION_METRIC,
        "secondary_selection_metric": SECONDARY_SELECTION_METRIC,
    }
    drift = {
        key: (value.get(key), wanted)
        for key, wanted in expected.items()
        if value.get(key) != wanted
    }
    if drift:
        raise NewHeadLRSelectionError(
            f"LR preregistration fixed contract drifted: {drift}"
        )
    if value.get("tie_break_rule") != list(TIE_BREAK_RULE):
        raise NewHeadLRSelectionError("LR preregistration selection rule drifted")
    return value, record


def _validate_summary(path: Path, *, rank_lr: float) -> CandidateSummary:
    summary, summary_file = _load_json(path, label=f"LR={rank_lr} summary")
    _require_canonical_payload(summary, label=f"LR={rank_lr} summary")
    required_top = {
        "schema": EVALUATION_SUMMARY_SCHEMA,
        "evaluation_scope": EVALUATION_SCOPE,
        "variant": EVALUATION_VARIANT,
        "evaluation_manifest_variant": EVALUATION_MANIFEST_VARIANT,
        "partition": SELECTION_PARTITION,
    }
    drift = {
        key: (summary.get(key), wanted)
        for key, wanted in required_top.items()
        if summary.get(key) != wanted
    }
    if drift:
        raise NewHeadLRSelectionError(f"LR={rank_lr} summary scope drifted: {drift}")

    checkpoint = _require_mapping(
        summary.get("checkpoint_contract"), label="checkpoint contract"
    )
    if (
        type(checkpoint.get("optimizer_updates")) is not int
        or checkpoint["optimizer_updates"] != OPTIMIZER_UPDATES_PER_CANDIDATE
        or checkpoint.get("experiment_id") != "DD1"
        or type(checkpoint.get("rank_lr")) is not float
        or checkpoint["rank_lr"] != rank_lr
        or checkpoint.get("formal_new_head_partition_evaluation") is not False
    ):
        raise NewHeadLRSelectionError(
            f"LR={rank_lr} checkpoint diagnostic contract drifted"
        )
    training_partition = checkpoint.get("training_partition_status")
    if not isinstance(training_partition, Mapping) or (
        training_partition.get("formal") is not True
    ):
        raise NewHeadLRSelectionError(
            f"LR={rank_lr} was not trained on the formal new-head partition"
        )
    execution = checkpoint.get("formal_execution_status")
    expected_execution = {
        "formal": False,
        "reason": "diagnostic_new_head_execution_scope",
        "execution_scope": EXECUTION_SCOPE,
        "formal_fresh_start": False,
        "declared_optimizer_updates": OPTIMIZER_UPDATES_PER_CANDIDATE,
        "observed_optimizer_updates": OPTIMIZER_UPDATES_PER_CANDIDATE,
        "formal_contract": EXECUTION_CONTRACT,
    }
    if not isinstance(execution, Mapping) or dict(execution) != expected_execution:
        raise NewHeadLRSelectionError(
            f"LR={rank_lr} execution scope/contract drifted"
        )

    paired = _require_mapping(
        checkpoint.get("paired_training_contract"),
        label="paired training contract",
    )
    if not (
        type(paired.get("stage_b_data_driven_rank_lr")) is float
        and paired["stage_b_data_driven_rank_lr"] == rank_lr
        and paired.get("stage_b_data_driven_execution_scope") == EXECUTION_SCOPE
        and paired.get("stage_b_data_driven_formal_fresh_start") is False
        and paired.get("stage_b_data_driven_formal_expected_optimizer_updates")
        == OPTIMIZER_UPDATES_PER_CANDIDATE
        and paired.get("stage_b_data_driven_new_head_formal_contract")
        == EXECUTION_CONTRACT
    ):
        raise NewHeadLRSelectionError(
            f"LR={rank_lr} paired training LR/probe contract drifted"
        )
    evaluation_config = _require_mapping(
        checkpoint.get("evaluation_config_training_contract"),
        label="evaluation config training contract",
    )
    if not (
        type(evaluation_config.get("stage_b_data_driven_rank_lr")) is float
        and evaluation_config["stage_b_data_driven_rank_lr"] == rank_lr
    ):
        raise NewHeadLRSelectionError(
            f"LR={rank_lr} evaluation config rank LR drifted"
        )

    metrics = _require_mapping(summary.get("metrics"), label="summary metrics")
    selected_metrics = {
        SELECTION_METRIC: _require_metric(
            metrics.get(SELECTION_METRIC),
            label=f"LR={rank_lr} {SELECTION_METRIC}",
            unit_interval=True,
        ),
        SECONDARY_SELECTION_METRIC: _require_metric(
            metrics.get(SECONDARY_SELECTION_METRIC),
            label=f"LR={rank_lr} {SECONDARY_SELECTION_METRIC}",
            unit_interval=False,
        ),
    }
    return CandidateSummary(
        rank_lr=rank_lr,
        summary_file=summary_file,
        checkpoint_file=_validate_bound_file_record(
            checkpoint.get("checkpoint"), label="checkpoint"
        ),
        config_file=_validate_bound_file_record(
            checkpoint.get("config"), label="evaluation config"
        ),
        metrics=selected_metrics,
        protocol=_require_mapping(summary.get("protocol"), label="protocol"),
        evaluation_inputs=_require_mapping(
            summary.get("evaluation_inputs"), label="evaluation inputs"
        ),
        initializer_provenance=_require_mapping(
            checkpoint.get("initializer_provenance"),
            label="initializer provenance",
        ),
        shared_training_provenance=_require_mapping(
            checkpoint.get("shared_training_provenance"),
            label="shared training provenance",
        ),
        paired_training_contract_without_rank_lr=_without_rank_lr(
            paired, label="paired training contract"
        ),
        evaluation_config_contract_without_rank_lr=_without_rank_lr(
            evaluation_config, label="evaluation config training contract"
        ),
    )


def _require_shared_contracts(
    candidates: Sequence[CandidateSummary],
) -> dict[str, str]:
    if len(candidates) != len(CANDIDATE_RANK_LRS):
        raise NewHeadLRSelectionError("exactly three candidate summaries are required")
    fields = (
        "protocol",
        "evaluation_inputs",
        "initializer_provenance",
        "shared_training_provenance",
        "paired_training_contract_without_rank_lr",
        "evaluation_config_contract_without_rank_lr",
    )
    first = candidates[0]
    hashes = {}
    for field in fields:
        expected = getattr(first, field)
        for candidate in candidates[1:]:
            if getattr(candidate, field) != expected:
                raise NewHeadLRSelectionError(
                    f"candidate summaries differ in shared contract: {field}"
                )
        hashes[f"{field}_sha256"] = _sha256_bytes(_canonical_bytes(expected))
    return hashes


def _select_candidate(candidates: Sequence[CandidateSummary]) -> CandidateSummary:
    return min(
        candidates,
        key=lambda item: (
            -item.metrics[SELECTION_METRIC],
            item.metrics[SECONDARY_SELECTION_METRIC],
            item.rank_lr,
        ),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise NewHeadLRSelectionError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("ascii") + b"\n"
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise NewHeadLRSelectionError(
                f"refusing concurrent replacement: {path}"
            ) from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def build_selection_receipt(
    candidate_paths: Mapping[float, Path],
    *,
    preregistration: Path,
    output: Path,
) -> dict[str, Any]:
    if set(candidate_paths) != set(CANDIDATE_RANK_LRS):
        raise NewHeadLRSelectionError(
            "candidate LRs must be exactly 3e-5, 1e-4, and 3e-4"
        )
    resolved = [path.expanduser().resolve(strict=True) for path in candidate_paths.values()]
    if len(set(resolved)) != len(resolved):
        raise NewHeadLRSelectionError("candidate summary paths must be distinct")
    _preregistration_value, preregistration_file = _validate_preregistration(
        preregistration
    )
    candidates = [
        _validate_summary(candidate_paths[rank_lr], rank_lr=rank_lr)
        for rank_lr in CANDIDATE_RANK_LRS
    ]
    shared_hashes = _require_shared_contracts(candidates)
    selected = _select_candidate(candidates)
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "candidate_rank_lrs": list(CANDIDATE_RANK_LRS),
        "optimizer_updates_per_candidate": OPTIMIZER_UPDATES_PER_CANDIDATE,
        "selection_partition": SELECTION_PARTITION,
        "selection_metric": SELECTION_METRIC,
        "secondary_selection_metric": SECONDARY_SELECTION_METRIC,
        "tie_break_rule": list(TIE_BREAK_RULE),
        "selected_rank_lr": selected.rank_lr,
        "candidates": [
            {
                "rank_lr": item.rank_lr,
                "summary_file": item.summary_file,
                "checkpoint_file": item.checkpoint_file,
                "config_file": item.config_file,
                "metrics": item.metrics,
            }
            for item in candidates
        ],
        "shared_contract_sha256": shared_hashes,
        "preregistration": preregistration_file,
        "invariants": {
            "preregistration_schema_status_hash_and_invariants_valid": True,
            "candidate_lr_set_matches_preregistration": True,
            "all_summaries_are_canonical_dd1_dev_screen_diagnostics": True,
            "all_candidates_use_the_shared_ordinary_primary_eval_manifest": True,
            "all_candidates_have_exactly_1000_optimizer_updates": True,
            "all_candidates_bind_the_formal_training_partition_only": True,
            "no_candidate_is_formal_headline_eligible": True,
            "checkpoint_and_config_files_match_summary_bindings": True,
            "protocol_evaluation_inputs_initializer_and_provenance_match": True,
            "paired_training_contracts_differ_only_in_rank_lr": True,
            "selection_rule_replayed_without_manual_override": True,
            "output_is_create_new_and_never_overwritten": True,
        },
    }
    payload["canonical_payload_sha256"] = _sha256_bytes(
        _canonical_bytes(payload)
    )
    _atomic_write_new(output, payload)
    return payload


def _parse_candidate_args(values: Sequence[str]) -> dict[float, Path]:
    if len(values) != len(CANDIDATE_RANK_LRS):
        raise NewHeadLRSelectionError("provide exactly three --candidate arguments")
    result: dict[float, Path] = {}
    for value in values:
        if "=" not in value:
            raise NewHeadLRSelectionError(
                f"candidate must use rank_lr=summary.json: {value!r}"
            )
        raw_lr, raw_path = value.split("=", 1)
        try:
            rank_lr = float(raw_lr)
        except ValueError as error:
            raise NewHeadLRSelectionError(
                f"candidate rank LR is invalid: {raw_lr!r}"
            ) from error
        if rank_lr not in CANDIDATE_RANK_LRS or not raw_path.strip():
            raise NewHeadLRSelectionError(f"unsupported candidate: {value!r}")
        if rank_lr in result:
            raise NewHeadLRSelectionError(f"duplicate candidate LR: {rank_lr}")
        result[rank_lr] = Path(raw_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="RANK_LR=SUMMARY.JSON",
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_selection_receipt(
        _parse_candidate_args(args.candidate),
        preregistration=args.preregistration,
        output=args.output,
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
