#!/usr/bin/env python3
"""Build a SHA-bound Stage-B paper-results manifest from evaluation roots.

The build spec names training artifacts and the three formal evaluation roots
for every run.  This tool derives record paths only from uniquely selected
summary rows, validates the row/checkpoint/run bindings, parses all ten record
files, and emits the ``stageb-paper-results-manifest-v1`` contract consumed by
``aggregate_stageb_paper_results.py``.

Output creation is fail-closed: an existing destination is never replaced.
``--validate`` asks the aggregator to load the generated manifest with a small
bootstrap count before the destination is atomically created.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_stageb_fpr95_records import (  # noqa: E402
    RecordComparisonError,
    exact_fpr95,
    load_manifest,
    load_tn_records,
)
from tools.stageb_eval_records import RECORD_SCHEMA  # noqa: E402
from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLITS,
)


SPEC_SCHEMA = "stageb-paper-results-build-spec-v1"
OUTPUT_SCHEMA = "stageb-paper-results-manifest-v1"
TN_SPLITS = ("strict2031", "strict1607")
EVALUATION_LAUNCH_SCHEMA = "pivot.stageb.paper_evaluation_launch/v1"
EVALUATION_POSTFLIGHT_SCHEMA = "pivot.stageb.paper_evaluation_postflight/v1"
FINAL_EVALUATION_PROFILE = "final"
_SHA_CACHE: Dict[tuple[str, int, int], str] = {}


class PaperManifestBuildError(ValueError):
    """Raised when source evidence does not prove a complete manifest."""


def _object(
    value: Any,
    *,
    label: str,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperManifestBuildError(f"{label}: expected an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unexpected = sorted(set(value) - allowed)
    if missing:
        raise PaperManifestBuildError(f"{label}: missing fields {missing}")
    if unexpected:
        raise PaperManifestBuildError(f"{label}: unexpected fields {unexpected}")
    return value


def _required_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise PaperManifestBuildError(f"{label}: expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    raise PaperManifestBuildError(f"{label}: expected an integer")


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperManifestBuildError(f"{label}: expected a non-empty string")
    return value.strip()


def _resolve_path(value: Any, base_dir: Path, *, label: str) -> Path:
    text = _nonempty_string(value, label=label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rendered = digest.hexdigest()
    _SHA_CACHE[key] = rendered
    return rendered


def _artifact_spec(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, str):
        return {"path": value}
    return _object(
        value,
        label=label,
        required=("path",),
        optional=("sha256", "size_bytes", "label"),
    )


def _file_record(value: Any, base_dir: Path, *, label: str) -> Dict[str, Any]:
    specification = _artifact_spec(value, label=label)
    path = _resolve_path(specification["path"], base_dir, label=f"{label}.path")
    if not path.is_file():
        raise PaperManifestBuildError(f"{label}: file does not exist: {path}")
    size = int(path.stat().st_size)
    if "size_bytes" in specification:
        expected_size = _required_int(
            specification["size_bytes"], label=f"{label}.size_bytes"
        )
        if size != expected_size:
            raise PaperManifestBuildError(
                f"{label}: size mismatch, expected {expected_size}, found {size}"
            )
    actual_sha = _sha256(path)
    if "sha256" in specification:
        expected_sha = _nonempty_string(
            specification["sha256"], label=f"{label}.sha256"
        )
        if expected_sha != expected_sha.lower() or len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise PaperManifestBuildError(
                f"{label}.sha256: expected 64 lowercase hexadecimal characters"
            )
        if actual_sha != expected_sha:
            raise PaperManifestBuildError(
                f"{label}: SHA-256 mismatch, expected {expected_sha}, found {actual_sha}"
            )
    result: Dict[str, Any] = {
        "path": str(path),
        "sha256": actual_sha,
        "size_bytes": size,
    }
    if "label" in specification:
        result["label"] = str(specification["label"])
    return result


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperManifestBuildError(f"{label}: cannot read JSON: {error}") from error


def _reported_candidates(raw_value: Any, *, summary_path: Path, spec_base: Path) -> tuple[Path, ...]:
    text = _nonempty_string(raw_value, label="reported path")
    raw = Path(text).expanduser()
    values = [raw] if raw.is_absolute() else [
        REPO_ROOT / raw,
        summary_path.parent / raw,
        spec_base / raw,
    ]
    unique: list[Path] = []
    for candidate in values:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _resolve_reported_file(
    value: Any,
    *,
    summary_path: Path,
    spec_base: Path,
    label: str,
) -> Path:
    try:
        candidates = _reported_candidates(
            value, summary_path=summary_path, spec_base=spec_base
        )
    except PaperManifestBuildError as error:
        raise PaperManifestBuildError(f"{label}: {error}") from error
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise PaperManifestBuildError(
            f"{label}: reported path must resolve to exactly one file; "
            f"found {len(existing)} among {[str(path) for path in candidates]}"
        )
    return existing[0]


def _reported_matches(
    value: Any,
    expected: Path,
    *,
    summary_path: Path,
    spec_base: Path,
) -> bool:
    try:
        candidates = _reported_candidates(
            value, summary_path=summary_path, spec_base=spec_base
        )
    except PaperManifestBuildError:
        return False
    return expected.resolve() in candidates


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _verify_postflight_file_record(
    value: Any,
    *,
    spec_base: Path,
    output_root: Path,
    expected_path: Path,
    label: str,
) -> Dict[str, Any]:
    specification = _object(
        value,
        label=label,
        required=("path", "sha256", "size_bytes"),
        optional=("mtime_ns", "roles"),
    )
    record = _file_record(
        {
            "path": specification["path"],
            "sha256": specification["sha256"],
            "size_bytes": specification["size_bytes"],
        },
        spec_base,
        label=label,
    )
    observed_path = Path(record["path"])
    if observed_path != expected_path.resolve():
        raise PaperManifestBuildError(
            f"{label}: path {observed_path} != expected {expected_path.resolve()}"
        )
    if not _inside(observed_path, output_root):
        raise PaperManifestBuildError(
            f"{label}: artifact escapes evaluation root {output_root}"
        )
    if "mtime_ns" in specification and _required_int(
        specification["mtime_ns"], label=f"{label}.mtime_ns"
    ) != int(observed_path.stat().st_mtime_ns):
        raise PaperManifestBuildError(f"{label}: mtime_ns mismatch")
    if "roles" in specification:
        roles = specification["roles"]
        if not isinstance(roles, list) or any(
            not isinstance(role, str) or not role for role in roles
        ) or len(set(roles)) != len(roles):
            raise PaperManifestBuildError(f"{label}: roles must be unique strings")
    return record


def _verify_final_postflight_artifacts(
    postflight: Mapping[str, Any],
    *,
    root: Path,
    spec_base: Path,
    label: str,
) -> None:
    primary_dir = root / "ref8_strict2031"
    supplemental_dir = root / "strict1607"
    primary_summary_path = primary_dir / "summary.json"
    supplemental_summary_path = supplemental_dir / "summary.json"
    artifacts = _object(
        postflight.get("artifacts"),
        label=f"{label}.postflight.artifacts",
        required=(
            "primary_summary",
            "supplemental_summary",
            "ref8",
            "strict2031",
            "strict1607",
        ),
    )
    _verify_postflight_file_record(
        artifacts["primary_summary"],
        spec_base=spec_base,
        output_root=root,
        expected_path=primary_summary_path,
        label=f"{label}.postflight.artifacts.primary_summary",
    )
    _verify_postflight_file_record(
        artifacts["supplemental_summary"],
        spec_base=spec_base,
        output_root=root,
        expected_path=supplemental_summary_path,
        label=f"{label}.postflight.artifacts.supplemental_summary",
    )
    primary_summary = _read_json(
        primary_summary_path, label=f"{label}.postflight.primary_summary"
    )
    supplemental_summary = _read_json(
        supplemental_summary_path, label=f"{label}.postflight.supplemental_summary"
    )
    for summary_name, summary in (
        ("primary", primary_summary),
        ("supplemental", supplemental_summary),
    ):
        if not isinstance(summary, Mapping) or set(summary) != {"refcoco", "tn"}:
            raise PaperManifestBuildError(
                f"{label}: {summary_name} summary shape differs from evaluator contract"
            )
        if not isinstance(summary["refcoco"], list) or not isinstance(
            summary["tn"], list
        ):
            raise PaperManifestBuildError(
                f"{label}: {summary_name} summary sections must be lists"
            )

    ref_rows: Dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(primary_summary["refcoco"]):
        if not isinstance(row, Mapping):
            raise PaperManifestBuildError(
                f"{label}: primary Ref summary row {index} is invalid"
            )
        split = str(row.get("dataset", ""))
        if split not in REF_SPLITS or split in ref_rows:
            raise PaperManifestBuildError(
                f"{label}: primary Ref summary split set is invalid"
            )
        ref_rows[split] = row
    if tuple(ref_rows) != REF_SPLITS:
        raise PaperManifestBuildError(
            f"{label}: primary Ref summary order/set differs from official Ref8"
        )
    ref_artifacts = _object(
        artifacts["ref8"],
        label=f"{label}.postflight.artifacts.ref8",
        required=REF_SPLITS,
    )
    for split in REF_SPLITS:
        evidence = _object(
            ref_artifacts[split],
            label=f"{label}.postflight.artifacts.ref8.{split}",
            required=(
                "summary_acc50",
                "manifest_n",
                "manifest_sha256",
                "records",
            ),
        )
        official = REF_SPLIT_CONTRACT[split]
        if _required_int(
            evidence["manifest_n"],
            label=f"{label}.postflight.artifacts.ref8.{split}.manifest_n",
        ) != int(official["rows"]) or str(evidence["manifest_sha256"]).lower() != str(
            official["sha256"]
        ):
            raise PaperManifestBuildError(
                f"{label}: postflight Ref artifact {split} differs from official contract"
            )
        row = ref_rows[split]
        try:
            evidence_acc = float(evidence["summary_acc50"])
            row_acc = float(row.get("acc50"))
        except (TypeError, ValueError) as error:
            raise PaperManifestBuildError(
                f"{label}: postflight Ref artifact {split} acc50 is invalid"
            ) from error
        if not math.isfinite(evidence_acc) or evidence_acc != row_acc:
            raise PaperManifestBuildError(
                f"{label}: postflight Ref artifact {split} differs from summary"
            )
        record_path = _resolve_reported_file(
            row.get("records_jsonl"),
            summary_path=primary_summary_path,
            spec_base=spec_base,
            label=f"{label}.postflight.ref8.{split}.records_jsonl",
        )
        _verify_postflight_file_record(
            evidence["records"],
            spec_base=spec_base,
            output_root=root,
            expected_path=record_path,
            label=f"{label}.postflight.artifacts.ref8.{split}.records",
        )

    strict_summaries = {
        "strict2031": primary_summary,
        "strict1607": supplemental_summary,
    }
    for strict_split, summary in strict_summaries.items():
        rows = summary["tn"]
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise PaperManifestBuildError(
                f"{label}: {strict_split} summary must contain exactly one TN row"
            )
        row = rows[0]
        evidence = _object(
            artifacts[strict_split],
            label=f"{label}.postflight.artifacts.{strict_split}",
            required=(
                "summary_fpr95",
                "manifest_binding_mode",
                "manifest_n",
                "source_manifest_sha256",
                "derived_manifest_sha256",
                "records",
            ),
        )
        if evidence["manifest_binding_mode"] != "source_to_derived_v1":
            raise PaperManifestBuildError(
                f"{label}: {strict_split} postflight lacks two-layer manifest binding"
            )
        if _required_int(
            evidence["manifest_n"],
            label=f"{label}.postflight.artifacts.{strict_split}.manifest_n",
        ) != _required_int(
            row.get("manifest_n"), label=f"{label}.{strict_split}.manifest_n"
        ):
            raise PaperManifestBuildError(
                f"{label}: {strict_split} postflight manifest_n differs from summary"
            )
        for evidence_field, row_field in (
            ("source_manifest_sha256", "source_manifest_sha256"),
            ("derived_manifest_sha256", "manifest_sha256"),
        ):
            if str(evidence[evidence_field]).lower() != str(row.get(row_field)).lower():
                raise PaperManifestBuildError(
                    f"{label}: {strict_split} postflight {evidence_field} "
                    "differs from summary"
                )
        try:
            evidence_fpr = float(evidence["summary_fpr95"])
            summary_fpr = float(row.get("fpr95tpr"))
        except (TypeError, ValueError) as error:
            raise PaperManifestBuildError(
                f"{label}: {strict_split} postflight FPR95 is invalid"
            ) from error
        if not math.isfinite(evidence_fpr) or not math.isclose(
            evidence_fpr, summary_fpr, rel_tol=0.0, abs_tol=1e-12
        ):
            raise PaperManifestBuildError(
                f"{label}: {strict_split} postflight FPR95 differs from summary"
            )
        summary_path = (
            primary_summary_path
            if strict_split == "strict2031"
            else supplemental_summary_path
        )
        record_path = _resolve_reported_file(
            row.get("records_jsonl"),
            summary_path=summary_path,
            spec_base=spec_base,
            label=f"{label}.postflight.{strict_split}.records_jsonl",
        )
        _verify_postflight_file_record(
            evidence["records"],
            spec_base=spec_base,
            output_root=root,
            expected_path=record_path,
            label=f"{label}.postflight.artifacts.{strict_split}.records",
        )


def _select_row(
    rows: Any,
    *,
    checkpoint: Path,
    summary_path: Path,
    spec_base: Path,
    label: str,
    dataset: Optional[str],
    run_id: Optional[str],
) -> Mapping[str, Any]:
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise PaperManifestBuildError(f"{label}: summary section must be a list of objects")
    candidates = list(rows)
    if dataset is not None:
        candidates = [row for row in candidates if row.get("dataset") == dataset]
    if run_id is not None:
        candidates = [row for row in candidates if row.get("run_id") == run_id]
    candidates = [
        row
        for row in candidates
        if _reported_matches(
            row.get("checkpoint"),
            checkpoint,
            summary_path=summary_path,
            spec_base=spec_base,
        )
    ]
    if len(candidates) != 1:
        selectors = [f"checkpoint={checkpoint}"]
        if dataset is not None:
            selectors.append(f"dataset={dataset!r}")
        if run_id is not None:
            selectors.append(f"run_id={run_id!r}")
        raise PaperManifestBuildError(
            f"{label}: expected exactly one summary row for {', '.join(selectors)}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _summary_run_id(row: Mapping[str, Any], *, label: str) -> str:
    return _nonempty_string(row.get("run_id"), label=f"{label}.run_id")


def _scan_ref_records(
    path: Path,
    *,
    split: str,
    row: Mapping[str, Any],
    run_id: str,
    label: str,
) -> Dict[str, Any]:
    expected_n = _required_int(row.get("manifest_n"), label=f"{label}.manifest_n")
    summary_n = _required_int(
        row.get("num_expressions"), label=f"{label}.num_expressions"
    )
    if expected_n <= 0 or summary_n != expected_n:
        raise PaperManifestBuildError(
            f"{label}: manifest_n and num_expressions must be the same positive value"
        )
    manifest_sha = _nonempty_string(
        row.get("manifest_sha256"), label=f"{label}.manifest_sha256"
    ).lower()
    if len(manifest_sha) != 64 or any(c not in "0123456789abcdef" for c in manifest_sha):
        raise PaperManifestBuildError(f"{label}: invalid manifest_sha256")
    official = REF_SPLIT_CONTRACT[split]
    if expected_n != int(official["rows"]):
        raise PaperManifestBuildError(
            f"{label}: manifest_n={expected_n} differs from official {split} "
            f"rows={official['rows']}"
        )
    if manifest_sha != str(official["sha256"]):
        raise PaperManifestBuildError(
            f"{label}: manifest_sha256 differs from official {split} contract"
        )
    if _required_int(row.get("max_batches", 0), label=f"{label}.max_batches") != 0:
        raise PaperManifestBuildError(f"{label}: max_batches must be zero")
    if _required_int(row.get("invalid_records", 0), label=f"{label}.invalid_records") != 0:
        raise PaperManifestBuildError(f"{label}: invalid_records must be zero")

    digest = hashlib.sha256()
    size = 0
    count = 0
    correct = 0
    identities = set()
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                size += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PaperManifestBuildError(
                        f"{label}:{line_number}: invalid JSON: {error}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise PaperManifestBuildError(f"{label}:{line_number}: expected object")
                location = f"{label}:{line_number}"
                expected_fields = {
                    "schema": RECORD_SCHEMA,
                    "task": "ref",
                    "split": split,
                    "manifest_key": f"ref:{split}",
                    "manifest_sha256": manifest_sha,
                    "manifest_n": expected_n,
                    "manifest_index": count,
                    "run_id": run_id,
                    "valid": True,
                }
                for field, expected in expected_fields.items():
                    if record.get(field) != expected or (
                        field == "valid" and type(record.get(field)) is not bool
                    ):
                        raise PaperManifestBuildError(
                            f"{location}: {field} does not match summary/protocol"
                        )
                if type(record.get("correct50")) is not bool:
                    raise PaperManifestBuildError(f"{location}: correct50 must be boolean")
                identity = (
                    record.get("sample_id"),
                    record.get("image_id"),
                    record.get("ann_id"),
                    record.get("ref_id"),
                    record.get("sent_id"),
                )
                if not str(identity[0] or "").strip() or identity in identities:
                    raise PaperManifestBuildError(
                        f"{location}: missing or duplicate record identity"
                    )
                identities.add(identity)
                correct += int(record["correct50"])
                count += 1
    except OSError as error:
        raise PaperManifestBuildError(f"{label}: cannot read records: {error}") from error
    if count != expected_n:
        raise PaperManifestBuildError(
            f"{label}: records N={count} != summary manifest_n={expected_n}"
        )
    try:
        reported_acc = float(row.get("acc50"))
    except (TypeError, ValueError) as error:
        raise PaperManifestBuildError(f"{label}.acc50: expected a number") from error
    measured_acc = correct / count
    if not math.isfinite(reported_acc) or not math.isclose(
        measured_acc, reported_acc, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PaperManifestBuildError(
            f"{label}: summary acc50={reported_acc} != records acc50={measured_acc}"
        )
    stat = path.stat()
    if int(stat.st_size) != size:
        raise PaperManifestBuildError(f"{label}: records changed while being read")
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "size_bytes": size}


def _load_source_manifests(
    specifications: Any, *, spec_base: Path
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    mapping = _object(
        specifications,
        label="strict_source_manifests",
        required=TN_SPLITS,
    )
    outputs: Dict[str, Any] = {}
    loaded: Dict[str, Any] = {}
    for split in TN_SPLITS:
        label = f"strict_source_manifests.{split}"
        specification = _object(
            mapping[split],
            label=label,
            required=("path", "expected_n"),
            optional=("sha256", "size_bytes", "label"),
        )
        artifact_input = {
            key: value
            for key, value in specification.items()
            if key != "expected_n"
        }
        record = _file_record(artifact_input, spec_base, label=label)
        expected_n = _required_int(specification["expected_n"], label=f"{label}.expected_n")
        if expected_n <= 0:
            raise PaperManifestBuildError(f"{label}.expected_n must be positive")
        try:
            manifest = load_manifest(record["path"])
        except (RecordComparisonError, OSError, ValueError) as error:
            raise PaperManifestBuildError(f"{label}: invalid TN manifest: {error}") from error
        if len(manifest.rows) != expected_n:
            raise PaperManifestBuildError(
                f"{label}: rows={len(manifest.rows)} != expected_n={expected_n}"
            )
        if dict(manifest.file_record) != {
            key: record[key] for key in ("path", "size_bytes", "sha256")
        }:
            raise PaperManifestBuildError(f"{label}: manifest changed while being parsed")
        outputs[split] = {"manifest": record, "expected_n": expected_n}
        loaded[split] = manifest
    return outputs, loaded


def _build_ref_result(
    root: Path,
    *,
    checkpoint: Path,
    run_selector: Optional[str],
    spec_base: Path,
    label: str,
) -> tuple[Dict[str, Any], str]:
    summary_path = root / "summary.json"
    summary_record = _file_record(str(summary_path), spec_base, label=f"{label}.summary")
    summary = _read_json(summary_path, label=f"{label}.summary")
    if not isinstance(summary, Mapping):
        raise PaperManifestBuildError(f"{label}.summary: root must be an object")
    records: Dict[str, Any] = {}
    selected_run_ids = set()
    for split in REF_SPLITS:
        row_label = f"{label}.{split}"
        row = _select_row(
            summary.get("refcoco"),
            checkpoint=checkpoint,
            summary_path=summary_path,
            spec_base=spec_base,
            label=row_label,
            dataset=split,
            run_id=run_selector,
        )
        selected_run_id = _summary_run_id(row, label=row_label)
        selected_run_ids.add(selected_run_id)
        record_path = _resolve_reported_file(
            row.get("records_jsonl"),
            summary_path=summary_path,
            spec_base=spec_base,
            label=f"{row_label}.records_jsonl",
        )
        if not _inside(record_path, root):
            raise PaperManifestBuildError(
                f"{row_label}: records path escapes evaluation root {root}"
            )
        records[split] = _scan_ref_records(
            record_path,
            split=split,
            row=row,
            run_id=selected_run_id,
            label=f"{row_label}.records",
        )
    if len(selected_run_ids) != 1:
        raise PaperManifestBuildError(
            f"{label}: selected Ref rows have mixed run_id values {sorted(selected_run_ids)}"
        )
    selected_run_id = next(iter(selected_run_ids))
    return {
        "summary": summary_record,
        "records": records,
        "run_id": selected_run_id,
    }, selected_run_id


def _build_tn_result(
    root: Path,
    *,
    split: str,
    checkpoint: Path,
    run_selector: Optional[str],
    expected_run_id: str,
    spec_base: Path,
    source_manifest: Any,
    label: str,
) -> Dict[str, Any]:
    summary_path = root / "summary.json"
    summary_record = _file_record(str(summary_path), spec_base, label=f"{label}.summary")
    summary = _read_json(summary_path, label=f"{label}.summary")
    if not isinstance(summary, Mapping):
        raise PaperManifestBuildError(f"{label}.summary: root must be an object")
    row = _select_row(
        summary.get("tn"),
        checkpoint=checkpoint,
        summary_path=summary_path,
        spec_base=spec_base,
        label=label,
        dataset=None,
        run_id=run_selector,
    )
    selected_run_id = _summary_run_id(row, label=label)
    if selected_run_id != expected_run_id:
        raise PaperManifestBuildError(
            f"{label}: run_id={selected_run_id!r} differs from Ref run_id={expected_run_id!r}"
        )
    record_path = _resolve_reported_file(
        row.get("records_jsonl"),
        summary_path=summary_path,
        spec_base=spec_base,
        label=f"{label}.records_jsonl",
    )
    if not _inside(record_path, root):
        raise PaperManifestBuildError(
            f"{label}: records path escapes evaluation root {root}"
        )
    try:
        records = load_tn_records(record_path, source_manifest, label=label)
    except (RecordComparisonError, OSError, ValueError) as error:
        raise PaperManifestBuildError(f"{label}: invalid TN records: {error}") from error
    if records.run_ids != (selected_run_id,):
        raise PaperManifestBuildError(
            f"{label}: TN record run_ids={records.run_ids} do not match selected row"
        )
    if not bool(records.valid.all()):
        raise PaperManifestBuildError(f"{label}: all formal TN records must be valid")
    if _required_int(row.get("manifest_n"), label=f"{label}.manifest_n") != len(records.rows):
        raise PaperManifestBuildError(f"{label}: summary manifest_n mismatch")
    if _required_int(row.get("num_pairs"), label=f"{label}.num_pairs") != len(records.rows):
        raise PaperManifestBuildError(f"{label}: summary num_pairs mismatch")
    if _required_int(row.get("max_batches", 0), label=f"{label}.max_batches") != 0:
        raise PaperManifestBuildError(f"{label}: max_batches must be zero")
    if _required_int(row.get("invalid_records", 0), label=f"{label}.invalid_records") != 0:
        raise PaperManifestBuildError(f"{label}: invalid_records must be zero")
    reported_source = _resolve_reported_file(
        row.get("source_manifest_path"),
        summary_path=summary_path,
        spec_base=spec_base,
        label=f"{label}.source_manifest_path",
    )
    if reported_source != source_manifest.path.resolve():
        raise PaperManifestBuildError(
            f"{label}: summary source manifest differs from locked {split} source"
        )
    if str(row.get("source_manifest_sha256", "")).lower() != source_manifest.sha256:
        raise PaperManifestBuildError(f"{label}: summary source manifest SHA mismatch")
    if _required_int(row.get("source_manifest_n"), label=f"{label}.source_manifest_n") != len(source_manifest.rows):
        raise PaperManifestBuildError(f"{label}: summary source manifest N mismatch")
    if _required_int(
        row.get("source_manifest_size_bytes"),
        label=f"{label}.source_manifest_size_bytes",
    ) != int(source_manifest.file_record["size_bytes"]):
        raise PaperManifestBuildError(f"{label}: summary source manifest size mismatch")
    record_manifest_hashes = {
        str(record.get("manifest_sha256", "")).lower() for record in records.rows
    }
    if record_manifest_hashes != {str(row.get("manifest_sha256", "")).lower()}:
        raise PaperManifestBuildError(f"{label}: summary/record manifest SHA mismatch")
    measured_fpr = float(exact_fpr95(records.positive, records.negative)["fpr"])
    try:
        reported_fpr = float(row.get("fpr95tpr"))
    except (TypeError, ValueError) as error:
        raise PaperManifestBuildError(f"{label}.fpr95tpr must be numeric") from error
    if not math.isfinite(reported_fpr) or not math.isclose(
        measured_fpr, reported_fpr, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PaperManifestBuildError(
            f"{label}: summary FPR95={reported_fpr} != records FPR95={measured_fpr}"
        )
    return {
        "summary": summary_record,
        "records": dict(records.file_record),
        "run_id": selected_run_id,
    }


def _bootstrap_spec(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"iterations": 5000, "confidence": 0.95, "seed": 20260717}
    specification = _object(
        value,
        label="bootstrap",
        required=("iterations", "confidence", "seed"),
    )
    iterations = _required_int(specification["iterations"], label="bootstrap.iterations")
    seed = _required_int(specification["seed"], label="bootstrap.seed")
    try:
        confidence = float(specification["confidence"])
    except (TypeError, ValueError) as error:
        raise PaperManifestBuildError("bootstrap.confidence must be numeric") from error
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise PaperManifestBuildError("bootstrap iterations/confidence are invalid")
    return {"iterations": iterations, "confidence": confidence, "seed": seed}


def _derive_run_from_evaluation_root(
    value: Any,
    *,
    spec_base: Path,
    train_seed: int,
    expected_training_run_id: str,
    label: str,
) -> Dict[str, Any]:
    """Derive one results run from a completed sealed final evaluation."""

    root = _resolve_path(value, spec_base, label=f"{label}.evaluation_root")
    if not root.is_dir():
        raise PaperManifestBuildError(
            f"{label}.evaluation_root is not a directory: {root}"
        )
    launch_path = root / "launch_manifest.json"
    postflight_path = root / "postflight.json"
    if not launch_path.is_file() or not postflight_path.is_file():
        raise PaperManifestBuildError(
            f"{label}.evaluation_root lacks launch_manifest.json/postflight.json"
        )
    launch = _read_json(launch_path, label=f"{label}.evaluation_launch")
    postflight = _read_json(postflight_path, label=f"{label}.evaluation_postflight")
    if not isinstance(launch, Mapping) or launch.get("schema") != EVALUATION_LAUNCH_SCHEMA:
        raise PaperManifestBuildError(f"{label}: evaluation launch schema mismatch")
    if launch.get("status") != "completed":
        raise PaperManifestBuildError(f"{label}: evaluation launch is not completed")
    if _resolve_path(
        launch.get("output_dir"), spec_base, label=f"{label}.evaluation_output_dir"
    ) != root:
        raise PaperManifestBuildError(f"{label}: evaluation output_dir mismatch")
    protocol = launch.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("profile") != FINAL_EVALUATION_PROFILE:
        raise PaperManifestBuildError(
            f"{label}: only the final evaluation profile may enter paper aggregation"
        )
    if protocol.get("processes") != ["ref8_strict2031", "strict1607"]:
        raise PaperManifestBuildError(f"{label}: final evaluation process contract drifted")
    completed = launch.get("completed_phases")
    if not isinstance(completed, list) or [
        value.get("phase_id") if isinstance(value, Mapping) else None
        for value in completed
    ] != ["ref8_strict2031", "strict1607"] or any(
        not isinstance(value, Mapping)
        or value.get("status") != "completed"
        or value.get("returncode") != 0
        for value in completed
    ):
        raise PaperManifestBuildError(f"{label}: final evaluation phases are incomplete")
    if not isinstance(postflight, Mapping) or postflight.get("schema") != (
        EVALUATION_POSTFLIGHT_SCHEMA
    ) or postflight.get("status") != "passed" or postflight.get("profile") != (
        FINAL_EVALUATION_PROFILE
    ):
        raise PaperManifestBuildError(f"{label}: final evaluation postflight did not pass")
    if launch.get("postflight") != postflight:
        raise PaperManifestBuildError(
            f"{label}: embedded and persisted evaluation postflight differ"
        )
    postflight_artifact = launch.get("postflight_artifact")
    if not isinstance(postflight_artifact, Mapping):
        raise PaperManifestBuildError(f"{label}: evaluation postflight is not bound")
    observed_postflight = _file_record(
        str(postflight_path), spec_base, label=f"{label}.evaluation_postflight"
    )
    if (
        str(postflight_artifact.get("sha256", ""))
        != observed_postflight["sha256"]
        or _required_int(
            postflight_artifact.get("size_bytes"),
            label=f"{label}.postflight_artifact.size_bytes",
        )
        != observed_postflight["size_bytes"]
        or Path(str(postflight_artifact.get("path", ""))).resolve()
        != postflight_path.resolve()
    ):
        raise PaperManifestBuildError(f"{label}: evaluation postflight binding mismatch")
    input_rehash = postflight.get("input_rehash")
    if not isinstance(input_rehash, Mapping) or input_rehash.get("status") != "passed":
        raise PaperManifestBuildError(f"{label}: evaluation input rehash did not pass")
    launch_inputs = launch.get("inputs")
    launch_records = (
        launch_inputs.get("records") if isinstance(launch_inputs, Mapping) else None
    )
    rehash_records = input_rehash.get("records")
    if not isinstance(launch_records, list) or not launch_records or not isinstance(
        rehash_records, list
    ) or len(rehash_records) != len(launch_records):
        raise PaperManifestBuildError(
            f"{label}: evaluation input/rehash record sets are incomplete"
        )
    launch_by_path: Dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(launch_records):
        if not isinstance(record, Mapping):
            raise PaperManifestBuildError(
                f"{label}: evaluation input record {index} is invalid"
            )
        path = str(Path(str(record.get("path", ""))).resolve())
        if not path or path in launch_by_path:
            raise PaperManifestBuildError(
                f"{label}: evaluation input paths are missing/duplicated"
            )
        launch_by_path[path] = record
    observed_paths: set[str] = set()
    for index, record in enumerate(rehash_records):
        if not isinstance(record, Mapping) or record.get("passed") is not True:
            raise PaperManifestBuildError(
                f"{label}: evaluation rehash record {index} did not pass"
            )
        path = str(Path(str(record.get("path", ""))).resolve())
        launch_record = launch_by_path.get(path)
        if launch_record is None or path in observed_paths:
            raise PaperManifestBuildError(
                f"{label}: evaluation rehash paths do not match launch inputs"
            )
        observed_paths.add(path)
        if not (
            record.get("expected_sha256") == launch_record.get("sha256")
            and record.get("observed_sha256") == launch_record.get("sha256")
            and _required_int(
                record.get("observed_size_bytes"),
                label=f"{label}.rehash[{index}].observed_size_bytes",
            )
            == _required_int(
                launch_record.get("size_bytes"),
                label=f"{label}.input[{index}].size_bytes",
            )
            and _required_int(
                record.get("observed_mtime_ns"),
                label=f"{label}.rehash[{index}].observed_mtime_ns",
            )
            == _required_int(
                launch_record.get("mtime_ns"),
                label=f"{label}.input[{index}].mtime_ns",
            )
        ):
            raise PaperManifestBuildError(
                f"{label}: evaluation rehash record {index} differs from launch"
            )
    if observed_paths != set(launch_by_path):
        raise PaperManifestBuildError(
            f"{label}: evaluation rehash did not cover every launch input"
        )
    contracts = postflight.get("contracts")
    required_contracts = {
        "ref_split_set_exact",
        "full_per_example_records",
        "zero_invalid_records",
        "locked_manifest_binding",
        "checkpoint_consistent_across_all_rows",
        "strict1607_skip_ref_observed",
    }
    if not isinstance(contracts, Mapping) or any(
        contracts.get(field) is not True for field in required_contracts
    ):
        raise PaperManifestBuildError(
            f"{label}: final evaluation postflight contracts are incomplete"
        )
    primary = root / "ref8_strict2031"
    supplemental = root / "strict1607"
    if not primary.is_dir() or not supplemental.is_dir():
        raise PaperManifestBuildError(
            f"{label}: final evaluation result directories are incomplete"
        )
    _verify_final_postflight_artifacts(
        postflight,
        root=root,
        spec_base=spec_base,
        label=label,
    )

    release_evidence = None
    if launch.get("headline_release") is not None:
        from tools import stageb_headline_release_contract as headline_release

        try:
            release_evidence = headline_release.validate_completed_final_plan(
                launch,
                final_artifacts=postflight.get("artifacts"),
            )
        except headline_release.HeadlineReleaseError as error:
            raise PaperManifestBuildError(
                f"{label}: headline release receipt replay failed: {error}"
            ) from error
        if postflight.get("headline_release") != release_evidence:
            raise PaperManifestBuildError(
                f"{label}: persisted headline release evidence differs from replay"
            )

    source = launch.get("source")
    if not isinstance(source, Mapping) or source.get("kind") not in {
        "pivot_paper_training_run",
        "pivot_token_ablation_training_run",
        "historical_pure_gdino_explicit",
    }:
        raise PaperManifestBuildError(
            f"{label}: evaluation_root mode requires a sealed PIVOT source or fixed b58"
        )
    fixed_baseline = source.get("kind") == "historical_pure_gdino_explicit"
    try:
        source_seed = int(42 if fixed_baseline else source.get("training_seed"))
    except (TypeError, ValueError) as error:
        raise PaperManifestBuildError(
            f"{label}: evaluation source training seed is invalid"
        ) from error
    if source_seed != train_seed:
        raise PaperManifestBuildError(
            f"{label}: evaluation source seed {source_seed} != declared {train_seed}"
        )
    training_run_id = _nonempty_string(
        (
            "gdino_stageb_data_ft_b58"
            if fixed_baseline
            else source.get("training_run_id")
        ),
        label=f"{label}.source.training_run_id",
    )
    if training_run_id != expected_training_run_id:
        raise PaperManifestBuildError(
            f"{label}: evaluation source run_id {training_run_id!r} != "
            f"declared {expected_training_run_id!r}"
        )
    if not fixed_baseline:
        try:
            run_id_seed = int(training_run_id.rsplit(":", 1)[1])
        except (ValueError, IndexError) as error:
            raise PaperManifestBuildError(
                f"{label}: evaluation source training_run_id is malformed"
            ) from error
        if run_id_seed != train_seed:
            raise PaperManifestBuildError(
                f"{label}: evaluation source run_id seed differs from train_seed"
            )
    checkpoint = _resolve_path(
        source.get("checkpoint"), spec_base, label=f"{label}.source.checkpoint"
    )
    config = _resolve_path(
        source.get("config"), spec_base, label=f"{label}.source.config"
    )
    if not checkpoint.is_file() or not config.is_file():
        raise PaperManifestBuildError(
            f"{label}: evaluation source checkpoint/config is missing"
        )
    checkpoint_evidence = postflight.get("checkpoint")
    if not isinstance(checkpoint_evidence, Mapping) or Path(
        str(checkpoint_evidence.get("path", ""))
    ).resolve() != checkpoint or str(checkpoint_evidence.get("sha256", "")) != (
        _sha256(checkpoint)
    ):
        raise PaperManifestBuildError(
            f"{label}: evaluation postflight checkpoint differs from source"
        )
    if release_evidence is not None:
        launch_inputs = launch.get("inputs")
        launch_records = (
            launch_inputs.get("records")
            if isinstance(launch_inputs, Mapping)
            else None
        )
        if not isinstance(launch_records, list):
            raise PaperManifestBuildError(
                f"{label}: evaluation input records are missing"
            )
        data = sorted(
            {
                str(Path(str(record.get("path", ""))).resolve(strict=True))
                for record in launch_records
                if isinstance(record, Mapping)
                and isinstance(record.get("roles"), list)
                and "evaluation_data_input" in record["roles"]
            }
        )
    else:
        raw_data = source.get("training_data")
        if not isinstance(raw_data, list) or not raw_data:
            raise PaperManifestBuildError(
                f"{label}: legacy evaluation source lacks training_data"
            )
        data = [
            str(
                _resolve_path(
                    item,
                    spec_base,
                    label=f"{label}.source.training_data[{index}]",
                )
            )
            for index, item in enumerate(raw_data)
        ]
    if len(set(data)) != len(data) or any(not Path(path).is_file() for path in data):
        raise PaperManifestBuildError(
            f"{label}: evaluation data closure is missing or duplicated"
        )
    if not data:
        raise PaperManifestBuildError(f"{label}: evaluation data closure is empty")
    return {
        "training_run_id": training_run_id,
        "evaluation_root": str(root),
        "checkpoint": str(checkpoint),
        "config": str(config),
        "data": data,
        "ref_eval_root": str(primary),
        "strict2031_eval_root": str(primary),
        "strict1607_eval_root": str(supplemental),
        "_headline_release_evidence": release_evidence,
    }


def build_manifest(spec_path: str | Path) -> Dict[str, Any]:
    path = Path(spec_path).expanduser().resolve()
    if not path.is_file():
        raise PaperManifestBuildError(f"spec does not exist: {path}")
    payload = _read_json(path, label="build spec")
    specification = _object(
        payload,
        label="build spec",
        required=(
            "schema",
            "expected_train_seeds",
            "baseline_experiment",
            "strict_source_manifests",
            "experiments",
        ),
        optional=("bootstrap",),
    )
    if specification["schema"] != SPEC_SCHEMA:
        raise PaperManifestBuildError(f"build spec schema must be exactly {SPEC_SCHEMA!r}")
    raw_seeds = specification["expected_train_seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise PaperManifestBuildError("expected_train_seeds must be a non-empty list")
    expected_seeds = [
        _required_int(seed, label=f"expected_train_seeds[{index}]")
        for index, seed in enumerate(raw_seeds)
    ]
    if len(set(expected_seeds)) != len(expected_seeds):
        raise PaperManifestBuildError("expected_train_seeds contains duplicates")
    baseline_id = _nonempty_string(
        specification["baseline_experiment"], label="baseline_experiment"
    )
    spec_base = path.parent
    protocol_tn, loaded_tn = _load_source_manifests(
        specification["strict_source_manifests"], spec_base=spec_base
    )

    raw_experiments = specification["experiments"]
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise PaperManifestBuildError("experiments must be a non-empty list")
    output_experiments = []
    experiment_ids = set()
    checkpoint_to_seed: Dict[Path, int] = {}
    headline_release_evaluations: list[Mapping[str, Any]] = []
    for experiment_index, raw_experiment in enumerate(raw_experiments):
        experiment = _object(
            raw_experiment,
            label=f"experiments[{experiment_index}]",
            required=("id", "runs"),
            optional=("label", "expected_train_seeds", "reference_role"),
        )
        experiment_id = _nonempty_string(
            experiment["id"], label=f"experiments[{experiment_index}].id"
        )
        if experiment_id in experiment_ids:
            raise PaperManifestBuildError(f"duplicate experiment id {experiment_id!r}")
        experiment_ids.add(experiment_id)
        raw_experiment_seeds = experiment.get(
            "expected_train_seeds", expected_seeds
        )
        if not isinstance(raw_experiment_seeds, list) or not raw_experiment_seeds:
            raise PaperManifestBuildError(
                f"experiment {experiment_id}: expected_train_seeds must be non-empty"
            )
        experiment_seeds = [
            _required_int(
                value,
                label=f"experiment {experiment_id}.expected_train_seeds[{index}]",
            )
            for index, value in enumerate(raw_experiment_seeds)
        ]
        if len(set(experiment_seeds)) != len(experiment_seeds):
            raise PaperManifestBuildError(
                f"experiment {experiment_id}: expected_train_seeds contains duplicates"
            )
        experiment_seed_set = set(experiment_seeds)
        reference_role = str(
            experiment.get("reference_role") or "training_seed_distribution"
        )
        if reference_role not in {
            "training_seed_distribution",
            "fixed_historical_checkpoint",
        }:
            raise PaperManifestBuildError(
                f"experiment {experiment_id}: invalid reference_role {reference_role!r}"
            )
        if reference_role == "fixed_historical_checkpoint" and (
            experiment_id != baseline_id or len(experiment_seeds) != 1
        ):
            raise PaperManifestBuildError(
                "fixed_historical_checkpoint is allowed only for the declared "
                "baseline with exactly one real training seed"
            )
        raw_runs = experiment["runs"]
        if not isinstance(raw_runs, list) or not raw_runs:
            raise PaperManifestBuildError(f"experiment {experiment_id}: runs must be non-empty")
        output_runs = []
        seen_seeds = set()
        seen_checkpoints = set()
        for run_index, raw_run in enumerate(raw_runs):
            run_label = f"experiment {experiment_id} run {run_index}"
            run = _object(
                raw_run,
                label=run_label,
                required=("train_seed",),
                optional=(
                    "evaluation_root",
                    "expected_training_run_id",
                    "checkpoint",
                    "config",
                    "data",
                    "ref_eval_root",
                    "strict2031_eval_root",
                    "strict1607_eval_root",
                    "run_id",
                ),
            )
            train_seed = _required_int(run["train_seed"], label=f"{run_label}.train_seed")
            if train_seed not in experiment_seed_set:
                raise PaperManifestBuildError(
                    f"{run_label}: unexpected train seed {train_seed}; "
                    f"expected {experiment_seeds}"
                )
            if train_seed in seen_seeds:
                raise PaperManifestBuildError(
                    f"experiment {experiment_id}: duplicate train seed {train_seed}"
                )
            seen_seeds.add(train_seed)
            explicit_fields = {
                "checkpoint",
                "config",
                "data",
                "ref_eval_root",
                "strict2031_eval_root",
                "strict1607_eval_root",
            }
            if "evaluation_root" in run:
                mixed = sorted(explicit_fields.intersection(run))
                if mixed:
                    raise PaperManifestBuildError(
                        f"{run_label}: evaluation_root cannot be mixed with {mixed}"
                    )
                expected_training_run_id = _nonempty_string(
                    run.get("expected_training_run_id"),
                    label=f"{run_label}.expected_training_run_id",
                )
                normalized_run = {
                    **_derive_run_from_evaluation_root(
                        run["evaluation_root"],
                        spec_base=spec_base,
                        train_seed=train_seed,
                        expected_training_run_id=expected_training_run_id,
                        label=run_label,
                    ),
                    **({"run_id": run["run_id"]} if "run_id" in run else {}),
                }
                release_evidence = normalized_run.pop(
                    "_headline_release_evidence", None
                )
                if release_evidence is not None:
                    if not isinstance(release_evidence, Mapping):
                        raise PaperManifestBuildError(
                            f"{run_label}: headline release evidence is invalid"
                        )
                    headline_release_evaluations.append(release_evidence)
            else:
                if experiment_id != baseline_id:
                    raise PaperManifestBuildError(
                        f"{run_label}: explicit artifact mode is restricted to "
                        "the declared historical baseline experiment"
                    )
                if "expected_training_run_id" in run:
                    raise PaperManifestBuildError(
                        f"{run_label}: expected_training_run_id requires evaluation_root"
                    )
                missing_explicit = sorted(explicit_fields - set(run))
                if missing_explicit:
                    raise PaperManifestBuildError(
                        f"{run_label}: explicit mode is missing fields {missing_explicit}"
                    )
                normalized_run = dict(run)
            checkpoint_record = _file_record(
                normalized_run["checkpoint"], spec_base, label=f"{run_label}.checkpoint"
            )
            checkpoint_path = Path(checkpoint_record["path"])
            if checkpoint_path in seen_checkpoints:
                raise PaperManifestBuildError(
                    f"experiment {experiment_id}: duplicate checkpoint path {checkpoint_path}"
                )
            seen_checkpoints.add(checkpoint_path)
            prior_seed = checkpoint_to_seed.get(checkpoint_path)
            if prior_seed is not None and prior_seed != train_seed:
                raise PaperManifestBuildError(
                    f"checkpoint {checkpoint_path} is assigned to train seeds "
                    f"{prior_seed} and {train_seed}"
                )
            checkpoint_to_seed[checkpoint_path] = train_seed
            config_record = _file_record(
                normalized_run["config"], spec_base, label=f"{run_label}.config"
            )
            raw_data = normalized_run["data"]
            if not isinstance(raw_data, list) or not raw_data:
                raise PaperManifestBuildError(f"{run_label}.data must be a non-empty list")
            data_records = [
                _file_record(item, spec_base, label=f"{run_label}.data[{index}]")
                for index, item in enumerate(raw_data)
            ]
            data_paths = [record["path"] for record in data_records]
            if len(set(data_paths)) != len(data_paths):
                raise PaperManifestBuildError(f"{run_label}.data contains duplicate paths")
            run_selector = (
                _nonempty_string(
                    normalized_run["run_id"], label=f"{run_label}.run_id"
                )
                if "run_id" in normalized_run
                else None
            )
            roots = {
                "ref": _resolve_path(normalized_run["ref_eval_root"], spec_base, label=f"{run_label}.ref_eval_root"),
                "strict2031": _resolve_path(
                    normalized_run["strict2031_eval_root"], spec_base, label=f"{run_label}.strict2031_eval_root"
                ),
                "strict1607": _resolve_path(
                    normalized_run["strict1607_eval_root"], spec_base, label=f"{run_label}.strict1607_eval_root"
                ),
            }
            for root_name, root in roots.items():
                if not root.is_dir():
                    raise PaperManifestBuildError(
                        f"{run_label}.{root_name}_eval_root is not a directory: {root}"
                    )
            ref_result, selected_run_id = _build_ref_result(
                roots["ref"],
                checkpoint=checkpoint_path,
                run_selector=run_selector,
                spec_base=spec_base,
                label=f"{run_label}.ref",
            )
            tn_results = {
                split: _build_tn_result(
                    roots[split],
                    split=split,
                    checkpoint=checkpoint_path,
                    run_selector=run_selector,
                    expected_run_id=selected_run_id,
                    spec_base=spec_base,
                    source_manifest=loaded_tn[split],
                    label=f"{run_label}.{split}",
                )
                for split in TN_SPLITS
            }
            output_runs.append(
                {
                    "train_seed": train_seed,
                    **(
                        {
                            "training_run_id": normalized_run["training_run_id"],
                            "evaluation_root": normalized_run["evaluation_root"],
                        }
                        if "training_run_id" in normalized_run
                        else {}
                    ),
                    "artifacts": {
                        "checkpoint": checkpoint_record,
                        "config": config_record,
                        "data": data_records,
                    },
                    "results": {"ref": ref_result, "tn": tn_results},
                }
            )
        missing = sorted(experiment_seed_set - seen_seeds)
        extra = sorted(seen_seeds - experiment_seed_set)
        if missing or extra:
            raise PaperManifestBuildError(
                f"experiment {experiment_id}: seed contract mismatch; "
                f"missing={missing}, extra={extra}"
            )
        output_experiment: Dict[str, Any] = {
            "id": experiment_id,
            "label": str(experiment.get("label") or experiment_id),
            "expected_train_seeds": experiment_seeds,
            "reference_role": reference_role,
            "runs": output_runs,
        }
        output_experiments.append(output_experiment)
    if baseline_id not in experiment_ids:
        raise PaperManifestBuildError(
            f"baseline_experiment {baseline_id!r} is not a declared experiment"
        )
    if headline_release_evaluations:
        from tools import stageb_headline_release_contract as headline_release

        try:
            headline_provenance = headline_release.build_release_provenance(
                headline_release_evaluations,
                bootstrap=_bootstrap_spec(specification.get("bootstrap")),
            )
        except headline_release.HeadlineReleaseError as error:
            raise PaperManifestBuildError(
                f"headline release provenance is incomplete: {error}"
            ) from error
    else:
        headline_provenance = {
            "schema": "stageb-headline-release-provenance-v1",
            "status": "unverified_legacy_manifest",
            "reason": (
                "no one-time selection/gate/parity/consumption receipt chain"
            ),
        }
    return {
        "schema": OUTPUT_SCHEMA,
        "expected_train_seeds": expected_seeds,
        "baseline_experiment": baseline_id,
        "protocol": {
            "ref_splits": list(REF_SPLITS),
            "tn_splits": protocol_tn,
            "bootstrap": _bootstrap_spec(specification.get("bootstrap")),
            "headline_release_provenance": headline_provenance,
        },
        "experiments": output_experiments,
    }


def validate_manifest(payload: Mapping[str, Any], *, bootstrap_iterations: int = 8) -> None:
    iterations = _required_int(
        bootstrap_iterations, label="validation bootstrap iterations"
    )
    if iterations <= 0:
        raise PaperManifestBuildError("validation bootstrap iterations must be positive")
    from tools.aggregate_stageb_paper_results import (  # noqa: WPS433
        PaperAggregationError,
        aggregate_manifest,
    )

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="stageb-paper-manifest-validation-",
            suffix=".json",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary_path = Path(handle.name)
        report = aggregate_manifest(
            temporary_path,
            allow_incomplete=False,
            bootstrap_iterations=iterations,
        )
        if report.get("status") != "complete" or not report.get("validation", {}).get("pass"):
            raise PaperManifestBuildError("aggregator validation did not return complete/pass")
    except (PaperAggregationError, RecordComparisonError, OSError, ValueError) as error:
        if isinstance(error, PaperManifestBuildError):
            raise
        raise PaperManifestBuildError(f"aggregator validation failed: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` while refusing an existing destination."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:  # pragma: no cover - supported Linux runtime
        raise PaperManifestBuildError(
            "atomic no-replace output requires Linux renameat2"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise PaperManifestBuildError(
            f"refusing to overwrite existing output: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def write_manifest_new(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PaperManifestBuildError(f"refusing to overwrite existing output: {path}")
    rendered = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help=f"{SPEC_SCHEMA} JSON file")
    parser.add_argument("--output", help="New manifest path; existing files are rejected")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the derived manifest and do not create --output.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the existing aggregator contract before writing/printing.",
    )
    parser.add_argument(
        "--validation-bootstrap-iterations",
        type=int,
        default=8,
        help="Small bootstrap count used only by --validate (default: 8).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.dry_run) == bool(args.output is not None):
        print("ERROR: choose exactly one of --dry-run or --output", file=sys.stderr)
        return 2
    if args.output is not None and Path(args.output).expanduser().resolve().exists():
        print(
            f"ERROR: refusing to overwrite existing output: "
            f"{Path(args.output).expanduser().resolve()}",
            file=sys.stderr,
        )
        return 2
    try:
        payload = build_manifest(args.spec)
        if args.validate:
            validate_manifest(
                payload,
                bootstrap_iterations=args.validation_bootstrap_iterations,
            )
        if args.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        else:
            output = write_manifest_new(payload, args.output)
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "output": str(output),
                        "experiments": len(payload["experiments"]),
                        "expected_train_seeds": payload["expected_train_seeds"],
                        "validated": bool(args.validate),
                    },
                    sort_keys=True,
                )
            )
    except (PaperManifestBuildError, RecordComparisonError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
