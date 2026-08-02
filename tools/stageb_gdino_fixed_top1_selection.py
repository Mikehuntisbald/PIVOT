#!/usr/bin/env python3
"""Leak-free image holdout and milestone selection for fixed-top1 Stage-B S."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_stageb_fpr95_records import (  # noqa: E402
    RecordComparisonError,
    compare_records,
    exact_fpr95,
    load_manifest,
    load_tn_records,
)
from tools.stageb_dependency_audit import (  # noqa: E402
    DependencyAuditError,
    config_import_chain,
    local_python_dependency_paths,
)


PARTITION_SCHEMA = "stageb-gdino-fixed-top1-image-partition-v1"
SELECTION_SCHEMA = "stageb-gdino-fixed-top1-milestone-selection-v1"
CALIBRATION_COMPLETION_SCHEMA = "stageb-gdino-fixed-top1-calibration-eval-v1"
FIXED_TOP1_SCHEMA = "stageb-gdino-adapter-fixed-top1-confidence-probe-v1"
P0_SCHEMA = "stageb-gdino-adapter-p0-v1"
STRICT2031_SHA256 = "0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918"
STRICT1607_SHA256 = "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25"
DEFAULT_PARTITION_SEED = 20260712
DEFAULT_SALT_CANDIDATES = 256
DEFAULT_DATA_ROOT = Path("/home/user/datasets/pivot_data")
DEFAULT_SOURCE_IMAGE_ROOT = DEFAULT_DATA_ROOT / "COCO/coco2014/train2014"
BASE_CALIBRATION_NUMERATOR = 1
BASE_CALIBRATION_DENOMINATOR = 10
MIN_CALIBRATION_ROWS = 1_000
MIN_CALIBRATION_IMAGES = 500
MIN_CALIBRATION_SOURCE_ROWS = 250
MIN_STRATUM_SUPPORT = 100
MAX_STRATUM_RATE_DEVIATION = Fraction(1, 20)
MILESTONES = (50, 100, 250, 500, 1000)
GLOBAL_BATCH = 8
BOOTSTRAP_ITERATIONS = 5_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260712
PROMOTION_PROBABILITY = 0.95
STRICT_SCORE_TOKENS = ("strict",)
CALIBRATION_CODE_ENTRIES = (
    "tools/eval_stageb_gdino_fixed_top1_calibration.py",
    "tools/eval_text_groundingdino_refcoco_tn.py",
)
CALIBRATION_CODE_INCLUDE = (
    "tools/stageb_gdino_fixed_top1_selection.py",
    "tools/stageb_gdino_fixed_top1_probe_audit.py",
    "tools/compare_stageb_fpr95_records.py",
    "tools/stageb_eval_records.py",
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "models/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/stage_b_gdino_score_adapter.py",
)
CALIBRATION_RUNTIME = {
    "image_set": "val",
    "resize_short_side": 800,
    "max_size": 1333,
    "batch_size": 16,
    "amp": True,
    "forward_order": "negative_then_positive_separate_calls",
    "score": "max_over_900_query_confidence_score",
    "threshold_tpr": 0.95,
    "seed": 42,
    "gflops_debug_shilong": "forbidden",
}
CALIBRATION_SCORE_CONTRACT = {
    "model_output": "stage_b_gdino_confidence_score",
    "query_reduction": "max",
    "query_count": 900,
    "positive_negative_forward": "separate_negative_then_positive_calls",
    "selection_metric": "exact_global_fpr_at_95tpr_from_per_example_records",
}


class SelectionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SelectionError(f"required file is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def calibration_code_records() -> list[Dict[str, Any]]:
    try:
        paths = local_python_dependency_paths(
            CALIBRATION_CODE_ENTRIES,
            root=REPO_ROOT,
            include=CALIBRATION_CODE_INCLUDE,
        )
    except DependencyAuditError as error:
        raise SelectionError(str(error)) from error
    return [file_record(path) for path in paths]


def calibration_config_records(config: Path) -> list[Dict[str, Any]]:
    try:
        paths = config_import_chain(config.resolve(), root=REPO_ROOT)
    except DependencyAuditError as error:
        raise SelectionError(str(error)) from error
    return [file_record(path) for path in paths]


def _checkpoint_run_prefix(checkpoint: Path) -> str:
    raw = f"{checkpoint.parent.name}_{checkpoint.stem}" if checkpoint.parent.name else checkpoint.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "run"


def replay_calibration_checkpoint_verification(
    *,
    role: str,
    checkpoint: Path,
    checkpoint_audit: Path,
    baseline_checkpoint: Path | None,
    milestone_iteration: int | None,
) -> Dict[str, Any]:
    try:
        if role == "p0":
            if baseline_checkpoint is None or milestone_iteration is not None:
                raise SelectionError("P0 verification arguments are inconsistent")
            from tools.make_stageb_gdino_adapter_p0 import (
                DEFAULT_CONFIG as P0_CONFIG,
                verify_p0,
                verify_p0_sidecar,
            )

            config = Path(P0_CONFIG)
            if not config.is_absolute():
                config = REPO_ROOT / config
            audit = verify_p0(
                baseline_checkpoint=baseline_checkpoint.resolve(),
                p0_checkpoint=checkpoint.resolve(),
                config=config.resolve(),
            )
            verified = verify_p0_sidecar(
                p0_checkpoint=checkpoint.resolve(),
                audit=audit,
                sidecar=checkpoint_audit.resolve(),
            )
            return {
                "schema": CALIBRATION_COMPLETION_SCHEMA,
                "role": "p0",
                "p0": verified,
                "functional_identity": dict(audit["functional_identity"]),
            }
        if role != "milestone" or baseline_checkpoint is not None or milestone_iteration is None:
            raise SelectionError("milestone verification arguments are inconsistent")
        from tools.stageb_gdino_fixed_top1_probe_audit import (
            verify_calibration_checkpoint,
        )

        verified = verify_calibration_checkpoint(
            checkpoint_path=checkpoint.resolve(),
            audit_path=checkpoint_audit.resolve(),
            expected_iteration=int(milestone_iteration),
        )
        return {
            "schema": CALIBRATION_COMPLETION_SCHEMA,
            "role": "milestone",
            "milestone": verified,
        }
    except SelectionError:
        raise
    except Exception as error:
        raise SelectionError(f"calibration checkpoint deep replay failed: {error}") from error


def _same_file_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "size_bytes", "sha256"))


def _current_file_record(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        raise SelectionError(f"{label} has no file record")
    current = file_record(Path(str(value["path"])))
    if not _same_file_record(value, current):
        raise SelectionError(f"{label} file identity drifted")
    return current


def _read_bound_json(record: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"could not read bound JSON {label}: {error}") from error
    observed = {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if not _same_file_record(record, observed):
        raise SelectionError(f"{label} changed while it was being consumed")
    if not isinstance(value, dict):
        raise SelectionError(f"expected a bound JSON object for {label}")
    return value


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionError(f"expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SelectionError(f"blank JSONL row at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SelectionError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise SelectionError(f"non-object JSONL row at {path}:{line_number}")
            yield line_number, value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), sort_keys=True, ensure_ascii=True, allow_nan=False)
                )
                handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonl_record(path: Path, rows: int, images: int) -> Dict[str, Any]:
    value = file_record(path)
    value.update({"rows": int(rows), "unique_images": int(images)})
    return value


def _normalized_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    elif value is None:
        raw = []
    else:
        raise SelectionError("replace_category must be a string, list, or null")
    tags = []
    for item in raw:
        tag = " ".join(str(item or "").strip().lower().split())
        if tag and tag not in tags:
            tags.append(tag)
    return tuple(tags or ["unknown"])


def _row_identity(row: Mapping[str, Any], *, context: str) -> tuple[str, int]:
    sample_id = str(row.get("sample_id", "")).strip()
    if not sample_id:
        raise SelectionError(f"{context} has no sample_id")
    try:
        image_id = int(row["image_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise SelectionError(f"{context} has invalid image_id") from error
    return sample_id, image_id


def _resolved_source_image(
    row: Mapping[str, Any], *, context: str, require_exists: bool = False
) -> str:
    image_path = row.get("image_path")
    path: Path | None = None
    if isinstance(image_path, str) and image_path.strip():
        candidate = Path(image_path).expanduser()
        if candidate.exists():
            path = candidate
    if path is None and row.get("image_id") is not None:
        try:
            path = DEFAULT_SOURCE_IMAGE_ROOT / f"COCO_train2014_{int(row['image_id']):012d}.jpg"
        except (TypeError, ValueError):
            path = None
    if path is None:
        file_name = row.get("file_name", row.get("filename"))
        if isinstance(file_name, str) and file_name.strip():
            path = DEFAULT_SOURCE_IMAGE_ROOT / Path(file_name).name
    if path is None:
        raise SelectionError(f"{context} has no loader-resolvable source image")
    path = path.resolve()
    if require_exists and not path.is_file():
        raise SelectionError(f"{context} loader-resolved source image is missing: {path}")
    return str(path)


def _image_identity_keys(
    rows: Sequence[Mapping[str, Any]], *, context: str, require_exists: bool = False
) -> tuple[list[str], Dict[str, Any]]:
    keys: list[str] = []
    id_to_path: Dict[int, str] = {}
    path_to_id: Dict[str, int] = {}
    for index, row in enumerate(rows):
        _sample_id, image_id = _row_identity(row, context=f"{context} row {index + 1}")
        path = _resolved_source_image(
            row,
            context=f"{context} row {index + 1}",
            require_exists=require_exists,
        )
        previous_path = id_to_path.setdefault(image_id, path)
        if previous_path != path:
            raise SelectionError(
                f"{context} image_id {image_id} aliases multiple source image paths"
            )
        previous_id = path_to_id.setdefault(path, image_id)
        if previous_id != image_id:
            raise SelectionError(
                f"{context} source image path aliases image_ids {previous_id} and {image_id}"
            )
        keys.append(path)
    rendered = json.dumps(
        [[image_id, path] for image_id, path in sorted(id_to_path.items())],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return keys, {
        "unique_image_ids": len(id_to_path),
        "unique_resolved_paths": len(path_to_id),
        "image_id_resolved_path_bijection": True,
        "mapping_sha256": hashlib.sha256(rendered).hexdigest(),
    }


def _row_strata(row: Mapping[str, Any]) -> tuple[str, ...]:
    dataset = " ".join(str(row.get("dataset", "")).strip().lower().split())
    source = " ".join(str(row.get("pair_source", "")).strip().lower().split())
    if not dataset or not source:
        raise SelectionError("partition row has no dataset or pair_source")
    values = [f"dataset:{dataset}", f"pair_source:{source}"]
    values.extend(f"replace_category:{tag}" for tag in _normalized_tags(row.get("replace_category")))
    return tuple(values)


def _load_strict_images(
    path: Path, *, label: str, expected_sha256: str, expected_rows: int
) -> tuple[set[int], set[str], Dict[str, Any]]:
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise SelectionError(f"{label} SHA-256 mismatch")
    images: set[int] = set()
    rows = 0
    parsed_rows = []
    for line_number, row in _iter_jsonl(path):
        try:
            images.add(int(row["image_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise SelectionError(f"{label} row {line_number} has invalid image_id") from error
        rows += 1
        parsed_rows.append(row)
    if rows != int(expected_rows):
        raise SelectionError(f"{label} rows mismatch: {rows} != {expected_rows}")
    image_keys, identity = _image_identity_keys(
        parsed_rows, context=label, require_exists=True
    )
    image_paths = set(image_keys)
    record.update(
        {
            "rows": rows,
            "unique_images": len(images),
            "image_identity": identity,
        }
    )
    return images, image_paths, record


def _hash_assignment(*, seed: int, salt: int, image_key: str, numerator: int, denominator: int) -> bool:
    digest = hashlib.sha256(
        f"{int(seed)}\0{int(salt)}\0{image_key}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value * int(denominator) < int(numerator) * (1 << 64)


def _deviation(selected: int, total: int, numerator: int, denominator: int) -> Fraction:
    if total <= 0:
        raise SelectionError("cannot compute a partition deviation over zero rows")
    return Fraction(abs(int(selected) * int(denominator) - int(total) * int(numerator)), int(total) * int(denominator))


def _fraction_payload(value: Fraction) -> Dict[str, Any]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "float": float(value),
    }


def _choose_partition(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    salt_candidates: int,
    image_keys: Sequence[str] | None = None,
) -> Dict[str, Any]:
    if int(salt_candidates) <= 0:
        raise SelectionError("salt_candidates must be positive")
    if image_keys is None:
        image_keys, _identity = _image_identity_keys(rows, context="accepted")
    if len(image_keys) != len(rows):
        raise SelectionError("partition image identity count differs from accepted rows")
    groups: Dict[str, list[int]] = defaultdict(list)
    sample_ids: set[str] = set()
    row_strata: list[tuple[str, ...]] = []
    stratum_totals: Counter[str] = Counter()
    for index, row in enumerate(rows):
        sample_id, image_id = _row_identity(row, context=f"accepted row {index + 1}")
        if sample_id in sample_ids:
            raise SelectionError(f"duplicate accepted sample_id: {sample_id}")
        sample_ids.add(sample_id)
        groups[str(image_keys[index])].append(index)
        labels = _row_strata(row)
        row_strata.append(labels)
        stratum_totals.update(labels)
    total_rows = len(rows)
    if total_rows <= MIN_CALIBRATION_ROWS:
        raise SelectionError(
            f"accepted rows must exceed the {MIN_CALIBRATION_ROWS}-row calibration floor"
        )
    desired_rows = max(
        int(math.ceil(total_rows * BASE_CALIBRATION_NUMERATOR / BASE_CALIBRATION_DENOMINATOR)),
        MIN_CALIBRATION_ROWS,
    )
    numerator = desired_rows
    denominator = total_rows
    eligible_strata = {
        key: count for key, count in stratum_totals.items() if count >= MIN_STRATUM_SUPPORT
    }
    dataset_strata = sorted(
        key for key in stratum_totals if key.startswith("dataset:")
    )
    candidates = []
    for salt in range(int(salt_candidates)):
        calibration_images = {
            image_key
            for image_key in groups
            if _hash_assignment(
                seed=seed,
                salt=salt,
                image_key=image_key,
                numerator=numerator,
                denominator=denominator,
            )
        }
        calibration_indices = [
            index for image_key in calibration_images for index in groups[image_key]
        ]
        selected_strata: Counter[str] = Counter()
        for index in calibration_indices:
            selected_strata.update(row_strata[index])
        deviations = {
            key: _deviation(selected_strata[key], total, numerator, denominator)
            for key, total in eligible_strata.items()
        }
        row_deviation = _deviation(
            len(calibration_indices), total_rows, numerator, denominator
        )
        image_deviation = _deviation(
            len(calibration_images), len(groups), numerator, denominator
        )
        all_deviations = list(deviations.values()) + [row_deviation, image_deviation]
        max_deviation = max(all_deviations)
        mean_deviation = sum(all_deviations, Fraction(0, 1)) / len(all_deviations)
        readiness_shortfall = max(0, desired_rows - len(calibration_indices))
        readiness_shortfall += max(
            0, MIN_CALIBRATION_IMAGES - len(calibration_images)
        )
        readiness_shortfall += sum(
            max(0, MIN_CALIBRATION_SOURCE_ROWS - selected_strata[key])
            for key in dataset_strata
        )
        candidates.append(
            (
                (
                    int(readiness_shortfall > 0),
                    int(readiness_shortfall),
                    max_deviation,
                    mean_deviation,
                    row_deviation,
                    image_deviation,
                    int(salt),
                ),
                calibration_images,
                selected_strata,
                deviations,
            )
        )
    candidates.sort(key=lambda item: item[0])
    score, calibration_images, selected_strata, deviations = candidates[0]
    train_indices = [index for index, key in enumerate(image_keys) if str(key) not in calibration_images]
    calibration_indices = [index for index, key in enumerate(image_keys) if str(key) in calibration_images]
    train_rows = [rows[index] for index in train_indices]
    calibration_rows = [rows[index] for index in calibration_indices]
    train_images = {str(image_keys[index]) for index in train_indices}
    source_counts: Counter[str] = Counter(
        " ".join(str(row.get("dataset", "")).strip().lower().split())
        for row in calibration_rows
    )
    readiness_errors = []
    if len(calibration_rows) < desired_rows:
        readiness_errors.append(
            f"calibration rows {len(calibration_rows)} < adaptive target {desired_rows}"
        )
    if len(calibration_images) < MIN_CALIBRATION_IMAGES:
        readiness_errors.append(
            f"calibration images {len(calibration_images)} < {MIN_CALIBRATION_IMAGES}"
        )
    expected_sources = sorted(
        key.removeprefix("dataset:")
        for key in stratum_totals
        if key.startswith("dataset:")
    )
    for source in expected_sources:
        if source_counts[source] < MIN_CALIBRATION_SOURCE_ROWS:
            readiness_errors.append(
                f"calibration source {source} rows {source_counts[source]} "
                f"< {MIN_CALIBRATION_SOURCE_ROWS}"
            )
    if score[2] > MAX_STRATUM_RATE_DEVIATION:
        readiness_errors.append(
            "best deterministic salt exceeds the maximum stratum-rate deviation"
        )
    supported = [target for target in MILESTONES if len(train_rows) >= GLOBAL_BATCH * target]
    if not supported:
        readiness_errors.append(
            f"train rows {len(train_rows)} cannot reach minimum milestone S{MILESTONES[0]}"
        )
    recommended = max(supported) if supported and not readiness_errors else None
    return {
        "train_rows": train_rows,
        "calibration_rows": calibration_rows,
        "train_images": train_images,
        "calibration_images": set(calibration_images),
        "policy": {
            "schema": "stageb-gdino-fixed-top1-image-hash-policy-v1",
            "group_key": "resolved_source_image_path",
            "seed": int(seed),
            "salt_candidates": int(salt_candidates),
            "hash": "sha256(seed\\0salt\\0resolved_source_image_path);big_endian_u64_prefix",
            "source_image_root": str(DEFAULT_SOURCE_IMAGE_ROOT.resolve()),
            "base_ratio": {
                "numerator": BASE_CALIBRATION_NUMERATOR,
                "denominator": BASE_CALIBRATION_DENOMINATOR,
            },
            "desired_calibration_rows": int(desired_rows),
            "threshold": {"numerator": int(numerator), "denominator": int(denominator)},
            "effective_target_ratio": float(Fraction(numerator, denominator)),
            "minimum_calibration_rows": MIN_CALIBRATION_ROWS,
            "minimum_calibration_images": MIN_CALIBRATION_IMAGES,
            "minimum_calibration_rows_per_dataset": MIN_CALIBRATION_SOURCE_ROWS,
            "minimum_stratum_support": MIN_STRATUM_SUPPORT,
            "strata": ["dataset", "pair_source", "replace_category_multilabel"],
            "salt_selection": (
                "lexicographic(min_readiness_failure,min_readiness_shortfall,"
                "min_max_rate_deviation,min_mean_rate_deviation,"
                "min_row_rate_deviation,min_image_rate_deviation,min_salt)"
            ),
            "selected_salt": int(score[6]),
            "selected_score": {
                "readiness_shortfall": int(score[1]),
                "max_rate_deviation": _fraction_payload(score[2]),
                "mean_rate_deviation": _fraction_payload(score[3]),
                "row_rate_deviation": _fraction_payload(score[4]),
                "image_rate_deviation": _fraction_payload(score[5]),
            },
        },
        "strata": {
            key: {
                "all_rows": int(total),
                "calibration_rows": int(selected_strata[key]),
                "calibration_rate": float(Fraction(selected_strata[key], total)),
                "target_rate_deviation": _fraction_payload(deviations[key]),
            }
            for key, total in sorted(eligible_strata.items())
        },
        "calibration_dataset_rows": dict(sorted(source_counts.items())),
        "readiness_errors": readiness_errors,
        "recommended_max_target": recommended,
    }


def _partition_payload(
    *,
    accepted_path: Path,
    verification_audit_path: Path,
    strict2031_path: Path,
    strict1607_path: Path,
    train_path: Path,
    calibration_path: Path,
    seed: int,
    salt_candidates: int,
    require_output_match: bool,
) -> Dict[str, Any]:
    accepted_rows = [row for _line, row in _iter_jsonl(accepted_path)]
    accepted_image_keys, accepted_identity = _image_identity_keys(
        accepted_rows, context="accepted", require_exists=True
    )
    split = _choose_partition(
        accepted_rows,
        seed=seed,
        salt_candidates=salt_candidates,
        image_keys=accepted_image_keys,
    )
    if require_output_match:
        actual_train = [row for _line, row in _iter_jsonl(train_path)]
        actual_calibration = [row for _line, row in _iter_jsonl(calibration_path)]
        if actual_train != split["train_rows"]:
            raise SelectionError("train partition rows drifted from deterministic replay")
        if actual_calibration != split["calibration_rows"]:
            raise SelectionError("calibration partition rows drifted from deterministic replay")
    strict2031_images, strict2031_paths, strict2031_record = _load_strict_images(
        strict2031_path,
        label="strict2031",
        expected_sha256=STRICT2031_SHA256,
        expected_rows=2031,
    )
    strict1607_images, strict1607_paths, strict1607_record = _load_strict_images(
        strict1607_path,
        label="strict1607",
        expected_sha256=STRICT1607_SHA256,
        expected_rows=1607,
    )
    accepted_images = {int(row["image_id"]) for row in accepted_rows}
    accepted_paths = set(accepted_image_keys)
    train_images = split["train_images"]
    calibration_images = split["calibration_images"]
    train_image_ids = {int(row["image_id"]) for row in split["train_rows"]}
    calibration_image_ids = {
        int(row["image_id"]) for row in split["calibration_rows"]
    }
    overlaps = {
        "accepted_strict2031_image_id": len(accepted_images & strict2031_images),
        "accepted_strict1607_image_id": len(accepted_images & strict1607_images),
        "accepted_strict2031_resolved_path": len(accepted_paths & strict2031_paths),
        "accepted_strict1607_resolved_path": len(accepted_paths & strict1607_paths),
        "train_calibration_image_id": len(train_image_ids & calibration_image_ids),
        "train_calibration_resolved_path": len(train_images & calibration_images),
        "train_strict2031_image_id": len(train_image_ids & strict2031_images),
        "train_strict1607_image_id": len(train_image_ids & strict1607_images),
        "train_strict2031_resolved_path": len(train_images & strict2031_paths),
        "train_strict1607_resolved_path": len(train_images & strict1607_paths),
        "calibration_strict2031_image_id": len(
            calibration_image_ids & strict2031_images
        ),
        "calibration_strict1607_image_id": len(
            calibration_image_ids & strict1607_images
        ),
        "calibration_strict2031_resolved_path": len(
            calibration_images & strict2031_paths
        ),
        "calibration_strict1607_resolved_path": len(
            calibration_images & strict1607_paths
        ),
    }
    if any(overlaps.values()):
        raise SelectionError(f"partition image-disjointness failed: {overlaps}")
    train_ids = {str(row["sample_id"]) for row in split["train_rows"]}
    calibration_ids = {str(row["sample_id"]) for row in split["calibration_rows"]}
    accepted_ids = [str(row["sample_id"]) for row in accepted_rows]
    if train_ids & calibration_ids or train_ids | calibration_ids != set(accepted_ids):
        raise SelectionError("partition sample union/disjointness failed")
    verification_record = file_record(verification_audit_path)
    return {
        "schema": PARTITION_SCHEMA,
        "kind": "completed_fixed_top1_image_partition",
        "inputs": {
            "accepted": _jsonl_record(accepted_path, len(accepted_rows), len(accepted_images)),
            "verification_audit": verification_record,
            "strict_holdout_manifests": {
                "strict2031": strict2031_record,
                "strict1607": strict1607_record,
            },
        },
        "outputs": {
            "train": _jsonl_record(train_path, len(split["train_rows"]), len(train_images)),
            "calibration": _jsonl_record(
                calibration_path,
                len(split["calibration_rows"]),
                len(calibration_images),
            ),
        },
        "policy": split["policy"],
        "accepted_image_identity": accepted_identity,
        "strata": split["strata"],
        "calibration_dataset_rows": split["calibration_dataset_rows"],
        "partition_contract": {
            "source_rows_are_byte_equivalent_json_objects": True,
            "source_order_preserved_within_each_output": True,
            "sample_union_exact": True,
            "sample_disjoint": True,
            "image_disjoint": True,
            "image_id_resolved_path_bijection": True,
            "model_scores_used_for_partition": False,
            "strict_scores_or_results_used_for_partition": False,
        },
        "image_overlaps": overlaps,
        "selection_readiness": {
            "pass": not split["readiness_errors"],
            "errors": list(split["readiness_errors"]),
            "calibration_rows": len(split["calibration_rows"]),
            "calibration_images": len(calibration_images),
        },
        "recommended_max_target": split["recommended_max_target"],
        "milestone_data_requirements": {
            str(target): GLOBAL_BATCH * target for target in MILESTONES
        },
    }


def create_partition(
    *,
    accepted_path: Path,
    verification_audit_path: Path,
    strict2031_path: Path,
    strict1607_path: Path,
    train_path: Path,
    calibration_path: Path,
    audit_path: Path,
    seed: int = DEFAULT_PARTITION_SEED,
    salt_candidates: int = DEFAULT_SALT_CANDIDATES,
    source_validator=None,
) -> Dict[str, Any]:
    outputs = (train_path, calibration_path, audit_path)
    if int(seed) != DEFAULT_PARTITION_SEED or int(salt_candidates) != DEFAULT_SALT_CANDIDATES:
        raise SelectionError(
            "production partition requires the pre-registered seed and salt candidate count"
        )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SelectionError(f"refusing to overwrite partition outputs: {existing}")
    if source_validator is not None:
        source_validator(accepted_path.resolve(), verification_audit_path.resolve())
    rows = [row for _line, row in _iter_jsonl(accepted_path)]
    split = _choose_partition(rows, seed=seed, salt_candidates=salt_candidates)
    try:
        _atomic_write_jsonl(train_path, split["train_rows"])
        _atomic_write_jsonl(calibration_path, split["calibration_rows"])
        payload = _partition_payload(
            accepted_path=accepted_path,
            verification_audit_path=verification_audit_path,
            strict2031_path=strict2031_path,
            strict1607_path=strict1607_path,
            train_path=train_path,
            calibration_path=calibration_path,
            seed=seed,
            salt_candidates=salt_candidates,
            require_output_match=True,
        )
        _atomic_write_json(audit_path, payload)
    except Exception:
        for path in outputs:
            path.unlink(missing_ok=True)
        raise
    return payload


def verify_partition(
    audit_path: Path,
    *,
    expected_accepted: Path | None = None,
    expected_verification_audit: Path | None = None,
    expected_train: Path | None = None,
) -> Dict[str, Any]:
    audit_path = audit_path.resolve()
    audit = read_json(audit_path)
    if audit.get("schema") != PARTITION_SCHEMA or audit.get("kind") != "completed_fixed_top1_image_partition":
        raise SelectionError("partition audit schema/kind is invalid")
    inputs = audit.get("inputs")
    outputs = audit.get("outputs")
    policy = audit.get("policy")
    if not all(isinstance(value, Mapping) for value in (inputs, outputs, policy)):
        raise SelectionError("partition audit is incomplete")
    accepted_record = _current_file_record(inputs.get("accepted"), label="partition accepted")
    verification_record = _current_file_record(
        inputs.get("verification_audit"), label="partition verification audit"
    )
    train_record = _current_file_record(outputs.get("train"), label="partition train")
    calibration_record = _current_file_record(
        outputs.get("calibration"), label="partition calibration"
    )
    for expected, observed, label in (
        (expected_accepted, accepted_record, "accepted"),
        (expected_verification_audit, verification_record, "verification audit"),
        (expected_train, train_record, "train"),
    ):
        if expected is not None and Path(observed["path"]) != expected.resolve():
            raise SelectionError(f"partition {label} path mismatch")
    strict = inputs.get("strict_holdout_manifests")
    if not isinstance(strict, Mapping):
        raise SelectionError("partition audit has no strict holdout manifest records")
    strict2031 = _current_file_record(strict.get("strict2031"), label="partition strict2031")
    strict1607 = _current_file_record(strict.get("strict1607"), label="partition strict1607")
    selected_salt = policy.get("selected_salt")
    salt_candidates = policy.get("salt_candidates")
    seed = policy.get("seed")
    if any(type(value) is not int for value in (selected_salt, salt_candidates, seed)):
        raise SelectionError("partition hash policy integer fields are invalid")
    if int(seed) != DEFAULT_PARTITION_SEED or int(salt_candidates) != DEFAULT_SALT_CANDIDATES:
        raise SelectionError("partition seed/salt policy is not the pre-registered policy")
    recomputed = _partition_payload(
        accepted_path=Path(accepted_record["path"]),
        verification_audit_path=Path(verification_record["path"]),
        strict2031_path=Path(strict2031["path"]),
        strict1607_path=Path(strict1607["path"]),
        train_path=Path(train_record["path"]),
        calibration_path=Path(calibration_record["path"]),
        seed=int(seed),
        salt_candidates=int(salt_candidates),
        require_output_match=True,
    )
    if recomputed != audit:
        raise SelectionError("partition audit failed exact deterministic replay")
    return {
        "audit": file_record(audit_path),
        "payload": audit,
        "accepted": dict(audit["inputs"]["accepted"]),
        "train": dict(audit["outputs"]["train"]),
        "calibration": dict(audit["outputs"]["calibration"]),
        "recommended_max_target": audit["recommended_max_target"],
        "selection_readiness": dict(audit["selection_readiness"]),
    }


def _completion_path(root: Path, label: str) -> Path:
    return root / label / "calibration_eval_complete.json"


def _validate_no_strict_score_path(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_no_strict_score_path(item, context=f"{context}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_no_strict_score_path(item, context=f"{context}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in STRICT_SCORE_TOKENS):
            raise SelectionError(f"strict score/result path leaked into {context}")


def verify_calibration_completion(
    path: Path,
    *,
    expected_manifest: Mapping[str, Any],
    expected_role: str,
    expected_iteration: int | None,
) -> Dict[str, Any]:
    path = path.resolve()
    value = read_json(path)
    if (
        value.get("schema") != CALIBRATION_COMPLETION_SCHEMA
        or value.get("kind") != "completed_calibration_evaluation"
        or value.get("input_role") != expected_role
    ):
        raise SelectionError(f"invalid calibration completion: {path}")
    if value.get("selection_input_scope") != "calibration_only":
        raise SelectionError("calibration completion lost its selection-only scope")
    isolation = value.get("strict_isolation")
    if isolation != {
        "strict_metric_inputs": [],
        "strict_result_paths": [],
        "strict_paths_consumed_for_scoring": False,
    }:
        raise SelectionError("calibration completion consumed strict score inputs")
    _validate_no_strict_score_path(
        {
            "manifest": value.get("manifest"),
            "records": value.get("records"),
            "summary": value.get("summary"),
        },
        context="calibration score inputs",
    )
    manifest = _current_file_record(value.get("manifest"), label="calibration manifest")
    if not _same_file_record(manifest, expected_manifest):
        raise SelectionError("calibration completion manifest differs from partition")
    checkpoint = _current_file_record(value.get("checkpoint"), label="calibration checkpoint")
    checkpoint_audit = _current_file_record(
        value.get("checkpoint_audit"), label="calibration checkpoint audit"
    )
    preflight = _current_file_record(value.get("preflight"), label="calibration preflight")
    records = _current_file_record(value.get("records"), label="calibration records")
    summary = _current_file_record(value.get("summary"), label="calibration summary")
    output_dir = path.parent
    expected_preflight_path = (output_dir / "calibration_eval_preflight.json").resolve()
    expected_summary_path = (output_dir / "summary.json").resolve()
    checkpoint_path = Path(checkpoint["path"])
    records_dir = output_dir / "per_example_records"
    expected_records_path = (
        records_dir
        / f"{_checkpoint_run_prefix(checkpoint_path)}__tn_global.records.jsonl"
    ).resolve()
    if Path(preflight["path"]) != expected_preflight_path:
        raise SelectionError("calibration preflight is outside its sealed output root")
    if Path(summary["path"]) != expected_summary_path:
        raise SelectionError("calibration summary is outside its sealed output root")
    if Path(records["path"]) != expected_records_path:
        raise SelectionError("calibration records path/name is not the evaluator's sealed output")
    if records_dir.is_symlink() or not records_dir.is_dir():
        raise SelectionError("calibration records directory is missing or symlinked")
    record_entries = list(records_dir.iterdir())
    if (
        len(record_entries) != 1
        or record_entries[0].is_symlink()
        or not record_entries[0].is_file()
        or record_entries[0].resolve() != expected_records_path
    ):
        raise SelectionError("calibration output must contain exactly one canonical record file")
    preflight_value = _read_bound_json(preflight, label="calibration preflight")
    if (
        preflight_value.get("schema") != CALIBRATION_COMPLETION_SCHEMA
        or preflight_value.get("kind") != "calibration_evaluation_preflight"
        or preflight_value.get("input_role") != expected_role
        or preflight_value.get("iteration") != expected_iteration
        or preflight_value.get("selection_input_scope") != "calibration_only"
        or preflight_value.get("strict_isolation") != isolation
    ):
        raise SelectionError("calibration preflight scope/role contract drifted")
    if preflight_value.get("manifest") != dict(value["manifest"]):
        raise SelectionError("calibration preflight manifest binding drifted")
    if preflight_value.get("checkpoint") != dict(value["checkpoint"]):
        raise SelectionError("calibration preflight checkpoint binding drifted")
    if preflight_value.get("checkpoint_audit") != dict(value["checkpoint_audit"]):
        raise SelectionError("calibration preflight checkpoint audit binding drifted")
    runtime = preflight_value.get("runtime")
    expected_runtime = dict(CALIBRATION_RUNTIME)
    if runtime != expected_runtime:
        raise SelectionError("calibration runtime is not the locked deploy geometry")
    runtime_actual = preflight_value.get("runtime_actual")
    if not isinstance(runtime_actual, Mapping):
        raise SelectionError("calibration preflight has no actual CUDA runtime evidence")
    data_root_value = runtime_actual.get("data_root")
    image_root_value = runtime_actual.get("image_root")
    if (
        runtime_actual.get("device_type") != "cuda"
        or not str(runtime_actual.get("device", "")).startswith("cuda")
        or runtime_actual.get("effective_amp") is not True
        or type(runtime_actual.get("cuda_device_index")) is not int
        or not str(runtime_actual.get("cuda_device_name", "")).strip()
        or not isinstance(runtime_actual.get("cuda_device_capability"), list)
        or len(runtime_actual["cuda_device_capability"]) != 2
        or any(type(value) is not int for value in runtime_actual["cuda_device_capability"])
        or type(runtime_actual.get("num_workers")) is not int
        or int(runtime_actual["num_workers"]) < 0
        or not isinstance(data_root_value, str)
        or not Path(data_root_value).is_absolute()
        or Path(data_root_value) != DEFAULT_DATA_ROOT.resolve()
        or not isinstance(image_root_value, str)
        or not Path(image_root_value).is_absolute()
        or Path(image_root_value)
        != Path(data_root_value) / "COCO/coco2014/train2014"
        or not str(runtime_actual.get("torch_version", "")).strip()
        or not str(runtime_actual.get("torch_cuda_version", "")).strip()
        or type(runtime_actual.get("cudnn_version")) is not int
        or runtime_actual.get("environment") != {"GFLOPS_DEBUG_SHILONG": None}
    ):
        raise SelectionError("calibration actual CUDA/AMP/runtime evidence is invalid")
    if preflight_value.get("score_contract") != CALIBRATION_SCORE_CONTRACT:
        raise SelectionError("calibration score/query contract drifted")
    code = preflight_value.get("code")
    config_chain = preflight_value.get("config_import_chain")
    config = _current_file_record(
        preflight_value.get("config"), label="calibration config"
    )
    probe_preflight = _current_file_record(
        preflight_value.get("probe_preflight"), label="calibration probe preflight"
    )
    partition_audit = _current_file_record(
        preflight_value.get("partition_audit"), label="calibration partition audit"
    )
    if code != calibration_code_records():
        raise SelectionError("calibration evaluator dependency closure drifted")
    if config_chain != calibration_config_records(Path(config["path"])):
        raise SelectionError("calibration config import closure drifted")
    probe_value = _read_bound_json(
        probe_preflight, label="calibration probe preflight"
    )
    probe_static = probe_value.get("static")
    if (
        probe_value.get("schema") != FIXED_TOP1_SCHEMA
        or probe_value.get("kind") != "phase_preflight"
        or probe_value.get("phase") != "fixed-top1-confidence"
        or not isinstance(probe_static, Mapping)
        or probe_static.get("config") != config
        or not isinstance(probe_static.get("partition"), Mapping)
        or probe_static["partition"].get("audit") != partition_audit
    ):
        raise SelectionError("calibration config/partition is not bound to its probe preflight")
    if expected_role == "milestone":
        if type(value.get("iteration")) is not int or int(value["iteration"]) != int(expected_iteration):
            raise SelectionError("calibration completion iteration mismatch")
        checkpoint_audit_value = _read_bound_json(
            checkpoint_audit, label="calibration milestone audit"
        )
        if (
            checkpoint_audit_value.get("schema") != FIXED_TOP1_SCHEMA
            or checkpoint_audit_value.get("kind") != "milestone_checkpoint"
            or int(checkpoint_audit_value.get("iteration", -1)) != int(expected_iteration)
            or not _same_file_record(checkpoint_audit_value.get("checkpoint", {}), checkpoint)
        ):
            raise SelectionError("calibration milestone audit/checkpoint mismatch")
        expected_verification = replay_calibration_checkpoint_verification(
            role="milestone",
            checkpoint=Path(checkpoint["path"]),
            checkpoint_audit=Path(checkpoint_audit["path"]),
            baseline_checkpoint=None,
            milestone_iteration=int(expected_iteration),
        )
        if preflight_value.get("checkpoint_verification") != expected_verification:
            raise SelectionError("calibration milestone deep verification evidence drifted")
    else:
        if expected_iteration is not None or value.get("iteration") is not None:
            raise SelectionError("P0 calibration completion must not have an iteration")
        identity = value.get("p0_identity")
        if not isinstance(identity, Mapping) or any(
            identity.get(key) is not True
            for key in (
                "rank_score_equals_base",
                "confidence_score_equals_base",
                "rank_residual_exact_zero",
                "confidence_gate_exact_zero",
            )
        ):
            raise SelectionError("P0 calibration completion lacks identity proof")
        p0_sidecar = _read_bound_json(
            checkpoint_audit, label="calibration P0 sidecar"
        )
        if (
            p0_sidecar.get("schema") != P0_SCHEMA
            or p0_sidecar.get("kind") != "p0_checkpoint_audit"
            or not _same_file_record(p0_sidecar.get("p0_checkpoint", {}), checkpoint)
            or p0_sidecar.get("functional_identity") != identity
        ):
            raise SelectionError("P0 sidecar does not prove calibration reference identity")
        p0_baseline = _current_file_record(
            p0_sidecar.get("baseline"), label="P0 authoritative baseline"
        )
        expected_verification = replay_calibration_checkpoint_verification(
            role="p0",
            checkpoint=Path(checkpoint["path"]),
            checkpoint_audit=Path(checkpoint_audit["path"]),
            baseline_checkpoint=Path(p0_baseline["path"]),
            milestone_iteration=None,
        )
        if preflight_value.get("checkpoint_verification") != expected_verification:
            raise SelectionError("calibration P0 deep verification evidence drifted")
    manifest_rows = int(value.get("manifest_rows", -1))
    if manifest_rows != int(expected_manifest.get("rows", -2)):
        raise SelectionError("calibration completion manifest row count mismatch")
    summary_value = _read_bound_json(summary, label="calibration summary")
    expected_forward_calls = 2 * int(
        math.ceil(manifest_rows / int(expected_runtime["batch_size"]))
    )
    expected_query_evidence = {
        "hook": "root_model_forward_hook",
        "checked_outputs": [
            "stage_b_gdino_confidence_score",
            "stage_b_gdino_base_score",
            "pred_boxes",
        ],
        "query_count_each_call": 900,
        "observed_forward_calls": expected_forward_calls,
        "expected_forward_calls": expected_forward_calls,
        "observed_examples_across_negative_positive": 2 * manifest_rows,
        "expected_examples_across_negative_positive": 2 * manifest_rows,
        "pass": True,
    }
    if (
        summary_value.get("manifest_sha256") != manifest["sha256"]
        or int(summary_value.get("manifest_n", -1)) != manifest_rows
        or int(summary_value.get("invalid_records", -1)) != 0
        or not isinstance(summary_value.get("records_jsonl"), str)
        or not Path(str(summary_value["records_jsonl"])).is_absolute()
        or Path(str(summary_value["records_jsonl"])).resolve() != expected_records_path
        or not isinstance(summary_value.get("checkpoint"), str)
        or not Path(str(summary_value["checkpoint"])).is_absolute()
        or Path(str(summary_value["checkpoint"])).resolve() != checkpoint_path
        or summary_value.get("run_id") != _checkpoint_run_prefix(checkpoint_path)
        or int(summary_value.get("batch_size", -1)) != int(expected_runtime["batch_size"])
        or int(summary_value.get("seed", -1)) != int(expected_runtime["seed"])
        or int(summary_value.get("max_batches", -1)) != 0
        or summary_value.get("runtime_actual") != runtime_actual
        or summary_value.get("score_contract") != CALIBRATION_SCORE_CONTRACT
        or summary_value.get("query_geometry_evidence") != expected_query_evidence
    ):
        raise SelectionError("calibration summary completeness binding failed")
    return {
        "completion": file_record(path),
        "checkpoint": checkpoint,
        "checkpoint_audit": checkpoint_audit,
        "preflight": preflight,
        "records": records,
        "summary": summary,
        "summary_value": summary_value,
        "preflight_value": preflight_value,
        "value": value,
    }


def _verify_summary_recomputed_from_records(
    completion: Mapping[str, Any], records: Any
) -> None:
    summary = completion.get("summary_value")
    checkpoint = completion.get("checkpoint")
    if not isinstance(summary, Mapping) or not isinstance(checkpoint, Mapping):
        raise SelectionError("calibration completion lacks summary/checkpoint evidence")
    if not bool(records.valid.all()):
        raise SelectionError("calibration summary cannot be recomputed from invalid records")
    expected_run_id = _checkpoint_run_prefix(Path(str(checkpoint["path"])))
    if tuple(records.run_ids) != (expected_run_id,):
        raise SelectionError("calibration record run_id is not checkpoint-derived")
    computed = exact_fpr95(records.positive, records.negative)
    expected_fields = {
        "num_pairs": int(records.positive.size),
        "threshold_at_95tpr": float(computed["threshold"]),
        "actual_tpr_at_95tpr": float(computed["actual_tpr"]),
        "fpr95tpr": float(computed["fpr"]),
        "tn_fpr": float(computed["fpr"]),
    }
    for key, expected in expected_fields.items():
        observed = summary.get(key)
        if type(expected) is int:
            matches = type(observed) is int and observed == expected
        else:
            matches = isinstance(observed, (int, float)) and float(observed) == expected
        if not matches:
            raise SelectionError(
                f"calibration summary field {key} was not exactly recomputed from records"
            )


def _selection_payload(*, probe_preflight_path: Path, calibration_root: Path) -> Dict[str, Any]:
    probe_preflight_path = probe_preflight_path.resolve()
    calibration_root = calibration_root.resolve()
    probe_preflight_record = file_record(probe_preflight_path)
    probe_preflight = _read_bound_json(
        probe_preflight_record, label="selection probe preflight"
    )
    if (
        probe_preflight.get("schema") != FIXED_TOP1_SCHEMA
        or probe_preflight.get("kind") != "phase_preflight"
        or probe_preflight.get("phase") != "fixed-top1-confidence"
    ):
        raise SelectionError("selection input is not a fixed-top1 phase preflight")
    launch = probe_preflight.get("launch")
    static = probe_preflight.get("static")
    if not isinstance(launch, Mapping) or not isinstance(static, Mapping):
        raise SelectionError("fixed-top1 preflight is incomplete")
    milestones = launch.get("milestones")
    if (
        not isinstance(milestones, list)
        or not milestones
        or tuple(int(value) for value in milestones)
        != tuple(target for target in MILESTONES if target <= int(launch.get("max_target", -1)))
    ):
        raise SelectionError("selection milestone prefix drifted")
    partition_value = static.get("partition")
    if not isinstance(partition_value, Mapping):
        raise SelectionError("fixed-top1 preflight has no partition binding")
    partition_audit = _current_file_record(
        partition_value.get("audit"), label="selection partition audit"
    )
    partition = verify_partition(Path(partition_audit["path"]))
    calibration_manifest = dict(partition["calibration"])
    if partition["recommended_max_target"] != int(launch["max_target"]):
        raise SelectionError("probe max target differs from partition recommendation")
    p0 = verify_calibration_completion(
        _completion_path(calibration_root, "p0"),
        expected_manifest=calibration_manifest,
        expected_role="p0",
        expected_iteration=None,
    )
    if not _same_file_record(
        p0["preflight_value"].get("partition_audit", {}), partition_audit
    ) or not _same_file_record(
        p0["preflight_value"].get("probe_preflight", {}),
        probe_preflight_record,
    ):
        raise SelectionError("P0 calibration did not use the sealed probe partition")
    p0_sidecar = _read_bound_json(
        p0["checkpoint_audit"], label="selection P0 sidecar"
    )
    p0_baseline = _current_file_record(
        p0_sidecar.get("baseline"), label="selection P0 baseline"
    )
    source_binding = probe_preflight.get("fixed_gdino_source_binding")
    if (
        not isinstance(source_binding, Mapping)
        or source_binding.get("matches_rank_initial_baseline") is not True
        or source_binding.get("checkpoint_sha256") != p0_baseline["sha256"]
    ):
        raise SelectionError("P0 reference is not the probe's authoritative baseline")
    manifest = load_manifest(calibration_manifest["path"])
    if not _same_file_record(manifest.file_record, calibration_manifest):
        raise SelectionError("calibration manifest changed while selection consumed it")
    baseline_records = load_tn_records(
        p0["records"]["path"], manifest, label="calibration P0"
    )
    if not _same_file_record(baseline_records.file_record, p0["records"]):
        raise SelectionError("P0 records changed while selection consumed them")
    _verify_summary_recomputed_from_records(p0, baseline_records)
    if not bool(baseline_records.valid.all()):
        raise SelectionError("P0 calibration records contain invalid rows")
    reports: Dict[str, Any] = {}
    milestone_inputs: Dict[str, Any] = {}
    eligible = []
    for raw_iteration in milestones:
        iteration = int(raw_iteration)
        label = f"s{iteration:06d}"
        expected_checkpoint = (
            probe_preflight_path.parent
            / "milestones"
            / f"checkpoint_iter_{iteration:06d}.pth"
        ).resolve()
        expected_audit = expected_checkpoint.with_suffix(".audit.json")
        completion = verify_calibration_completion(
            _completion_path(calibration_root, label),
            expected_manifest=calibration_manifest,
            expected_role="milestone",
            expected_iteration=iteration,
        )
        if not _same_file_record(
            completion["preflight_value"].get("partition_audit", {}),
            partition_audit,
        ) or not _same_file_record(
            completion["preflight_value"].get("probe_preflight", {}),
            probe_preflight_record,
        ):
            raise SelectionError(
                f"S{iteration} calibration did not use the sealed probe partition"
            )
        if Path(completion["checkpoint"]["path"]) != expected_checkpoint:
            raise SelectionError(
                f"S{iteration} calibration used a checkpoint outside the sealed probe prefix"
            )
        if Path(completion["checkpoint_audit"]["path"]) != expected_audit:
            raise SelectionError(
                f"S{iteration} calibration used the wrong milestone audit path"
            )
        records = load_tn_records(
            completion["records"]["path"], manifest, label=f"calibration S{iteration}"
        )
        if not _same_file_record(records.file_record, completion["records"]):
            raise SelectionError(
                f"S{iteration} records changed while selection consumed them"
            )
        _verify_summary_recomputed_from_records(completion, records)
        if not bool(records.valid.all()):
            raise SelectionError(f"S{iteration} calibration records contain invalid rows")
        try:
            report = compare_records(
                baseline_records,
                records,
                manifest,
                bootstrap_iterations=BOOTSTRAP_ITERATIONS,
                confidence=BOOTSTRAP_CONFIDENCE,
                seed=BOOTSTRAP_SEED,
            )
        except RecordComparisonError as error:
            raise SelectionError(f"S{iteration} FPR95 comparison failed: {error}") from error
        delta = float(report["global"]["candidate_minus_baseline_fpr95"])
        probability = float(report["paired_bootstrap"]["probability_delta_below_zero"])
        candidate_fpr = float(report["global"]["candidate"]["fpr95"]["fpr"])
        passes = delta < 0.0 and probability >= PROMOTION_PROBABILITY
        reports[str(iteration)] = {
            "eligible": passes,
            "eligibility": {
                "candidate_minus_p0_fpr95_strictly_below_zero": delta < 0.0,
                "paired_image_bootstrap_probability_delta_below_zero": probability,
                "minimum_probability": PROMOTION_PROBABILITY,
            },
            "comparison": report,
        }
        milestone_inputs[str(iteration)] = {
            key: completion[key]
            for key in ("completion", "checkpoint", "checkpoint_audit", "preflight", "records", "summary")
        }
        if passes:
            eligible.append((candidate_fpr, iteration, completion))
    if not eligible:
        raise SelectionError(
            "no fixed-top1 milestone passes held-out exact FPR95 and bootstrap promotion"
        )
    eligible.sort(key=lambda item: (item[0], item[1]))
    selected_fpr, selected_iteration, selected = eligible[0]
    score_inputs = {
        "manifest": calibration_manifest,
        "p0_records": p0["records"],
        "milestone_records": {
            key: value["records"] for key, value in milestone_inputs.items()
        },
    }
    _validate_no_strict_score_path(score_inputs, context="selection score inputs")
    return {
        "schema": SELECTION_SCHEMA,
        "kind": "completed_fixed_top1_milestone_selection",
        "probe_preflight": probe_preflight_record,
        "calibration_root": {
            "path": str(calibration_root),
            "layout": "p0_and_six_digit_milestone_subdirectories",
        },
        "partition_audit": partition_audit,
        "selection_policy": {
            "metric": "exact_global_fpr_at_95tpr",
            "threshold_rule": "ceil(0.95*N)_positive_order_statistic_with_greater_equal_ties",
            "reference": "P0_exact_identity_to_authoritative_fixed_baseline",
            "bootstrap": {
                "unit": "image_cluster",
                "iterations": BOOTSTRAP_ITERATIONS,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "seed": BOOTSTRAP_SEED,
                "minimum_probability_delta_below_zero": PROMOTION_PROBABILITY,
            },
            "eligibility": (
                "candidate_minus_p0_fpr95<0 and "
                "paired_bootstrap_probability_delta_below_zero>=0.95"
            ),
            "ordering": "minimum_candidate_fpr95_then_minimum_iteration",
            "all_preflight_milestones_required_exactly_once": True,
        },
        "selection_input_contract": {
            "scope": "calibration_only",
            "score_inputs": score_inputs,
            "strict_metric_inputs": [],
            "strict_result_paths": [],
            "strict_paths_consumed_for_scoring": False,
            "partition_holdout_proof": partition_audit,
        },
        "p0_input": {
            key: p0[key]
            for key in ("completion", "checkpoint", "checkpoint_audit", "preflight", "records", "summary")
        },
        "milestone_inputs": milestone_inputs,
        "reports": reports,
        "selected_iteration": selected_iteration,
        "selected_candidate_fpr95": selected_fpr,
        "selected_checkpoint": selected["checkpoint"],
        "selected_milestone_audit": selected["checkpoint_audit"],
        "selection_complete": True,
    }


def create_selection(*, probe_preflight_path: Path, calibration_root: Path, output: Path) -> Dict[str, Any]:
    if output.exists():
        raise SelectionError(f"refusing to overwrite selection audit: {output}")
    payload = _selection_payload(
        probe_preflight_path=probe_preflight_path,
        calibration_root=calibration_root,
    )
    _atomic_write_json(output, payload)
    return payload


def verify_selection(
    path: Path,
    *,
    expected_checkpoint: Path | None = None,
    expected_milestone_audit: Path | None = None,
    expected_calibration_root: Path | None = None,
) -> Dict[str, Any]:
    path = path.resolve()
    value = read_json(path)
    if (
        value.get("schema") != SELECTION_SCHEMA
        or value.get("kind") != "completed_fixed_top1_milestone_selection"
        or value.get("selection_complete") is not True
    ):
        raise SelectionError("selection audit schema/kind is invalid")
    preflight_record = _current_file_record(
        value.get("probe_preflight"), label="selection probe preflight"
    )
    p0_input = value.get("p0_input")
    if not isinstance(p0_input, Mapping):
        raise SelectionError("selection audit has no P0 input")
    p0_completion = _current_file_record(
        p0_input.get("completion"), label="selection P0 completion"
    )
    calibration_root = Path(p0_completion["path"]).parent.parent
    root_contract = value.get("calibration_root")
    if root_contract != {
        "path": str(calibration_root),
        "layout": "p0_and_six_digit_milestone_subdirectories",
    }:
        raise SelectionError("selection calibration root/layout binding drifted")
    if (
        expected_calibration_root is not None
        and calibration_root != expected_calibration_root.resolve()
    ):
        raise SelectionError("selection used an unexpected calibration evaluation root")
    recomputed = _selection_payload(
        probe_preflight_path=Path(preflight_record["path"]),
        calibration_root=calibration_root,
    )
    if recomputed != value:
        raise SelectionError("selection audit failed exact records-based replay")
    selected_checkpoint = _current_file_record(
        value.get("selected_checkpoint"), label="selected checkpoint"
    )
    selected_audit = _current_file_record(
        value.get("selected_milestone_audit"), label="selected milestone audit"
    )
    if expected_checkpoint is not None and Path(selected_checkpoint["path"]) != expected_checkpoint.resolve():
        raise SelectionError("formal checkpoint is not the uniquely selected milestone")
    if expected_milestone_audit is not None and Path(selected_audit["path"]) != expected_milestone_audit.resolve():
        raise SelectionError("formal checkpoint audit is not the selected milestone audit")
    contract = value.get("selection_input_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("scope") != "calibration_only"
        or contract.get("strict_metric_inputs") != []
        or contract.get("strict_result_paths") != []
        or contract.get("strict_paths_consumed_for_scoring") is not False
    ):
        raise SelectionError("selection audit does not prove strict-score isolation")
    _validate_no_strict_score_path(
        contract.get("score_inputs"), context="verified selection score inputs"
    )
    return {
        "audit": file_record(path),
        "selected_checkpoint": selected_checkpoint,
        "selected_milestone_audit": selected_audit,
        "selected_iteration": int(value["selected_iteration"]),
        "calibration_root": dict(root_contract),
        "input_scope": "calibration_only",
        "strict_paths_consumed_for_scoring": False,
        "verified": True,
    }


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _cmd_partition(args: argparse.Namespace) -> None:
    from tools.stageb_gdino_fixed_top1_probe_audit import validate_verified_pairs

    payload = create_partition(
        accepted_path=_resolve(args.accepted),
        verification_audit_path=_resolve(args.verification_audit),
        strict2031_path=_resolve(args.strict2031),
        strict1607_path=_resolve(args.strict1607),
        train_path=_resolve(args.train_output),
        calibration_path=_resolve(args.calibration_output),
        audit_path=_resolve(args.audit_output),
        seed=int(args.seed),
        salt_candidates=int(args.salt_candidates),
        source_validator=validate_verified_pairs,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_verify_partition(args: argparse.Namespace) -> None:
    result = verify_partition(_resolve(args.audit))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_select(args: argparse.Namespace) -> None:
    payload = create_selection(
        probe_preflight_path=_resolve(args.probe_preflight),
        calibration_root=_resolve(args.calibration_root),
        output=_resolve(args.output),
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


def _cmd_verify_selection(args: argparse.Namespace) -> None:
    result = verify_selection(
        _resolve(args.audit),
        expected_checkpoint=_resolve(args.checkpoint) if args.checkpoint else None,
        expected_milestone_audit=(
            _resolve(args.milestone_audit) if args.milestone_audit else None
        ),
        expected_calibration_root=(
            _resolve(args.calibration_root) if args.calibration_root else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    partition = subparsers.add_parser("partition")
    partition.add_argument("--accepted", required=True)
    partition.add_argument("--verification-audit", required=True)
    partition.add_argument("--strict2031", required=True)
    partition.add_argument("--strict1607", required=True)
    partition.add_argument("--train-output", required=True)
    partition.add_argument("--calibration-output", required=True)
    partition.add_argument("--audit-output", required=True)
    partition.add_argument("--seed", type=int, default=DEFAULT_PARTITION_SEED)
    partition.add_argument("--salt-candidates", type=int, default=DEFAULT_SALT_CANDIDATES)
    partition.set_defaults(func=_cmd_partition)

    verify_partition_parser = subparsers.add_parser("verify-partition")
    verify_partition_parser.add_argument("--audit", required=True)
    verify_partition_parser.set_defaults(func=_cmd_verify_partition)

    select = subparsers.add_parser("select")
    select.add_argument("--probe-preflight", required=True)
    select.add_argument("--calibration-root", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(func=_cmd_select)

    verify = subparsers.add_parser("verify-selection")
    verify.add_argument("--audit", required=True)
    verify.add_argument("--checkpoint")
    verify.add_argument("--milestone-audit")
    verify.add_argument("--calibration-root")
    verify.set_defaults(func=_cmd_verify_selection)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (SelectionError, RecordComparisonError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
