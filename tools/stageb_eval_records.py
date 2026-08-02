#!/usr/bin/env python3
"""Canonical per-example records shared by the Stage-B evaluators."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import torch
except ModuleNotFoundError:  # Summary-only tooling does not require tensors.
    torch = None

from tools.stageb_ref_split_contract import REF_SPLIT_CONTRACT


RECORD_SCHEMA = "stageb-eval-record-v1"
TN_DERIVED_MANIFEST_BINDING_SCHEMA = "stageb-tn-derived-manifest-binding-v1"
TN_DERIVATION_ALGORITHM = "stageb_tn_eval_split_filter_v1"

# These are the only split rewrites performed by ``_build_tn_eval_jsonl``.
# Keeping the table here lets record-only auditors reject a sidecar that tries
# to authorize an arbitrary source-to-evaluation split rewrite.
TN_ALLOWED_SPLIT_NORMALIZATIONS = {
    ("refcoco_unc", "val"): "refcoco_val",
    ("refcoco_unc", "testA"): "refcoco_testA",
    ("refcoco_unc", "testB"): "refcoco_testB",
    ("refcoco+_unc", "val"): "refcocop_val",
    ("refcoco+_unc", "testA"): "refcocop_testA",
    ("refcoco+_unc", "testB"): "refcocop_testB",
    ("refcocog_google", "val"): "refcocog_val",
    ("refcocog_umd", "val"): "refcocog_umd_val",
}


class RefRecordContractError(ValueError):
    """Raised when formal Ref records do not satisfy their fixed contract."""


@dataclass(frozen=True)
class RefRecords:
    """Validated formal Ref records and the identity data used for pairing."""

    path: Path
    file_record: Mapping[str, Any]
    identities: tuple[tuple[Any, ...], ...]
    image_ids: Any
    correct50: Any
    manifest_sha256: str
    manifest_n: int


def _formal_ref_required_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise RefRecordContractError(f"{label}: expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        rendered = value.strip()
        if rendered and rendered.lstrip("+-").isdigit():
            return int(rendered)
    raise RefRecordContractError(f"{label}: expected an integer")


def _formal_ref_finite_unit(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RefRecordContractError(
            f"{label}: expected a finite number"
        ) from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RefRecordContractError(
            f"{label}: expected a finite number in [0, 1]"
        )
    return result


def _formal_ref_resolve_path(path: Any, base_dir: Path, *, label: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise RefRecordContractError(f"{label}: path must be a non-empty string")
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = base_dir / result
    return result.resolve()


def _formal_ref_artifact_record(
    value: Any,
    base_dir: Path,
    *,
    label: str,
) -> Dict[str, Any]:
    if isinstance(value, str):
        specification: Mapping[str, Any] = {"path": value}
    elif isinstance(value, Mapping):
        specification = value
    else:
        raise RefRecordContractError(
            f"{label}: artifact must be a path string or object"
        )
    unexpected = set(specification) - {"path", "sha256", "size_bytes", "label"}
    if unexpected:
        raise RefRecordContractError(
            f"{label}: unexpected artifact fields {sorted(unexpected)}"
        )
    path = _formal_ref_resolve_path(specification.get("path"), base_dir, label=label)
    if not path.is_file():
        raise RefRecordContractError(f"{label}: file does not exist: {path}")
    size = int(path.stat().st_size)
    if "size_bytes" in specification:
        try:
            expected_size = int(specification["size_bytes"])
        except (TypeError, ValueError) as error:
            raise RefRecordContractError(
                f"{label}: size_bytes must be an integer"
            ) from error
        if size != expected_size:
            raise RefRecordContractError(
                f"{label}: size mismatch, expected {expected_size}, found {size}"
            )
    expected_sha = specification.get("sha256")
    if expected_sha is not None:
        expected_sha = str(expected_sha).strip().lower()
        if len(expected_sha) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha
        ):
            raise RefRecordContractError(
                f"{label}: sha256 must be 64 lowercase hex characters"
            )
    actual_sha = sha256_file(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise RefRecordContractError(
            f"{label}: SHA-256 mismatch, expected {expected_sha}, found {actual_sha}"
        )
    result: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": size,
        "sha256": actual_sha,
        "hash_verified": expected_sha is not None,
    }
    if "label" in specification:
        result["label"] = str(specification["label"])
    return result


def _read_formal_ref_jsonl(path: Path, *, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        rendered = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RefRecordContractError(
            f"{label}: could not read JSONL: {error}"
        ) from error
    for line_number, line in enumerate(rendered.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RefRecordContractError(
                f"{label}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise RefRecordContractError(
                f"{label}:{line_number}: expected an object"
            )
        rows.append(row)
    return rows


def _assert_formal_ref_artifact_unchanged(
    record: Mapping[str, Any], *, label: str
) -> None:
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    expected_size = _formal_ref_required_int(
        record.get("size_bytes"), label=f"{label}.size_bytes"
    )
    expected_sha = str(record.get("sha256", ""))
    stat = path.stat()
    if int(stat.st_size) != expected_size or sha256_file(path) != expected_sha:
        raise RefRecordContractError(
            f"{label}: artifact changed between identity verification and parsing"
        )


def _formal_ref_summary_record_path_matches(
    reported: Any,
    explicit: Path,
    *,
    summary_path: Path,
    manifest_base: Path,
) -> bool:
    if not isinstance(reported, str) or not reported.strip():
        return False
    raw = Path(reported).expanduser()
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [raw] if raw.is_absolute() else [
        repo_root / raw,
        summary_path.parent / raw,
        manifest_base / raw,
    ]
    return any(candidate.resolve() == explicit.resolve() for candidate in candidates)


def load_formal_ref_records(
    artifact: Any,
    *,
    base_dir: Path,
    label: str,
    split: str,
    summary_row: Mapping[str, Any],
    summary_path: Path,
    split_contract: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> RefRecords:
    """Load and replay one fixed-protocol Ref record artifact.

    The returned arrays are suitable for paired image-cluster bootstrap.  The
    optional contract argument lets callers bind a separately attested split
    contract while the canonical evaluator contract remains the default.
    """

    contract = REF_SPLIT_CONTRACT if split_contract is None else split_contract
    base_dir = Path(base_dir)
    summary_path = Path(summary_path)
    file_record = _formal_ref_artifact_record(
        artifact, base_dir, label=label
    )
    path = Path(file_record["path"])
    if not _formal_ref_summary_record_path_matches(
        summary_row.get("records_jsonl"),
        path,
        summary_path=summary_path,
        manifest_base=base_dir,
    ):
        raise RefRecordContractError(
            f"{label}: explicit record path is not the records_jsonl bound by summary"
        )
    rows = _read_formal_ref_jsonl(path, label=label)
    _assert_formal_ref_artifact_unchanged(file_record, label=label)
    if not rows:
        raise RefRecordContractError(f"{label}: record file is empty")
    summary_manifest_n = _formal_ref_required_int(
        summary_row.get("manifest_n"), label=f"{label}.summary.manifest_n"
    )
    summary_manifest_sha = str(summary_row.get("manifest_sha256", "")).lower()
    if len(summary_manifest_sha) != 64:
        raise RefRecordContractError(
            f"{label}: summary has no valid manifest_sha256"
        )
    official = contract[split]
    if summary_manifest_n != int(official["rows"]):
        raise RefRecordContractError(
            f"{label}: manifest_n={summary_manifest_n} differs from official "
            f"{split} rows={official['rows']}"
        )
    if summary_manifest_sha != str(official["sha256"]):
        raise RefRecordContractError(
            f"{label}: manifest_sha256 differs from official {split} contract"
        )
    if len(rows) != summary_manifest_n:
        raise RefRecordContractError(
            f"{label}: records N={len(rows)} != full manifest N={summary_manifest_n}"
        )
    summary_n = _formal_ref_required_int(
        summary_row.get("num_expressions"),
        label=f"{label}.summary.num_expressions",
    )
    if summary_n != len(rows):
        raise RefRecordContractError(
            f"{label}: summary num_expressions={summary_n} != records N={len(rows)}"
        )
    if (
        _formal_ref_required_int(
            summary_row.get("max_batches", 0), label=f"{label}.max_batches"
        )
        != 0
    ):
        raise RefRecordContractError(
            f"{label}: max_batches must be zero for a formal result"
        )

    identities: List[tuple[Any, ...]] = []
    images: List[int] = []
    correct: List[bool] = []
    seen = set()
    expected_run_id = summary_row.get("run_id")
    if not isinstance(expected_run_id, str) or not expected_run_id:
        raise RefRecordContractError(f"{label}: summary run_id is missing")
    for index, row in enumerate(rows):
        location = f"{label} record {index}"
        if row.get("schema") != RECORD_SCHEMA:
            raise RefRecordContractError(f"{location}: wrong record schema")
        if row.get("task") != "ref" or row.get("split") != split:
            raise RefRecordContractError(f"{location}: task/split mismatch")
        if row.get("manifest_key") != f"ref:{split}":
            raise RefRecordContractError(f"{location}: manifest_key mismatch")
        if str(row.get("manifest_sha256", "")).lower() != summary_manifest_sha:
            raise RefRecordContractError(f"{location}: manifest hash mismatch")
        if (
            _formal_ref_required_int(
                row.get("manifest_n"), label=f"{location}.manifest_n"
            )
            != len(rows)
        ):
            raise RefRecordContractError(f"{location}: manifest_n mismatch")
        if (
            _formal_ref_required_int(
                row.get("manifest_index"), label=f"{location}.manifest_index"
            )
            != index
        ):
            raise RefRecordContractError(f"{location}: manifest order mismatch")
        if type(row.get("valid")) is not bool or row.get("valid") is not True:
            raise RefRecordContractError(
                f"{location}: formal records require valid=true"
            )
        if type(row.get("correct50")) is not bool:
            raise RefRecordContractError(
                f"{location}: correct50 must be an exact boolean"
            )
        top1_iou = _formal_ref_finite_unit(
            row.get("top1_iou"), label=f"{location}.top1_iou"
        )
        if bool(row["correct50"]) != bool(top1_iou >= 0.5):
            raise RefRecordContractError(
                f"{location}: correct50 does not replay from top1_iou >= 0.5"
            )
        if row.get("run_id") != expected_run_id:
            raise RefRecordContractError(f"{location}: run_id mismatch")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise RefRecordContractError(f"{location}: sample_id is required")
        image_id = _formal_ref_required_int(
            row.get("image_id"), label=f"{location}.image_id"
        )
        sample_identity = (
            sample_id,
            image_id,
            row.get("ann_id"),
            row.get("ref_id"),
            row.get("sent_id"),
        )
        if sample_identity in seen:
            raise RefRecordContractError(
                f"{location}: duplicate record identity"
            )
        seen.add(sample_identity)
        identities.append(
            (
                row.get("task"),
                row.get("manifest_key"),
                row.get("manifest_sha256"),
                row.get("manifest_n"),
                row.get("manifest_index"),
                sample_id,
                image_id,
                row.get("ann_id"),
                row.get("ref_id"),
                row.get("sent_id"),
                row.get("split"),
            )
        )
        images.append(image_id)
        correct.append(bool(row["correct50"]))
    # Summary-only auditors import this module under ``python -S``.  Keep
    # NumPy local to the formal-record loader so those imports remain light.
    import numpy as np

    measured = float(np.mean(np.asarray(correct, dtype=np.float64)))
    reported = _formal_ref_finite_unit(
        summary_row.get("acc50"), label=f"{label}.summary.acc50"
    )
    if not math.isclose(measured, reported, rel_tol=0.0, abs_tol=1e-12):
        raise RefRecordContractError(
            f"{label}: summary acc50={reported} != records acc50={measured}"
        )
    if (
        _formal_ref_required_int(
            summary_row.get("invalid_records", 0),
            label=f"{label}.invalid_records",
        )
        != 0
    ):
        raise RefRecordContractError(
            f"{label}: summary declares invalid records"
        )
    return RefRecords(
        path=path,
        file_record=file_record,
        identities=tuple(identities),
        image_ids=np.asarray(images, dtype=np.int64),
        correct50=np.asarray(correct, dtype=np.bool_),
        manifest_sha256=summary_manifest_sha,
        manifest_n=len(rows),
    )


def tn_manifest_derivation_contract() -> Dict[str, Any]:
    """Return the immutable source-to-data-manifest contract for preflights."""

    normalizations = [
        {
            "pair_source": pair_source,
            "source_split": source_split,
            "eval_split": eval_split,
        }
        for (pair_source, source_split), eval_split in sorted(
            TN_ALLOWED_SPLIT_NORMALIZATIONS.items()
        )
    ]
    return {
        "schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
        "algorithm": TN_DERIVATION_ALGORITHM,
        "source_role": "locked_verified_source_manifest",
        "derived_role": "exact_dataset_input_manifest",
        "record_manifest_sha256_role": "derived_dataset_input_manifest",
        "required_row_identity": [
            "sample_id",
            "image_id",
            "ann_id",
            "ref_id",
            "sent_id",
        ],
        "allowed_row_changes": [
            "select_first_instance",
            "set_instances_0_text_is_negative_true",
            "set_tn_eval_split",
            "set_tn_eval_pair_source",
            "set_tn_eval_source_split",
        ],
        "allowed_split_normalizations": normalizations,
        "legacy_policy": (
            "legacy direct-source v1 records remain readable by generic tools; "
            "fixed-protocol derived records without this binding must be rerun"
        ),
    }


def _gdino_single_expression_caption(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GDINO adapter expression captions must be non-empty strings")
    caption = value.strip()
    return caption if caption.endswith(".") else f"{caption} ."


def extract_adapter_tn_pair_captions(
    raw_targets: List[Mapping[str, Any]],
) -> tuple[List[str], List[str], torch.Tensor]:
    """Extract one positive and one TN expression without joint-prompt leakage.

    Fixed data-FT rows expose ``cap_list=[positive, TN]`` with an exact boolean
    ``is_tn`` vector.  Strict-v2 rows expose one TN slot and carry its positive
    counterfactual in ``rank_positive_captions``.  Any other shape is rejected
    because forwarding the joint dataset caption changes the score definition.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required to extract adapter TN pairs")
    positive_captions: List[str] = []
    negative_captions: List[str] = []
    for row_index, target in enumerate(raw_targets):
        cap_list = target.get("cap_list", None)
        is_tn = target.get("is_tn", None)
        if not isinstance(cap_list, list) or not cap_list:
            raise ValueError(
                f"adapter TN target {row_index} requires a non-empty cap_list"
            )
        if (
            not torch.is_tensor(is_tn)
            or is_tn.dtype != torch.bool
            or is_tn.dim() != 1
            or int(is_tn.numel()) != len(cap_list)
        ):
            raise ValueError(
                f"adapter TN target {row_index} requires is_tn bool[{len(cap_list)}]"
            )
        captions = [_gdino_single_expression_caption(value) for value in cap_list]
        tn_indices = is_tn.detach().cpu().nonzero(as_tuple=False).flatten().tolist()
        positive_indices = (
            (~is_tn.detach().cpu()).nonzero(as_tuple=False).flatten().tolist()
        )
        if len(tn_indices) != 1:
            raise ValueError(
                f"adapter TN target {row_index} must contain exactly one TN slot"
            )
        tn_index = int(tn_indices[0])

        if len(captions) == 2 and len(positive_indices) == 1:
            positive_caption = captions[int(positive_indices[0])]
        elif len(captions) == 1 and not positive_indices:
            rank_captions = target.get("rank_positive_captions", None)
            has_rank_positive = target.get("has_rank_positive", None)
            if (
                not isinstance(rank_captions, list)
                or len(rank_captions) != 1
                or not torch.is_tensor(has_rank_positive)
                or has_rank_positive.dtype != torch.bool
                or has_rank_positive.numel() != 1
                or bool(has_rank_positive.detach().reshape(-1)[0].item()) is not True
            ):
                raise ValueError(
                    f"adapter TN target {row_index} lacks an exact positive counterfactual"
                )
            positive_caption = _gdino_single_expression_caption(rank_captions[0])
        else:
            raise ValueError(
                f"adapter TN target {row_index} has unsupported paired layout"
            )

        positive_captions.append(positive_caption)
        negative_captions.append(captions[tn_index])
    return (
        positive_captions,
        negative_captions,
        torch.ones((len(raw_targets),), dtype=torch.bool),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, rows: Optional[int] = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    record: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def tn_manifest_binding_path(derived_manifest_path: Path) -> Path:
    path = Path(derived_manifest_path)
    return path.with_suffix(path.suffix + ".binding.json")


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_derived_tn_row(
    source: Mapping[str, Any], mapping: Mapping[str, Any]
) -> Dict[str, Any]:
    instances = source.get("instances")
    if not isinstance(instances, list) or not instances or not isinstance(instances[0], Mapping):
        raise ValueError("TN source row requires a non-empty object instances list")
    pair_source = str(mapping.get("pair_source", ""))
    source_split = str(mapping.get("source_split", ""))
    eval_split = str(mapping.get("eval_split", ""))
    if TN_ALLOWED_SPLIT_NORMALIZATIONS.get((pair_source, source_split)) != eval_split:
        raise ValueError(
            "TN manifest binding contains a disallowed split normalization: "
            f"{pair_source!r}/{source_split!r} -> {eval_split!r}"
        )
    source_pair = str(
        instances[0].get("pair_source")
        or source.get("pair_source")
        or source.get("source")
        or ""
    )
    if source_pair != pair_source:
        raise ValueError(
            f"TN manifest binding pair_source drift: {pair_source!r} != {source_pair!r}"
        )
    instance = copy.deepcopy(dict(instances[0]))
    instance["text_is_negative"] = True
    expected = copy.deepcopy(dict(source))
    expected["tn_eval_split"] = eval_split
    expected["tn_eval_pair_source"] = pair_source
    expected["tn_eval_source_split"] = source_split
    expected["instances"] = [instance]
    return expected


@dataclass(frozen=True)
class TNDerivedManifestBinding:
    path: Path
    sha256: str
    size_bytes: int
    source_manifest: Mapping[str, Any]
    derived_manifest: Mapping[str, Any]
    derivation: Mapping[str, Any]
    row_mapping_sha256: str
    row_mapping: List[Dict[str, Any]]


def _validate_tn_manifest_binding_payload(
    payload: Mapping[str, Any],
    *,
    binding_path: Path,
    expected_derived_manifest: Optional[Path] = None,
) -> TNDerivedManifestBinding:
    if set(payload) != {
        "schema",
        "kind",
        "derivation",
        "source_manifest",
        "derived_manifest",
        "row_mapping_sha256",
        "row_mapping",
    }:
        raise ValueError("TN manifest binding has unexpected or missing fields")
    if payload.get("schema") != TN_DERIVED_MANIFEST_BINDING_SCHEMA or payload.get(
        "kind"
    ) != "deterministic_tn_eval_manifest_derivation":
        raise ValueError("TN manifest binding has the wrong schema or kind")
    derivation = payload.get("derivation")
    if not isinstance(derivation, Mapping) or set(derivation) != {
        "contract",
        "requested_splits",
        "max_pairs",
        "max_pairs_per_split",
        "holdout_level",
    } or derivation.get("contract") != tn_manifest_derivation_contract():
        raise ValueError("TN manifest binding derivation contract drifted")
    source_record = payload.get("source_manifest")
    derived_record = payload.get("derived_manifest")
    if not isinstance(source_record, Mapping) or not isinstance(derived_record, Mapping):
        raise ValueError("TN manifest binding is missing source or derived file records")
    for label, record in (("source", source_record), ("derived", derived_record)):
        if set(record) != {"path", "size_bytes", "sha256", "rows"}:
            raise ValueError(f"TN {label} manifest file record has unexpected fields")
        path = Path(str(record.get("path", ""))).resolve()
        if not path.is_file():
            raise ValueError(f"TN {label} manifest is missing: {path}")
        rows = list(_iter_jsonl(path))
        if dict(record) != _file_record(path, rows=len(rows)):
            raise ValueError(f"TN {label} manifest changed after binding")
    source_path = Path(str(source_record["path"])).resolve()
    derived_path = Path(str(derived_record["path"])).resolve()
    if expected_derived_manifest is not None and derived_path != Path(
        expected_derived_manifest
    ).resolve():
        raise ValueError("TN binding points at a different derived data manifest")
    source_rows = list(_iter_jsonl(source_path))
    derived_rows = list(_iter_jsonl(derived_path))
    row_mapping = payload.get("row_mapping")
    if not isinstance(row_mapping, list) or len(row_mapping) != len(derived_rows):
        raise ValueError("TN manifest binding row_mapping length mismatch")
    if payload.get("row_mapping_sha256") != _canonical_json_sha256(row_mapping):
        raise ValueError("TN manifest binding row_mapping digest mismatch")
    observed_source_indices: List[int] = []
    observed_sample_ids: List[str] = []
    for derived_index, (mapping, derived_row) in enumerate(zip(row_mapping, derived_rows)):
        if not isinstance(mapping, Mapping):
            raise ValueError(f"TN manifest binding row {derived_index} is not an object")
        if set(mapping) != {
            "derived_index",
            "source_index",
            "sample_id",
            "pair_source",
            "source_split",
            "eval_split",
        }:
            raise ValueError(
                f"TN manifest binding row {derived_index} has unexpected or missing fields"
            )
        if int(mapping.get("derived_index", -1)) != derived_index:
            raise ValueError("TN manifest binding derived indices are not exact 0..N-1")
        source_index = int(mapping.get("source_index", -1))
        if source_index < 0 or source_index >= len(source_rows):
            raise ValueError(f"TN manifest binding source index {source_index} is invalid")
        observed_source_indices.append(source_index)
        source_row = source_rows[source_index]
        expected = _expected_derived_tn_row(source_row, mapping)
        if derived_row != expected:
            raise ValueError(
                f"TN derived manifest row {derived_index} is not the declared deterministic transform"
            )
        source_sample_id = sample_id_from_meta(
            source_row, task="tn", split="global", index=source_index
        )
        derived_sample_id = sample_id_from_meta(
            derived_row, task="tn", split="global", index=derived_index
        )
        if source_sample_id != derived_sample_id or mapping.get("sample_id") != source_sample_id:
            raise ValueError(f"TN manifest binding sample identity drift at row {derived_index}")
        for key in ("image_id", "ann_id", "ref_id", "sent_id"):
            if int(source_row[key]) != int(derived_row[key]):
                raise ValueError(
                    f"TN manifest binding {key} drift at row {derived_index}"
                )
        observed_sample_ids.append(source_sample_id)
    if observed_source_indices != sorted(set(observed_source_indices)):
        raise ValueError("TN manifest binding source indices must be unique source order")
    if len(set(observed_sample_ids)) != len(observed_sample_ids):
        raise ValueError("TN manifest binding contains duplicate sample identities")
    binding_record = _file_record(binding_path)
    return TNDerivedManifestBinding(
        path=Path(binding_record["path"]),
        sha256=str(binding_record["sha256"]),
        size_bytes=int(binding_record["size_bytes"]),
        source_manifest=dict(source_record),
        derived_manifest=dict(derived_record),
        derivation=dict(derivation),
        row_mapping_sha256=str(payload["row_mapping_sha256"]),
        row_mapping=[dict(row) for row in row_mapping],
    )


def write_tn_derived_manifest_binding(
    *,
    source_manifest_path: Path,
    derived_manifest_path: Path,
    row_mapping: List[Mapping[str, Any]],
    requested_splits: List[str],
    max_pairs: int,
    max_pairs_per_split: int,
    holdout_level: str,
) -> Path:
    source_path = Path(source_manifest_path).resolve()
    derived_path = Path(derived_manifest_path).resolve()
    source_rows = list(_iter_jsonl(source_path))
    derived_rows = list(_iter_jsonl(derived_path))
    normalized_mapping = [dict(row) for row in row_mapping]
    payload = {
        "schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
        "kind": "deterministic_tn_eval_manifest_derivation",
        "derivation": {
            "contract": tn_manifest_derivation_contract(),
            "requested_splits": [str(value) for value in requested_splits],
            "max_pairs": int(max_pairs),
            "max_pairs_per_split": int(max_pairs_per_split),
            "holdout_level": str(holdout_level),
        },
        "source_manifest": _file_record(source_path, rows=len(source_rows)),
        "derived_manifest": _file_record(derived_path, rows=len(derived_rows)),
        "row_mapping_sha256": _canonical_json_sha256(normalized_mapping),
        "row_mapping": normalized_mapping,
    }
    binding_path = tn_manifest_binding_path(derived_path)
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = binding_path.with_suffix(binding_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, binding_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    _validate_tn_manifest_binding_payload(
        payload,
        binding_path=binding_path,
        expected_derived_manifest=derived_path,
    )
    return binding_path


def load_tn_derived_manifest_binding(
    binding_path: Path, *, expected_derived_manifest: Optional[Path] = None
) -> TNDerivedManifestBinding:
    path = Path(binding_path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read TN manifest binding {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"TN manifest binding must be a JSON object: {path}")
    return _validate_tn_manifest_binding_payload(
        payload,
        binding_path=path,
        expected_derived_manifest=expected_derived_manifest,
    )


def tn_manifest_binding_summary_fields(
    manifest: "EvalManifest",
) -> Dict[str, Any]:
    binding = manifest.tn_binding
    if binding is None:
        return {}
    return {
        "manifest_path": str(binding.derived_manifest["path"]),
        "manifest_size_bytes": int(binding.derived_manifest["size_bytes"]),
        "manifest_binding_schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
        "manifest_derivation_algorithm": TN_DERIVATION_ALGORITHM,
        "manifest_binding_path": str(binding.path),
        "manifest_binding_sha256": binding.sha256,
        "manifest_binding_size_bytes": binding.size_bytes,
        "source_manifest_path": str(binding.source_manifest["path"]),
        "source_manifest_sha256": str(binding.source_manifest["sha256"]),
        "source_manifest_size_bytes": int(binding.source_manifest["size_bytes"]),
        "source_manifest_n": int(binding.source_manifest["rows"]),
        "manifest_row_mapping_sha256": binding.row_mapping_sha256,
    }


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def sample_id_from_meta(meta: Mapping[str, Any], *, task: str, split: str, index: int) -> str:
    explicit = meta.get("sample_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    identity = [meta.get(key) for key in ("image_id", "ann_id", "ref_id", "sent_id")]
    if any(value is None for value in identity):
        raise ValueError(
            f"Manifest row {index} for {task}/{split} has no sample_id and an incomplete "
            "(image_id, ann_id, ref_id, sent_id) identity"
        )
    return f"{task}:{split}:" + ":".join(str(int(value)) for value in identity)


@dataclass(frozen=True)
class EvalManifest:
    path: Path
    task: str
    manifest_key: str
    split: str
    sha256: str
    rows: List[Dict[str, Any]]
    tn_binding: Optional[TNDerivedManifestBinding] = None

    @property
    def size(self) -> int:
        return len(self.rows)


def validate_eval_manifest_batch_alignment(
    raw_targets: List[Mapping[str, Any]],
    manifest: EvalManifest,
    start_index: int,
) -> None:
    """Prove that a dataset batch still follows immutable manifest order."""
    if torch is None:
        raise RuntimeError("PyTorch is required to validate tensor batch identities")
    for local_index, target in enumerate(raw_targets):
        manifest_index = int(start_index) + local_index
        if manifest_index >= manifest.size:
            raise IndexError(
                f"Evaluation dataset produced row {manifest_index} beyond manifest size {manifest.size}"
            )
        source = manifest.rows[manifest_index]
        for key in ("image_id", "ann_id", "ref_id", "sent_id"):
            value = target.get(key, None)
            if not torch.is_tensor(value) or value.numel() != 1:
                raise ValueError(
                    f"Evaluation dataset target {manifest_index} is missing scalar identity {key}"
                )
            observed = int(value.detach().reshape(-1)[0].item())
            if observed != int(source[key]):
                raise ValueError(
                    f"Evaluation dataset/manifest order drift at row {manifest_index}: "
                    f"{key}={observed} != {source[key]}"
                )
        source_sample_id = source.get("sample_id", None)
        if (
            source_sample_id is not None
            and target.get("sample_id", None) != source_sample_id
        ):
            raise ValueError(
                f"Evaluation dataset/manifest sample_id drift at row {manifest_index}"
            )


def load_eval_manifest(
    path: Path,
    *,
    task: str,
    split: str,
    manifest_key: Optional[str] = None,
) -> EvalManifest:
    path = Path(path)
    if task not in {"tn", "ref"}:
        raise ValueError(f"Unsupported evaluation task: {task!r}")
    rows = list(_iter_jsonl(path))
    if not rows:
        raise ValueError(f"Evaluation manifest is empty: {path}")
    key = manifest_key or ("tn_global" if task == "tn" else f"ref:{split}")
    ids = [sample_id_from_meta(row, task=task, split=split, index=i) for i, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Evaluation manifest contains duplicate sample IDs: {path}")
    tn_binding = None
    binding_path = tn_manifest_binding_path(path)
    if task == "tn" and binding_path.is_file():
        tn_binding = load_tn_derived_manifest_binding(
            binding_path, expected_derived_manifest=path
        )
    return EvalManifest(
        path=path,
        task=task,
        manifest_key=key,
        split=split,
        sha256=sha256_file(path),
        rows=rows,
        tn_binding=tn_binding,
    )


def make_eval_record(
    manifest: EvalManifest,
    *,
    index: int,
    run_id: str,
    valid: bool,
    values: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if index < 0 or index >= manifest.size:
        raise IndexError(f"Manifest index {index} is outside [0, {manifest.size})")
    source = manifest.rows[index]
    if meta is not None:
        for key in ("image_id", "ann_id", "ref_id", "sent_id", "sample_id"):
            left = source.get(key)
            right = meta.get(key)
            if left is not None and right is not None and str(left) != str(right):
                raise ValueError(
                    f"Evaluator metadata drift at manifest row {index}: {key}={right!r} != {left!r}"
                )
    if manifest.tn_binding is not None:
        # The binding's normalized evaluation split is authoritative.  The
        # derived row intentionally retains the source ``eval_split`` for
        # provenance, so consulting that field first would silently relabel
        # e.g. refcocop_val back to refcocoplus_unc_val in the metric records.
        record_split = manifest.tn_binding.row_mapping[index]["eval_split"]
    elif manifest.task == "tn":
        record_split = (
            source.get("tn_eval_split")
            or source.get("eval_split")
            or (meta or {}).get("eval_split")
            or manifest.split
        )
    else:
        record_split = (
            source.get("eval_split")
            or (meta or {}).get("eval_split")
            or manifest.split
        )
    record: Dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "task": manifest.task,
        "manifest_key": manifest.manifest_key,
        "manifest_sha256": manifest.sha256,
        "manifest_n": manifest.size,
        "manifest_index": int(index),
        "sample_id": sample_id_from_meta(
            source,
            task=manifest.task,
            split=manifest.split,
            index=index,
        ),
        "split": str(record_split),
        "image_id": source.get("image_id"),
        "ann_id": source.get("ann_id"),
        "ref_id": source.get("ref_id"),
        "sent_id": source.get("sent_id"),
        "run_id": str(run_id),
        "valid": bool(valid),
    }
    if manifest.tn_binding is not None:
        binding = manifest.tn_binding
        mapping = binding.row_mapping[index]
        record.update(
            {
                "manifest_path": str(binding.derived_manifest["path"]),
                "manifest_size_bytes": int(binding.derived_manifest["size_bytes"]),
                "manifest_binding_schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
                "manifest_derivation_algorithm": TN_DERIVATION_ALGORITHM,
                "manifest_binding_path": str(binding.path),
                "manifest_binding_sha256": binding.sha256,
                "manifest_binding_size_bytes": binding.size_bytes,
                "manifest_row_mapping_sha256": binding.row_mapping_sha256,
                "source_manifest_path": str(binding.source_manifest["path"]),
                "source_manifest_sha256": str(binding.source_manifest["sha256"]),
                "source_manifest_size_bytes": int(binding.source_manifest["size_bytes"]),
                "source_manifest_n": int(binding.source_manifest["rows"]),
                "source_manifest_index": int(mapping["source_index"]),
            }
        )
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        record[str(key)] = value
    return record


def write_eval_records(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=True, sort_keys=True, allow_nan=False))
                handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path
