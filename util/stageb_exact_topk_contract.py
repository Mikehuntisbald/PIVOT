"""Fail-closed contracts for fixed Stage-A Top-K TN supervision.

The exact label is intentionally narrower than semantic image absence: it says
that every candidate consumed by Stage B was reviewed for one frozen Stage-A
checkpoint, deterministic query transform, fixed support patch, and selection
contract.  Old cached-proposal or semantic-probe rows cannot satisfy it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from util.path_compat import remap_legacy_path


EXACT_TOPK_PROTOCOL = "fixed-stagea-checkpoint-transform-support-topk-v1"
EXACT_TOPK_TN_SCOPE = "image_global_fixed_stagea_topk_exact_verified"
EXACT_TOPK_PAIR_SCHEMA = "stage-b-v15-fixed-stagea-topk-exact-pair-v1"
EXACT_TOPK_RESULT_AUDIT_SCHEMA = (
    "stage-b-v15-fixed-stagea-topk-exact-results-audit-v1"
)
EXACT_TOPK_EXTRACTION_SCHEMA = "stage-b-v15-fixed-stagea-topk-extraction-v1"
EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA = (
    "stage-b-v15-fixed-stagea-topk-extraction-audit-v1"
)
EXACT_TOPK_JUDGMENT_SCHEMA = "stage-b-v15-fixed-stagea-topk-judgment-v1"
EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA = (
    "stage-b-v15-fixed-stagea-topk-judgment-audit-v1"
)

EXACT_CONTRACT_SHA_KEYS = (
    "checkpoint_sha256",
    "model_config_sha256",
    "data_config_sha256",
    "canonical_classes_sha256",
    "query_transform_contract_sha256",
    "support_transform_contract_sha256",
    "candidate_selection_contract_sha256",
)
EXACT_CONTRACT_KEYS = EXACT_CONTRACT_SHA_KEYS + (
    "candidate_topk",
    "candidate_box_atol",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExactTopKContractError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ExactTopKContractError(
            f"value is not canonical-JSON serializable: {error}"
        ) from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    path = remap_legacy_path(path).expanduser().resolve()
    if not path.is_file():
        raise ExactTopKContractError(f"required file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ExactTopKContractError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ExactTopKContractError(f"{label} must be finite")
    return result


def _bbox(value: Any, *, label: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise ExactTopKContractError(f"{label} must be a four-value cxcywh box")
    box = [_finite_float(item, label=label) for item in value]
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise ExactTopKContractError(f"{label} must have positive width and height")
    if any(item < 0.0 or item > 1.0 for item in box):
        raise ExactTopKContractError(f"{label} must be normalized to [0,1]")
    return box


def normalize_exact_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactTopKContractError("exact contract must be a mapping")
    result: dict[str, Any] = {}
    for key in EXACT_CONTRACT_SHA_KEYS:
        observed = value.get(key)
        if not is_sha256(observed):
            raise ExactTopKContractError(f"exact contract has invalid {key}")
        result[key] = str(observed)
    try:
        topk = int(value.get("candidate_topk"))
    except (TypeError, ValueError) as error:
        raise ExactTopKContractError("candidate_topk must be an integer") from error
    if isinstance(value.get("candidate_topk"), bool) or topk <= 0:
        raise ExactTopKContractError("candidate_topk must be positive")
    atol = _finite_float(value.get("candidate_box_atol"), label="candidate_box_atol")
    if atol < 0.0 or atol > 1.0e-3:
        raise ExactTopKContractError(
            "candidate_box_atol must be in the fail-closed range [0,1e-3]"
        )
    result["candidate_topk"] = topk
    result["candidate_box_atol"] = atol
    if set(value) != set(EXACT_CONTRACT_KEYS):
        extra = sorted(set(value).difference(EXACT_CONTRACT_KEYS))
        missing = sorted(set(EXACT_CONTRACT_KEYS).difference(value))
        raise ExactTopKContractError(
            f"exact contract keys drifted: missing={missing}, extra={extra}"
        )
    return result


def candidate_payload(value: Any, *, expected_rank: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactTopKContractError(f"candidate rank {expected_rank} is not an object")
    try:
        rank = int(value.get("rank"))
        query_index = int(value.get("query_index"))
    except (TypeError, ValueError) as error:
        raise ExactTopKContractError(
            f"candidate rank {expected_rank} has invalid rank/query_index"
        ) from error
    if rank != expected_rank or query_index < 0:
        raise ExactTopKContractError(
            f"candidate coverage is not ordered 0..K-1 at rank {expected_rank}"
        )
    return {
        "rank": rank,
        "query_index": query_index,
        "bbox_cxcywh_normalized": _bbox(
            value.get("bbox_cxcywh_normalized"),
            label=f"candidate rank {rank} bbox",
        ),
        "patch_logit": _finite_float(
            value.get("patch_logit"), label=f"candidate rank {rank} patch_logit"
        ),
    }


def validate_extraction_candidates(
    candidates: Any,
    *,
    candidate_topk: int,
    candidate_set_sha256: Any,
) -> list[dict[str, Any]]:
    if not isinstance(candidates, list) or len(candidates) != int(candidate_topk):
        raise ExactTopKContractError(
            f"candidate list must contain exactly K={candidate_topk} rows"
        )
    payloads = [
        candidate_payload(candidate, expected_rank=rank)
        for rank, candidate in enumerate(candidates)
    ]
    query_indices = [row["query_index"] for row in payloads]
    if len(set(query_indices)) != len(query_indices):
        raise ExactTopKContractError("candidate query indices are not unique")
    observed = canonical_sha256(payloads)
    if candidate_set_sha256 != observed:
        raise ExactTopKContractError(
            "candidate_set_sha256 does not bind the complete ordered candidate list"
        )
    for source, payload in zip(candidates, payloads):
        if source.get("candidate_sha256") != canonical_sha256(payload):
            raise ExactTopKContractError(
                f"candidate rank {payload['rank']} candidate_sha256 drifted"
            )
    return payloads


def validate_support_patch(value: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactTopKContractError("exact row has no fixed support patch")
    path = value.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ExactTopKContractError("fixed support patch path is missing")
    if not is_sha256(value.get("sha256")):
        raise ExactTopKContractError("fixed support patch sha256 is malformed")
    try:
        class_id = int(value.get("class_id"))
    except (TypeError, ValueError) as error:
        raise ExactTopKContractError("fixed support patch class_id is invalid") from error
    if class_id < 0:
        raise ExactTopKContractError("fixed support patch class_id must be non-negative")
    if value.get("transform_contract_sha256") != contract[
        "support_transform_contract_sha256"
    ]:
        raise ExactTopKContractError("fixed support patch transform contract drifted")
    return {
        "path": path,
        "sha256": str(value["sha256"]),
        "class_id": class_id,
        "transform_contract_sha256": str(value["transform_contract_sha256"]),
    }


def validate_exact_pair_row(
    row: Any,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract = normalize_exact_contract(expected_contract)
    if not isinstance(row, Mapping):
        raise ExactTopKContractError("exact pair row is not an object")
    if row.get("adapter_pair_schema") != EXACT_TOPK_PAIR_SCHEMA:
        raise ExactTopKContractError("exact pair schema is invalid")
    if row.get("exact_topk_protocol") != EXACT_TOPK_PROTOCOL:
        raise ExactTopKContractError("exact pair protocol is invalid")
    required_claims = {
        "tn_scope": EXACT_TOPK_TN_SCOPE,
        "global_tn_verified": True,
        "fixed_stagea_topk_exact_verified": True,
        "proposalset_proxy_verified": False,
        "all_stagea_topk_candidates_verified": True,
        "all_stagea_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": False,
        "image_global_semantic_absence_proven": False,
        "portable_to_other_checkpoint_or_transform": False,
    }
    for key, expected in required_claims.items():
        if row.get(key) != expected:
            raise ExactTopKContractError(
                f"exact pair claim {key} must be exactly {expected!r}"
            )
    row_contract = normalize_exact_contract(row.get("fixed_stagea_exact_contract"))
    if row_contract != contract:
        raise ExactTopKContractError("exact pair provenance does not match dataset contract")
    topk = contract["candidate_topk"]
    candidates = row.get("fixed_stagea_candidates")
    payloads = validate_extraction_candidates(
        candidates,
        candidate_topk=topk,
        candidate_set_sha256=row.get("fixed_stagea_candidate_set_sha256"),
    )
    judgment_bindings = []
    for rank, candidate in enumerate(candidates):
        if candidate.get("answer") != "no":
            raise ExactTopKContractError(
                f"exact pair candidate rank {rank} is not explicitly judged no"
            )
        for key in ("candidate_sha256", "judgment_sha256", "evidence_sha256"):
            if not is_sha256(candidate.get(key)):
                raise ExactTopKContractError(
                    f"exact pair candidate rank {rank} has invalid {key}"
                )
        confidence = _finite_float(
            candidate.get("confidence"),
            label=f"candidate rank {rank} judgment confidence",
        )
        if not 0.0 <= confidence <= 1.0:
            raise ExactTopKContractError(
                f"candidate rank {rank} judgment confidence is outside [0,1]"
            )
        judgment_bindings.append(
            {
                "rank": rank,
                "candidate_sha256": candidate["candidate_sha256"],
                "judgment_sha256": candidate["judgment_sha256"],
                "evidence_sha256": candidate["evidence_sha256"],
            }
        )
    if row.get("fixed_stagea_judgment_coverage_sha256") != canonical_sha256(
        judgment_bindings
    ):
        raise ExactTopKContractError("candidate judgment coverage hash drifted")
    coverage = row.get("fixed_stagea_candidate_coverage")
    expected_coverage = {
        "expected": topk,
        "observed": topk,
        "verified_ranks": list(range(topk)),
        "all_candidates_complete": True,
        "all_candidates_no": True,
    }
    if coverage != expected_coverage:
        raise ExactTopKContractError("exact pair candidate coverage is incomplete")
    support = validate_support_patch(row.get("fixed_stagea_support_patch"), contract)
    for key in (
        "fixed_stagea_extraction_row_sha256",
        "fixed_stagea_judge_contract_sha256",
    ):
        if not is_sha256(row.get(key)):
            raise ExactTopKContractError(f"exact pair has invalid {key}")
    if int(row.get("class_id", -1)) != support["class_id"]:
        raise ExactTopKContractError("fixed support patch class does not match pair class_id")
    return {
        "candidate_indices": [item["query_index"] for item in payloads],
        "candidate_boxes": [item["bbox_cxcywh_normalized"] for item in payloads],
        "support_patch": support,
        "contract": contract,
    }


def _validate_file_record(value: Any, *, expected_path: Path | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        raise ExactTopKContractError("audit file record is missing")
    path = remap_legacy_path(str(value["path"])).expanduser().resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ExactTopKContractError(
            f"audit annotation path mismatch: {path} != {expected_path.resolve()}"
        )
    rows = value.get("rows")
    if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows < 0):
        raise ExactTopKContractError("audit file record rows is invalid")
    current = file_record(path, rows=rows if isinstance(rows, int) else None)
    for key in ("path", "size_bytes", "sha256"):
        if value.get(key) != current[key]:
            raise ExactTopKContractError(f"audit file record {key} drifted for {path}")
    if isinstance(rows, int):
        with path.open("r", encoding="utf-8") as handle:
            observed_rows = sum(1 for line in handle if line.strip())
        if observed_rows != rows:
            raise ExactTopKContractError(
                f"audit file record row count drifted for {path}: "
                f"audited={rows}, current={observed_rows}"
            )
    return current


def validate_exact_pair_collection(
    rows: Iterable[Mapping[str, Any]],
    *,
    annotation_path: Path,
    audit_path: Path,
    expected_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contract = normalize_exact_contract(expected_contract)
    audit_path = remap_legacy_path(audit_path).expanduser().resolve()
    if not audit_path.is_file():
        raise ExactTopKContractError(f"exact verification audit is missing: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExactTopKContractError(f"invalid exact verification audit: {error}") from error
    if not isinstance(audit, Mapping):
        raise ExactTopKContractError("exact verification audit is not an object")
    if (
        audit.get("schema") != EXACT_TOPK_RESULT_AUDIT_SCHEMA
        or audit.get("protocol") != EXACT_TOPK_PROTOCOL
        or audit.get("complete") is not True
        or audit.get("tn_scope") != EXACT_TOPK_TN_SCOPE
    ):
        raise ExactTopKContractError("exact verification audit contract is invalid")
    audit_contract = normalize_exact_contract(audit.get("exact_contract"))
    if audit_contract != contract:
        raise ExactTopKContractError("exact verification audit provenance mismatches config")
    _validate_file_record(audit.get("annotation"), expected_path=annotation_path)
    inputs = audit.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "extractions",
        "extraction_audit",
        "judgments",
        "judgment_audit",
    }:
        raise ExactTopKContractError("exact verification audit input closure is incomplete")
    input_records = {
        name: _validate_file_record(record)
        for name, record in inputs.items()
    }
    _validate_file_record(audit.get("decisions"))
    source_rows = audit.get("source_rows")
    candidate_judgments = audit.get("candidate_judgments")
    if (
        not isinstance(source_rows, int)
        or isinstance(source_rows, bool)
        or source_rows <= 0
        or candidate_judgments != source_rows * contract["candidate_topk"]
        or input_records["extractions"].get("rows") != source_rows
        or input_records["judgments"].get("rows") != candidate_judgments
    ):
        raise ExactTopKContractError("exact verification audit coverage totals drifted")
    judge_contract = audit.get("judge_contract")
    if not isinstance(judge_contract, Mapping):
        raise ExactTopKContractError("exact verification audit judge contract is missing")
    judge_payload = dict(judge_contract)
    judge_sha = judge_payload.pop("sha256", None)
    if not is_sha256(judge_sha) or canonical_sha256(judge_payload) != judge_sha:
        raise ExactTopKContractError("exact verification judge contract hash drifted")
    try:
        min_no_confidence = float(judge_payload.get("min_no_confidence"))
    except (TypeError, ValueError) as error:
        raise ExactTopKContractError(
            "exact verification min_no_confidence is invalid"
        ) from error
    if not 0.0 <= min_no_confidence <= 1.0:
        raise ExactTopKContractError(
            "exact verification min_no_confidence is outside [0,1]"
        )
    rows = list(rows)
    if not rows:
        raise ExactTopKContractError("exact verified training annotation is empty")
    if int(audit.get("accepted_rows", -1)) != len(rows):
        raise ExactTopKContractError("exact verification audit row count drifted")
    validated = [
        validate_exact_pair_row(row, expected_contract=contract) for row in rows
    ]
    for row in rows:
        if row.get("fixed_stagea_judge_contract_sha256") != judge_sha:
            raise ExactTopKContractError("exact pair judge contract binding drifted")
        if any(
            float(candidate["confidence"]) < min_no_confidence
            for candidate in row["fixed_stagea_candidates"]
        ):
            raise ExactTopKContractError(
                "exact pair contains a below-threshold no judgment"
            )
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(
        sample_ids
    ):
        raise ExactTopKContractError("exact pair sample_id values are missing or duplicate")
    return validated


__all__ = [
    "EXACT_CONTRACT_KEYS",
    "EXACT_CONTRACT_SHA_KEYS",
    "EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA",
    "EXACT_TOPK_EXTRACTION_SCHEMA",
    "EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA",
    "EXACT_TOPK_JUDGMENT_SCHEMA",
    "EXACT_TOPK_PAIR_SCHEMA",
    "EXACT_TOPK_PROTOCOL",
    "EXACT_TOPK_RESULT_AUDIT_SCHEMA",
    "EXACT_TOPK_TN_SCOPE",
    "ExactTopKContractError",
    "candidate_payload",
    "canonical_json",
    "canonical_sha256",
    "file_record",
    "is_sha256",
    "normalize_exact_contract",
    "sha256_file",
    "validate_exact_pair_collection",
    "validate_exact_pair_row",
    "validate_extraction_candidates",
    "validate_support_patch",
]
