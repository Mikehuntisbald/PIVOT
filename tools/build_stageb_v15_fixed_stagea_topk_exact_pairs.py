#!/usr/bin/env python3
"""Build Stage-B TN pairs from exhaustively reviewed fixed Stage-A Top-K rows.

This is a CPU-only verifier/builder. Candidate extraction is a separate GPU
step and visual judgments are a separate human/VLM step. This command refuses
partial coverage, mixed provenance, orphan judgments, changed files, or legacy
cached-proposal/semantic scopes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.path_compat import remap_legacy_path
from util.stageb_exact_topk_contract import (
    EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA,
    EXACT_TOPK_EXTRACTION_SCHEMA,
    EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA,
    EXACT_TOPK_JUDGMENT_SCHEMA,
    EXACT_TOPK_PAIR_SCHEMA,
    EXACT_TOPK_PROTOCOL,
    EXACT_TOPK_RESULT_AUDIT_SCHEMA,
    EXACT_TOPK_TN_SCOPE,
    ExactTopKContractError,
    canonical_sha256,
    file_record,
    is_sha256,
    normalize_exact_contract,
    sha256_file,
    validate_exact_pair_row,
    validate_extraction_candidates,
    validate_support_patch,
)


DECISION_SCHEMA = "stage-b-v15-fixed-stagea-topk-exact-decision-v1"
JUDGE_CONTRACT_SCHEMA = "stage-b-v15-fixed-stagea-topk-judge-contract-v1"
CANDIDATE_SELECTION_SCHEMA = (
    "stage-b-v15-fixed-stagea-topk-candidate-selection-contract-v1"
)


class ExactTopKBuildError(RuntimeError):
    pass


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = remap_legacy_path(path).expanduser().resolve()
    if not path.is_file():
        raise ExactTopKBuildError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExactTopKBuildError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExactTopKBuildError(f"{label} is not a JSON object: {path}")
    return value


def _iter_jsonl(path: Path, *, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    path = remap_legacy_path(path).expanduser().resolve()
    if not path.is_file():
        raise ExactTopKBuildError(f"missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ExactTopKBuildError(f"blank row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExactTopKBuildError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ExactTopKBuildError(f"non-object row at {path}:{line_number}")
            yield line_number, row


def _count_rows(path: Path) -> int:
    return sum(1 for _line, _row in _iter_jsonl(path, label="JSONL"))


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_record(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value.get("path"):
        raise ExactTopKBuildError(f"{label} file record is missing")
    path = remap_legacy_path(str(value["path"])).expanduser().resolve()
    if expected_path is not None and path != expected_path.expanduser().resolve():
        raise ExactTopKBuildError(
            f"{label} path drifted: audited={path}, expected={expected_path.resolve()}"
        )
    current = file_record(path)
    for key in ("path", "size_bytes", "sha256"):
        if value.get(key) != current[key]:
            raise ExactTopKBuildError(f"{label} {key} drifted")
    if expected_rows is not None:
        rows = _count_rows(path)
        if rows != expected_rows or value.get("rows") != rows:
            raise ExactTopKBuildError(
                f"{label} row count drifted: audited={value.get('rows')}, current={rows}"
            )
        current["rows"] = rows
    return current


def _validate_hashed_contract(
    value: Any,
    *,
    label: str,
    expected_sha256: str,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExactTopKBuildError(f"{label} contract is missing")
    payload = dict(value)
    observed_sha = payload.pop("sha256", None)
    if observed_sha != expected_sha256 or canonical_sha256(payload) != observed_sha:
        raise ExactTopKBuildError(f"{label} canonical hash drifted")
    if expected_schema is not None and payload.get("schema") != expected_schema:
        raise ExactTopKBuildError(f"{label} schema drifted")
    return payload


def _validate_extraction_audit(
    audit_path: Path,
    extraction_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = _read_json(audit_path, label="extraction audit")
    if (
        audit.get("schema") != EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA
        or audit.get("protocol") != EXACT_TOPK_PROTOCOL
        or audit.get("complete") is not True
    ):
        raise ExactTopKBuildError("extraction audit is not a completed exact run")
    try:
        contract = normalize_exact_contract(audit.get("exact_contract"))
    except ExactTopKContractError as error:
        raise ExactTopKBuildError(str(error)) from error
    rows = audit.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ExactTopKBuildError("extraction audit rows must be a positive integer")
    _validate_record(
        audit.get("extractions"),
        label="extractions",
        expected_path=extraction_path,
        expected_rows=rows,
    )
    provenance = audit.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ExactTopKBuildError("extraction audit provenance is missing")
    for name, contract_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("model_config", "model_config_sha256"),
        ("data_config", "data_config_sha256"),
        ("canonical_classes", "canonical_classes_sha256"),
    ):
        record = _validate_record(provenance.get(name), label=name)
        if record["sha256"] != contract[contract_key]:
            raise ExactTopKBuildError(f"{name} does not match the exact contract")
    _validate_hashed_contract(
        audit.get("query_transform_contract"),
        label="query transform",
        expected_sha256=contract["query_transform_contract_sha256"],
    )
    _validate_hashed_contract(
        audit.get("support_transform_contract"),
        label="support transform",
        expected_sha256=contract["support_transform_contract_sha256"],
    )
    selection = _validate_hashed_contract(
        audit.get("candidate_selection_contract"),
        label="candidate selection",
        expected_sha256=contract["candidate_selection_contract_sha256"],
        expected_schema=CANDIDATE_SELECTION_SCHEMA,
    )
    required_selection = {
        "candidate_topk": contract["candidate_topk"],
        "score_source": "score_patch_logits",
        "selection": "torch.topk(largest=true,sorted=true)",
        "candidate_order": "descending_patch_logit",
        "candidate_box_space": "normalized_cxcywh",
        "fixed_support_patch_per_row": True,
        "deterministic_query_transform": True,
        "dynamic_candidate_replay_must_match": True,
        "candidate_box_atol": contract["candidate_box_atol"],
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise ExactTopKBuildError(
                f"candidate selection {key} must be exactly {expected!r}"
            )
    return audit, contract


def _validate_judgment_audit(
    audit_path: Path,
    judgment_path: Path,
    *,
    extraction_path: Path,
    extraction_rows: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    audit = _read_json(audit_path, label="judgment audit")
    if (
        audit.get("schema") != EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA
        or audit.get("protocol") != EXACT_TOPK_PROTOCOL
        or audit.get("complete") is not True
    ):
        raise ExactTopKBuildError("judgment audit is not a completed exact run")
    _validate_record(
        audit.get("extractions"),
        label="judgment-audit extractions",
        expected_path=extraction_path,
        expected_rows=extraction_rows,
    )
    judgment_rows = audit.get("rows")
    if not isinstance(judgment_rows, int) or isinstance(judgment_rows, bool):
        raise ExactTopKBuildError("judgment audit rows is invalid")
    _validate_record(
        audit.get("judgments"),
        label="judgments",
        expected_path=judgment_path,
        expected_rows=judgment_rows,
    )
    contract_value = audit.get("judge_contract")
    if not isinstance(contract_value, Mapping):
        raise ExactTopKBuildError("judge contract is missing")
    judge_contract = dict(contract_value)
    judge_sha = judge_contract.pop("sha256", None)
    if (
        not is_sha256(judge_sha)
        or canonical_sha256(judge_contract) != judge_sha
        or judge_contract.get("schema") != JUDGE_CONTRACT_SCHEMA
    ):
        raise ExactTopKBuildError("judge contract canonical hash drifted")
    if judge_contract.get("judge_type") not in {"human", "model", "hybrid"}:
        raise ExactTopKBuildError("judge_type must be human, model, or hybrid")
    for key in ("prompt_template_sha256", "evidence_asset_policy_sha256"):
        if not is_sha256(judge_contract.get(key)):
            raise ExactTopKBuildError(f"judge contract has invalid {key}")
    try:
        min_confidence = float(judge_contract.get("min_no_confidence"))
    except (TypeError, ValueError) as error:
        raise ExactTopKBuildError("min_no_confidence is invalid") from error
    if not 0.0 <= min_confidence <= 1.0:
        raise ExactTopKBuildError("min_no_confidence must be in [0,1]")
    return audit, {**judge_contract, "sha256": judge_sha}, min_confidence


def _validate_image_record(value: Any, *, label: str) -> dict[str, Any]:
    return _validate_record(value, label=label)


def _validate_extraction_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    exact_contract: Mapping[str, Any],
    support_hash_cache: dict[Path, str],
) -> dict[str, Any]:
    context = f"extraction row {line_number}"
    if (
        row.get("schema") != EXACT_TOPK_EXTRACTION_SCHEMA
        or row.get("protocol") != EXACT_TOPK_PROTOCOL
    ):
        raise ExactTopKBuildError(f"{context} schema/protocol is invalid")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ExactTopKBuildError(f"{context} has no sample_id")
    try:
        row_contract = normalize_exact_contract(row.get("exact_contract"))
    except ExactTopKContractError as error:
        raise ExactTopKBuildError(f"{context}: {error}") from error
    if row_contract != exact_contract:
        raise ExactTopKBuildError(f"{context} mixes exact provenance")
    source_pair = row.get("source_pair")
    if not isinstance(source_pair, Mapping):
        raise ExactTopKBuildError(f"{context} has no source_pair")
    if row.get("source_pair_sha256") != canonical_sha256(source_pair):
        raise ExactTopKBuildError(f"{context} source_pair hash drifted")
    if source_pair.get("sample_id") != sample_id:
        raise ExactTopKBuildError(f"{context} source pair sample_id drifted")
    if str(source_pair.get("split", "")).strip().lower() != "train":
        raise ExactTopKBuildError(f"{context} source pair is not a train row")
    positive = str(source_pair.get("sent", "")).strip()
    negative = str(source_pair.get("try_tn", "")).strip()
    if not positive or not negative or positive.casefold() == negative.casefold():
        raise ExactTopKBuildError(f"{context} source pair text is invalid")
    try:
        class_id = int(source_pair.get("class_id"))
        bbox = [float(value) for value in source_pair.get("target_bbox_used")]
    except (TypeError, ValueError) as error:
        raise ExactTopKBuildError(f"{context} source pair class/bbox is invalid") from error
    if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise ExactTopKBuildError(f"{context} source pair bbox is invalid")
    image = _validate_image_record(row.get("image"), label=f"{context} image")
    source_image_path = source_pair.get("image_path")
    if not isinstance(source_image_path, str) or not source_image_path.strip():
        raise ExactTopKBuildError(
            f"{context} source pair must bind the absolute extracted image_path"
        )
    if (
        remap_legacy_path(source_image_path).expanduser().resolve()
        != Path(image["path"]).resolve()
    ):
        raise ExactTopKBuildError(f"{context} extracted image path drifted")
    transform_trace = row.get("query_transform_trace")
    if not isinstance(transform_trace, Mapping):
        raise ExactTopKBuildError(f"{context} query transform trace is missing")
    if row.get("query_transform_trace_sha256") != canonical_sha256(
        transform_trace
    ):
        raise ExactTopKBuildError(f"{context} query transform trace hash drifted")
    try:
        support = validate_support_patch(row.get("fixed_support_patch"), exact_contract)
    except ExactTopKContractError as error:
        raise ExactTopKBuildError(f"{context}: {error}") from error
    support_path = remap_legacy_path(support["path"]).expanduser().resolve()
    if not support_path.is_file():
        raise ExactTopKBuildError(f"{context} fixed support patch is missing: {support_path}")
    observed_support_sha = support_hash_cache.get(support_path)
    if observed_support_sha is None:
        observed_support_sha = sha256_file(support_path)
        support_hash_cache[support_path] = observed_support_sha
    if observed_support_sha != support["sha256"]:
        raise ExactTopKBuildError(f"{context} fixed support patch hash drifted")
    try:
        payloads = validate_extraction_candidates(
            row.get("candidates"),
            candidate_topk=exact_contract["candidate_topk"],
            candidate_set_sha256=row.get("candidate_set_sha256"),
        )
    except ExactTopKContractError as error:
        raise ExactTopKBuildError(f"{context}: {error}") from error
    if class_id != support["class_id"]:
        raise ExactTopKBuildError(f"{context} support class does not match source pair")
    return {
        "sample_id": sample_id.strip(),
        "source_pair": dict(source_pair),
        "image": image,
        "support": support,
        "candidate_payloads": payloads,
        "candidates": row["candidates"],
        "candidate_set_sha256": row["candidate_set_sha256"],
        "extraction_row_sha256": canonical_sha256(row),
    }


def _judgment_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "judgment_sha256"}


def _load_judgments(
    path: Path,
    *,
    judge_contract_sha256: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, row in _iter_jsonl(path, label="judgments"):
        context = f"judgment row {line_number}"
        if (
            row.get("schema") != EXACT_TOPK_JUDGMENT_SCHEMA
            or row.get("protocol") != EXACT_TOPK_PROTOCOL
            or row.get("status") != "complete"
            or row.get("judge_contract_sha256") != judge_contract_sha256
        ):
            raise ExactTopKBuildError(f"{context} contract is incomplete or mixed")
        sample_id = row.get("sample_id")
        try:
            rank = int(row.get("candidate_rank"))
            query_index = int(row.get("query_index"))
            confidence = float(row.get("confidence"))
        except (TypeError, ValueError) as error:
            raise ExactTopKBuildError(f"{context} identifiers/confidence are invalid") from error
        if not isinstance(sample_id, str) or not sample_id or rank < 0 or query_index < 0:
            raise ExactTopKBuildError(f"{context} identifiers are invalid")
        if row.get("answer") not in {"no", "yes", "uncertain"}:
            raise ExactTopKBuildError(f"{context} answer is invalid")
        if not 0.0 <= confidence <= 1.0:
            raise ExactTopKBuildError(f"{context} confidence is outside [0,1]")
        for key_name in (
            "candidate_sha256",
            "candidate_set_sha256",
            "evidence_sha256",
            "extraction_row_sha256",
        ):
            if not is_sha256(row.get(key_name)):
                raise ExactTopKBuildError(f"{context} has invalid {key_name}")
        if row.get("judgment_sha256") != canonical_sha256(_judgment_payload(row)):
            raise ExactTopKBuildError(f"{context} judgment hash drifted")
        key = (sample_id, rank)
        if key in result:
            raise ExactTopKBuildError(f"duplicate judgment for {key}")
        result[key] = dict(row)
    return result


def _bind_judgments(
    extraction: Mapping[str, Any],
    judgments: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    min_no_confidence: float,
) -> tuple[str, list[dict[str, Any]], str]:
    bound: list[dict[str, Any]] = []
    reasons = []
    for rank, (source, payload) in enumerate(
        zip(extraction["candidates"], extraction["candidate_payloads"])
    ):
        judgment = judgments.get((extraction["sample_id"], rank))
        if judgment is None:
            raise ExactTopKBuildError(
                f"missing judgment for {(extraction['sample_id'], rank)}"
            )
        expected_candidate_sha = canonical_sha256(payload)
        bindings = {
            "query_index": payload["query_index"],
            "candidate_sha256": expected_candidate_sha,
            "candidate_set_sha256": extraction["candidate_set_sha256"],
            "extraction_row_sha256": extraction["extraction_row_sha256"],
        }
        for key, expected in bindings.items():
            if judgment.get(key) != expected:
                raise ExactTopKBuildError(
                    f"judgment binding {key} drifted for {(extraction['sample_id'], rank)}"
                )
        answer = str(judgment["answer"])
        confidence = float(judgment["confidence"])
        if answer == "yes":
            reasons.append("candidate_yes")
        elif answer != "no":
            reasons.append("candidate_uncertain")
        elif confidence < min_no_confidence:
            reasons.append("candidate_no_below_confidence")
        bound.append(
            {
                **payload,
                "candidate_sha256": expected_candidate_sha,
                "answer": answer,
                "confidence": confidence,
                "judgment_sha256": judgment["judgment_sha256"],
                "evidence_sha256": judgment["evidence_sha256"],
            }
        )
    if "candidate_yes" in reasons:
        return "rejected", bound, "candidate_yes"
    if reasons:
        return "quarantine", bound, sorted(reasons)[0]
    return "accepted", bound, "all_topk_candidates_complete_confident_no"


def _make_pair(
    extraction: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    *,
    exact_contract: Mapping[str, Any],
    judge_contract_sha256: str,
) -> dict[str, Any]:
    topk = int(exact_contract["candidate_topk"])
    judgment_bindings = [
        {
            "rank": rank,
            "candidate_sha256": candidate["candidate_sha256"],
            "judgment_sha256": candidate["judgment_sha256"],
            "evidence_sha256": candidate["evidence_sha256"],
        }
        for rank, candidate in enumerate(candidates)
    ]
    output = dict(extraction["source_pair"])
    output.update(
        {
            "adapter_pair_schema": EXACT_TOPK_PAIR_SCHEMA,
            "source": "stage_b_v15_fixed_stagea_topk_exact",
            "sample_id": extraction["sample_id"],
            "exact_topk_protocol": EXACT_TOPK_PROTOCOL,
            "tn_scope": EXACT_TOPK_TN_SCOPE,
            "global_tn_verified": True,
            "fixed_stagea_topk_exact_verified": True,
            "proposalset_proxy_verified": False,
            "all_stagea_topk_candidates_verified": True,
            "all_stagea_queries_verified": False,
            "global_max_label_is_semantic_extrapolation": False,
            "image_global_semantic_absence_proven": False,
            "portable_to_other_checkpoint_or_transform": False,
            "fixed_stagea_exact_contract": dict(exact_contract),
            "fixed_stagea_support_patch": dict(extraction["support"]),
            "fixed_stagea_candidate_set_sha256": extraction[
                "candidate_set_sha256"
            ],
            "fixed_stagea_extraction_row_sha256": extraction[
                "extraction_row_sha256"
            ],
            "fixed_stagea_judge_contract_sha256": judge_contract_sha256,
            "fixed_stagea_candidates": candidates,
            "fixed_stagea_judgment_coverage_sha256": canonical_sha256(
                judgment_bindings
            ),
            "fixed_stagea_candidate_coverage": {
                "expected": topk,
                "observed": topk,
                "verified_ranks": list(range(topk)),
                "all_candidates_complete": True,
                "all_candidates_no": True,
            },
        }
    )
    try:
        validate_exact_pair_row(output, expected_contract=exact_contract)
    except ExactTopKContractError as error:
        raise ExactTopKBuildError(f"built exact pair failed self-check: {error}") from error
    return output


def build(args: argparse.Namespace) -> dict[str, Any]:
    extraction_path = Path(args.extractions).expanduser().resolve()
    extraction_audit_path = Path(args.extraction_audit).expanduser().resolve()
    judgment_path = Path(args.judgments).expanduser().resolve()
    judgment_audit_path = Path(args.judgment_audit).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    decisions_path = Path(args.decisions).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()

    extraction_audit, exact_contract = _validate_extraction_audit(
        extraction_audit_path, extraction_path
    )
    extraction_rows = int(extraction_audit["rows"])
    judgment_audit, judge_contract, min_confidence = _validate_judgment_audit(
        judgment_audit_path,
        judgment_path,
        extraction_path=extraction_path,
        extraction_rows=extraction_rows,
    )
    judgments = _load_judgments(
        judgment_path, judge_contract_sha256=judge_contract["sha256"]
    )
    expected_judgments = extraction_rows * int(exact_contract["candidate_topk"])
    if len(judgments) != expected_judgments:
        raise ExactTopKBuildError(
            f"judgment coverage must be rows*K={expected_judgments}, got {len(judgments)}"
        )

    support_hash_cache: dict[Path, str] = {}
    extractions = []
    seen_samples = set()
    for line_number, row in _iter_jsonl(extraction_path, label="extractions"):
        extraction = _validate_extraction_row(
            row,
            line_number=line_number,
            exact_contract=exact_contract,
            support_hash_cache=support_hash_cache,
        )
        if extraction["sample_id"] in seen_samples:
            raise ExactTopKBuildError(
                f"duplicate extraction sample_id: {extraction['sample_id']}"
            )
        seen_samples.add(extraction["sample_id"])
        extractions.append(extraction)
    if len(extractions) != extraction_rows:
        raise ExactTopKBuildError("extraction row count changed during replay")

    accepted = []
    decisions = []
    used_judgments = set()
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for extraction in extractions:
        decision, candidates, reason = _bind_judgments(
            extraction, judgments, min_no_confidence=min_confidence
        )
        for rank in range(int(exact_contract["candidate_topk"])):
            used_judgments.add((extraction["sample_id"], rank))
        decision_counts[decision] += 1
        reason_counts[reason] += 1
        decisions.append(
            {
                "schema": DECISION_SCHEMA,
                "protocol": EXACT_TOPK_PROTOCOL,
                "sample_id": extraction["sample_id"],
                "decision": decision,
                "reason": reason,
                "candidate_set_sha256": extraction["candidate_set_sha256"],
                "candidate_count": len(candidates),
                "extraction_row_sha256": extraction["extraction_row_sha256"],
            }
        )
        if decision == "accepted":
            accepted.append(
                _make_pair(
                    extraction,
                    candidates,
                    exact_contract=exact_contract,
                    judge_contract_sha256=judge_contract["sha256"],
                )
            )
    orphan = set(judgments).difference(used_judgments)
    if orphan:
        raise ExactTopKBuildError(f"found {len(orphan)} orphan judgment rows")

    accepted.sort(key=lambda row: str(row["sample_id"]))
    decisions.sort(key=lambda row: str(row["sample_id"]))
    _atomic_write_jsonl(output_path, accepted)
    _atomic_write_jsonl(decisions_path, decisions)
    audit = {
        "schema": EXACT_TOPK_RESULT_AUDIT_SCHEMA,
        "protocol": EXACT_TOPK_PROTOCOL,
        "complete": True,
        "tn_scope": EXACT_TOPK_TN_SCOPE,
        "definition": (
            "every rank 0..K-1 candidate is bound to one frozen Stage-A "
            "checkpoint, model/data config, deterministic query transform, "
            "fixed support patch, and complete visual judgment"
        ),
        "exact_contract": dict(exact_contract),
        "judge_contract": dict(judge_contract),
        "inputs": {
            "extractions": file_record(extraction_path, rows=extraction_rows),
            "extraction_audit": file_record(extraction_audit_path),
            "judgments": file_record(judgment_path, rows=len(judgments)),
            "judgment_audit": file_record(judgment_audit_path),
        },
        "source_rows": extraction_rows,
        "candidate_judgments": len(judgments),
        "accepted_rows": len(accepted),
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "annotation": file_record(output_path, rows=len(accepted)),
        "decisions": file_record(decisions_path, rows=len(decisions)),
    }
    _atomic_write_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractions", required=True)
    parser.add_argument("--extraction-audit", required=True)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--judgment-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    try:
        result = build(args)
    except (ExactTopKBuildError, ExactTopKContractError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
