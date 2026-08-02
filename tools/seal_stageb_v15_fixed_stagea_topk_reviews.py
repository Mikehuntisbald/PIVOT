#!/usr/bin/env python3
"""Seal externally supplied D4 review decisions into builder-ready judgments.

This command never calls a judge and never supplies a default answer.  It
accepts exactly one explicit decision for every pending review key, preserves
all extraction/evidence bindings, computes canonical judgment hashes, and then
writes the completed judgment audit consumed by the exact-pair CPU builder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_stageb_v15_fixed_stagea_topk_exact_pairs import (  # noqa: E402
    _judgment_payload,
    _load_judgments,
    _validate_judgment_audit,
)
from tools.render_stageb_v15_fixed_stagea_topk_evidence import (  # noqa: E402
    validate_review_bundle,
)
from util.stageb_exact_topk_contract import (  # noqa: E402
    EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA,
    EXACT_TOPK_PROTOCOL,
    ExactTopKContractError,
    canonical_sha256,
    file_record,
)


DECISION_SCHEMA = "stage-b-v15-fixed-stagea-topk-external-review-decision-v1"


class SealError(RuntimeError):
    pass


def _iter_jsonl(path: Path, *, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SealError(f"missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SealError(f"blank row at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SealError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise SealError(f"non-object row at {path}:{line_number}")
            yield line_number, value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                dict(value),
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


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
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


def _load_pending(
    review_path: Path,
    review_audit_path: Path,
    *,
    extraction_path: Path,
    extraction_audit_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    review_audit, _judge_contract, pending, extraction_rows = (
        validate_review_bundle(
            extraction_path=extraction_path,
            extraction_audit_path=extraction_audit_path,
            review_path=review_path,
            audit_path=review_audit_path,
        )
    )
    return review_audit, pending, extraction_rows


def _load_decisions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, row in _iter_jsonl(path, label="external review decisions"):
        if row.get("schema") != DECISION_SCHEMA:
            raise SealError(f"decision row {line_number} has invalid schema")
        allowed = {
            "schema",
            "sample_id",
            "candidate_rank",
            "answer",
            "confidence",
            "reviewer_note",
        }
        if set(row).difference(allowed):
            raise SealError(
                f"decision row {line_number} has unrecognized provenance fields"
            )
        raw_sample_id = row.get("sample_id")
        raw_rank = row.get("candidate_rank")
        raw_confidence = row.get("confidence")
        if (
            not isinstance(raw_sample_id, str)
            or not raw_sample_id
            or not isinstance(raw_rank, int)
            or isinstance(raw_rank, bool)
            or isinstance(raw_confidence, bool)
        ):
            raise SealError(
                f"decision row {line_number} has invalid key/confidence"
            )
        try:
            key = (raw_sample_id, raw_rank)
            confidence = float(raw_confidence)
        except (KeyError, TypeError, ValueError) as error:
            raise SealError(
                f"decision row {line_number} has invalid key/confidence"
            ) from error
        if not key[0] or key[1] < 0:
            raise SealError(f"decision row {line_number} has invalid key")
        if row.get("answer") not in {"no", "yes", "uncertain"}:
            raise SealError(f"decision row {line_number} has invalid answer")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise SealError(f"decision row {line_number} confidence is outside [0,1]")
        if row.get("reviewer_note") is not None and not isinstance(
            row.get("reviewer_note"), str
        ):
            raise SealError(f"decision row {line_number} reviewer_note is not text")
        if key in result:
            raise SealError(f"duplicate external decision: {key}")
        result[key] = {**row, "confidence": confidence}
    return result


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    extraction_path = Path(args.extractions).expanduser().resolve()
    extraction_audit_path = Path(args.extraction_audit).expanduser().resolve()
    review_path = Path(args.review_manifest).expanduser().resolve()
    review_audit_path = Path(args.review_audit).expanduser().resolve()
    decisions_path = Path(args.completed_reviews).expanduser().resolve()
    review_audit, pending, extraction_rows = _load_pending(
        review_path,
        review_audit_path,
        extraction_path=extraction_path,
        extraction_audit_path=extraction_audit_path,
    )
    decisions = _load_decisions(decisions_path)
    pending_keys = {
        (str(row["sample_id"]), int(row["candidate_rank"])) for row in pending
    }
    if set(decisions) != pending_keys:
        missing = len(pending_keys.difference(decisions))
        orphan = len(set(decisions).difference(pending_keys))
        raise SealError(
            f"external review coverage is not exact: missing={missing}, orphan={orphan}"
        )
    judgments: list[dict[str, Any]] = []
    for row in pending:
        key = (str(row["sample_id"]), int(row["candidate_rank"]))
        decision = decisions[key]
        judgment = dict(row)
        judgment["status"] = "complete"
        judgment["answer"] = decision["answer"]
        judgment["confidence"] = decision["confidence"]
        judgment["external_review_decision_sha256"] = canonical_sha256(decision)
        if decision.get("reviewer_note") is not None:
            judgment["reviewer_note"] = str(decision["reviewer_note"])
        judgment["judgment_sha256"] = canonical_sha256(_judgment_payload(judgment))
        judgments.append(judgment)
    return review_audit, judgments, extraction_rows


def seal(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.judgments).expanduser().resolve()
    audit_path = Path(args.judgment_audit).expanduser().resolve()
    if output.exists() or audit_path.exists():
        if output.is_file() and audit_path.is_file():
            return verify(args)
        raise SealError("partial finalized judgment/audit exists; refuse overwrite")
    review_audit, judgments, extraction_rows = prepare(args)
    _atomic_jsonl(output, judgments)
    extraction_path = Path(args.extractions).expanduser().resolve()
    audit = {
        "schema": EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA,
        "protocol": EXACT_TOPK_PROTOCOL,
        "complete": True,
        "rows": len(judgments),
        "extractions": file_record(extraction_path, rows=extraction_rows),
        "judgments": file_record(output, rows=len(judgments)),
        "judge_contract": review_audit["judge_contract"],
        "inputs": {
            "extraction_audit": file_record(Path(args.extraction_audit)),
            "pending_review_manifest": file_record(
                Path(args.review_manifest), rows=len(judgments)
            ),
            "pending_review_audit": file_record(Path(args.review_audit)),
            "external_review_decisions": file_record(
                Path(args.completed_reviews), rows=len(judgments)
            ),
        },
        "review_complete": True,
        "answers_generated_by_tool": False,
        "judgment_rows_packaged_by_tool": True,
        "judgments_source": "explicit_external_review_decisions",
    }
    _atomic_json(audit_path, audit)
    return audit


def verify(args: argparse.Namespace) -> dict[str, Any]:
    extraction_path = Path(args.extractions).expanduser().resolve()
    output = Path(args.judgments).expanduser().resolve()
    audit_path = Path(args.judgment_audit).expanduser().resolve()
    review_audit, expected, extraction_rows = prepare(args)
    audit, judge_contract, _minimum = _validate_judgment_audit(
        audit_path,
        output,
        extraction_path=extraction_path,
        extraction_rows=extraction_rows,
    )
    if judge_contract != review_audit["judge_contract"]:
        raise SealError("sealed judge contract drifted")
    loaded = _load_judgments(
        output, judge_contract_sha256=judge_contract["sha256"]
    )
    expected_by_key = {
        (str(row["sample_id"]), int(row["candidate_rank"])): row for row in expected
    }
    if loaded != expected_by_key:
        raise SealError("sealed judgments differ from explicit review decisions")
    if (
        audit.get("answers_generated_by_tool") is not False
        or audit.get("judgment_rows_packaged_by_tool") is not True
    ):
        raise SealError("sealed audit falsely describes answer provenance")
    return {
        "schema": EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA,
        "verified": True,
        "rows": len(loaded),
        "judgments": file_record(output, rows=len(loaded)),
        "audit": file_record(audit_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--extraction-audit", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--review-audit", type=Path, required=True)
    parser.add_argument("--completed-reviews", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--judgment-audit", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.verify_only:
            result = verify(args)
        elif args.dry_run:
            review_audit, judgments, extraction_rows = prepare(args)
            result = {
                "schema": EXACT_TOPK_JUDGMENT_AUDIT_SCHEMA,
                "kind": "dry_run_no_outputs_written",
                "source_rows": extraction_rows,
                "judgment_rows": len(judgments),
                "judge_contract": review_audit["judge_contract"],
                "all_decisions_explicit": True,
            }
        else:
            result = seal(args)
    except (SealError, ExactTopKContractError, RuntimeError, ValueError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
