#!/usr/bin/env python3
"""Audit DD3 lexical token roles against the production trace algorithm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.bertwarper import (  # noqa: E402
    generate_masks_with_special_tokens_and_transfer_map,
)
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    build_direct_trace_token_roles,
)
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data/ablations/stageb_tn_table_b_equal_exposure_20260717"
    / "d3_proposal_covered_train.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_trace_audit_20260721"
    / "receipt.json"
)
PRODUCTION_SOURCE = (
    REPO_ROOT / "models/GroundingDINO/stage_b_data_driven_score.py"
)
PRODUCTION_FUNCTION = "build_direct_trace_token_roles"
SOURCE_SHA256 = "c5313440988ae488d785920ddaf2341fc42d72b551f53b1f4b551ff89958b28b"
EXPECTED_ROWS = 14_196
RECEIPT_SCHEMA = "pivot.stageb.data_driven_trace_roles_audit/v2"
TRACE_TOKEN_PATTERN = r"[A-Za-z0-9]+"
TRACE_TOKEN_RE = re.compile(TRACE_TOKEN_PATTERN)
FAILURE_REASONS = (
    "empty_replace_tokens",
    "no_exact_reconstruction",
    "ambiguous_reconstruction_without_declared_span",
    "no_changed_target_tokens",
    "changed_target_index_out_of_bounds",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _clean_caption_phrase(value: Any) -> str:
    text = str(value).replace("_", " ").replace(".", " ").strip()
    return " ".join(text.split())


def _lexical_tokens(value: Any) -> list[dict[str, Any]]:
    text = str(value or "")
    return [
        {
            "text": match.group(0),
            "norm": match.group(0).lower(),
            "start": int(match.start()),
            "end": int(match.end()),
        }
        for match in TRACE_TOKEN_RE.finditer(text)
    ]


def _production_function_record(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == PRODUCTION_FUNCTION
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {PRODUCTION_FUNCTION} definition in {path}, found {len(matches)}"
        )
    node = matches[0]
    if node.end_lineno is None:
        raise RuntimeError("Python AST did not expose the function end line")
    lines = source.splitlines(keepends=True)
    function_source = "".join(lines[node.lineno - 1 : node.end_lineno]).encode("utf-8")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "function": PRODUCTION_FUNCTION,
        "function_source_sha256": _sha256_bytes(function_source),
    }


def _require_trace(row: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    edits = row.get("tn_edits")
    if not (
        isinstance(edits, list)
        and len(edits) == 1
        and isinstance(edits[0], Mapping)
    ):
        raise ValueError(f"{context}: expected exactly one tn_edits object")
    trace = edits[0]
    for key in ("category", "replace_from", "replace_to"):
        if not isinstance(trace.get(key), str) or not trace[key].strip():
            raise ValueError(f"{context}: trace {key} must be a non-empty string")
    span = trace.get("replace_span")
    if not (
        isinstance(span, list)
        and len(span) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
        and 0 <= span[0] < span[1]
    ):
        raise ValueError(f"{context}: trace replace_span is malformed")
    return trace


def _require_row_contract(row: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"].strip():
        raise ValueError(f"{context}: sample_id must be a non-empty string")
    for key in ("sent", "try_tn"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{context}: {key} must be a non-empty string")
    if row.get("tn_scope") != "proposal_covered_verified":
        raise ValueError(f"{context}: TN scope contract drifted")
    if row.get("proposal_covered_verified") is not True:
        raise ValueError(f"{context}: proposal_covered_verified must be true")
    if row.get("global_tn_verified") is not False:
        raise ValueError(f"{context}: global_tn_verified must remain false")
    return _require_trace(row, context=context)


def _changed_target_indices(
    from_norm: Sequence[str], to_norm: Sequence[str]
) -> list[int]:
    changed: list[int] = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(
        None, list(from_norm), list(to_norm)
    ).get_opcodes():
        if tag in {"replace", "insert"}:
            changed.extend(range(int(j1), int(j2)))
    return changed


def _audit_row(row: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    positive_tokens = _lexical_tokens(_clean_caption_phrase(row["sent"]))
    negative_tokens = _lexical_tokens(_clean_caption_phrase(row["try_tn"]))
    from_tokens = _lexical_tokens(trace["replace_from"])
    to_tokens = _lexical_tokens(trace["replace_to"])
    positive_norm = [token["norm"] for token in positive_tokens]
    negative_norm = [token["norm"] for token in negative_tokens]
    from_norm = [token["norm"] for token in from_tokens]
    to_norm = [token["norm"] for token in to_tokens]
    changed_local_indices = _changed_target_indices(from_norm, to_norm)

    decision: dict[str, Any] = {
        "status": "invalid",
        "failure_reason": None,
        "selection": None,
        "selected_span": None,
        "changed_local_indices": changed_local_indices,
        "changed_lexical_indices": [],
        "changed_target_tokens": [],
        "reconstructing_span_count": 0,
    }
    if not from_norm or not to_norm:
        decision["failure_reason"] = "empty_replace_tokens"
        return decision

    reconstructing_spans: list[tuple[int, int]] = []
    for candidate_start in range(0, len(positive_norm) - len(from_norm) + 1):
        candidate_end = candidate_start + len(from_norm)
        if positive_norm[candidate_start:candidate_end] != from_norm:
            continue
        reconstructed = (
            positive_norm[:candidate_start]
            + to_norm
            + positive_norm[candidate_end:]
        )
        if reconstructed == negative_norm:
            reconstructing_spans.append((candidate_start, candidate_end))
    decision["reconstructing_span_count"] = len(reconstructing_spans)

    declared_span = tuple(int(value) for value in trace["replace_span"])
    if declared_span in reconstructing_spans:
        selected_span = declared_span
        decision["selection"] = "declared_span"
    elif len(reconstructing_spans) == 1:
        selected_span = reconstructing_spans[0]
        decision["selection"] = "unique_reconstruction_fallback"
    else:
        decision["failure_reason"] = (
            "no_exact_reconstruction"
            if not reconstructing_spans
            else "ambiguous_reconstruction_without_declared_span"
        )
        return decision
    decision["selected_span"] = list(selected_span)

    if not changed_local_indices:
        decision["failure_reason"] = "no_changed_target_tokens"
        return decision
    start = selected_span[0]
    changed_lexical_indices = [start + index for index in changed_local_indices]
    if any(index >= len(negative_tokens) for index in changed_lexical_indices):
        decision["failure_reason"] = "changed_target_index_out_of_bounds"
        return decision

    decision.update(
        {
            "status": "valid",
            "failure_reason": None,
            "changed_lexical_indices": changed_lexical_indices,
            "changed_target_tokens": [
                negative_tokens[index]["norm"] for index in changed_lexical_indices
            ],
        }
    )
    return decision


def _normalize_expression(value: Any) -> str:
    value = _clean_caption_phrase(value)
    return value if value.endswith((".", "?")) else value + " ."


def _runtime_role_audit(
    records: Sequence[tuple[str, list[str], Mapping[str, Any], bool]],
    *,
    batch_size: int = 256,
    max_text_len: int = 256,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", local_files_only=True
    )
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    runtime_valid = 0
    lexical_valid = 0
    bert_rejected = 0
    changed_wordpieces = 0
    maximum_input_tokens = 0
    runtime_valid_digest = hashlib.sha256()
    runtime_invalid_digest = hashlib.sha256()
    with torch.no_grad():
        for offset in range(0, len(records), int(batch_size)):
            chunk = records[offset : offset + int(batch_size)]
            pairs = [record[1] for record in chunk]
            traces = [record[2] for record in chunk]
            flat = [caption for pair in pairs for caption in pair]
            tokenized = tokenizer(flat, padding="longest", return_tensors="pt")
            (
                text_self_attention_masks,
                _position_ids,
                category_masks,
            ) = generate_masks_with_special_tokens_and_transfer_map(
                tokenized, special_tokens, tokenizer
            )
            if text_self_attention_masks.shape[1] > int(max_text_len):
                for key in ("input_ids", "attention_mask", "token_type_ids"):
                    if key in tokenized:
                        tokenized[key] = tokenized[key][:, : int(max_text_len)]
            token_count = int(tokenized["input_ids"].shape[1])
            maximum_input_tokens = max(maximum_input_tokens, token_count)
            phrase_mask = torch.zeros(
                (len(flat), token_count), dtype=torch.bool
            )
            for row_index, rows in enumerate(category_masks):
                rows = rows[:, :token_count].bool()
                if rows.numel():
                    phrase_mask[row_index, : rows.shape[1]] = rows.any(dim=0)
            ids = tokenized["input_ids"].reshape(len(chunk), 2, token_count)
            phrase_mask = phrase_mask.reshape(len(chunk), 2, token_count)
            roles = build_direct_trace_token_roles(
                tokenizer,
                pairs,
                traces,
                ids,
                phrase_mask,
                max_text_len=int(max_text_len),
            )
            for record, is_runtime_valid in zip(chunk, roles["valid"].tolist()):
                sample_id, _pair, _trace, is_lexical_valid = record
                lexical_valid += int(is_lexical_valid)
                runtime_valid += int(is_runtime_valid)
                if is_lexical_valid and not is_runtime_valid:
                    bert_rejected += 1
                digest = runtime_valid_digest if is_runtime_valid else runtime_invalid_digest
                digest.update(sample_id.encode("utf-8") + b"\n")
            changed_wordpieces += int(roles["changed"].sum().item())
    return {
        "tokenizer": "bert-base-uncased",
        "max_text_len": int(max_text_len),
        "maximum_padded_input_tokens": maximum_input_tokens,
        "lexical_valid_rows": lexical_valid,
        "runtime_token_role_valid_rows": runtime_valid,
        "runtime_token_role_invalid_rows": len(records) - runtime_valid,
        "bert_alignment_or_truncation_rejections": bert_rejected,
        "changed_wordpiece_total": changed_wordpieces,
        "runtime_valid_sample_ids_sha256": runtime_valid_digest.hexdigest(),
        "runtime_invalid_sample_ids_sha256": runtime_invalid_digest.hexdigest(),
    }


def audit(source_path: Path) -> dict[str, Any]:
    source_path = source_path.resolve(strict=True)
    source_sha256 = _sha256_file(source_path)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            "sealed D3 source SHA256 drifted: "
            f"expected {SOURCE_SHA256}, observed {source_sha256}"
        )

    counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    changed_count_histogram: Counter[int] = Counter()
    reconstructing_span_histogram: Counter[int] = Counter()
    seen_sample_ids: set[str] = set()
    decision_digest = hashlib.sha256()
    valid_sample_digest = hashlib.sha256()
    invalid_sample_digest = hashlib.sha256()
    runtime_records = []

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{source_path}:{line_number}: blank rows are forbidden")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{source_path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{source_path}:{line_number}: expected a JSON object")
            context = f"{source_path}:{line_number}"
            trace = _require_row_contract(row, context=context)
            sample_id = str(row["sample_id"])
            if sample_id in seen_sample_ids:
                raise ValueError(f"{context}: duplicate sample_id {sample_id!r}")
            seen_sample_ids.add(sample_id)

            decision = _audit_row(row, trace)
            runtime_records.append(
                (
                    sample_id,
                    [
                        _normalize_expression(row["sent"]),
                        _normalize_expression(row["try_tn"]),
                    ],
                    dict(trace),
                    decision["status"] == "valid",
                )
            )
            counts["rows"] += 1
            all_changed = _changed_target_indices(
                [token["norm"] for token in _lexical_tokens(trace["replace_from"])],
                [token["norm"] for token in _lexical_tokens(trace["replace_to"])],
            )
            counts[
                "rows_with_target_side_replace_or_insert"
                if all_changed
                else "rows_without_target_side_replace_or_insert"
            ] += 1
            reconstructing_span_histogram[int(decision["reconstructing_span_count"])] += 1
            if decision["selection"] is not None:
                counts["exact_reconstruction_selected"] += 1
                counts[str(decision["selection"])] += 1

            decision_record = {
                "line_number": line_number,
                "sample_id": sample_id,
                **decision,
            }
            decision_digest.update(_canonical_bytes(decision_record) + b"\n")
            if decision["status"] == "valid":
                counts["token_role_valid_rows"] += 1
                changed_count = len(decision["changed_local_indices"])
                counts["changed_target_token_total"] += changed_count
                changed_count_histogram[changed_count] += 1
                valid_sample_digest.update(sample_id.encode("utf-8") + b"\n")
            else:
                reason = str(decision["failure_reason"])
                if reason not in FAILURE_REASONS:
                    raise AssertionError(f"unrecognized failure reason: {reason}")
                failure_counts[reason] += 1
                invalid_sample_digest.update(sample_id.encode("utf-8") + b"\n")

    if counts["rows"] != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} source rows, observed {counts['rows']}"
        )
    invalid_rows = sum(failure_counts.values())
    if counts["token_role_valid_rows"] + invalid_rows != counts["rows"]:
        raise AssertionError("valid/invalid row accounting does not close")
    if counts["declared_span"] + counts["unique_reconstruction_fallback"] != counts[
        "exact_reconstruction_selected"
    ]:
        raise AssertionError("span-selection accounting does not close")
    if counts["rows_with_target_side_replace_or_insert"] + counts[
        "rows_without_target_side_replace_or_insert"
    ] != counts["rows"]:
        raise AssertionError("target-side change accounting does not close")
    runtime = _runtime_role_audit(runtime_records)
    if runtime["lexical_valid_rows"] != counts["token_role_valid_rows"]:
        raise AssertionError("runtime audit lexical-valid accounting drifted")

    return {
        "schema": RECEIPT_SCHEMA,
        "source": {
            "path": str(source_path.relative_to(REPO_ROOT)),
            "sha256": source_sha256,
            "size_bytes": source_path.stat().st_size,
            "rows": int(counts["rows"]),
            "unique_sample_ids": len(seen_sample_ids),
        },
        "algorithm": {
            "lexical_token_pattern": TRACE_TOKEN_PATTERN,
            "normalization": "ASCII lexical tokens lowercased after dataset caption cleaning",
            "span_selection": "declared span if reconstructing, else the sole reconstructing span",
            "changed_target_opcodes": ["replace", "insert"],
            "runtime_tokenizer": runtime["tokenizer"],
            "max_text_len": runtime["max_text_len"],
            "production_binding": _production_function_record(PRODUCTION_SOURCE),
        },
        "counts": {
            "rows_with_target_side_replace_or_insert": int(
                counts["rows_with_target_side_replace_or_insert"]
            ),
            "rows_without_target_side_replace_or_insert": int(
                counts["rows_without_target_side_replace_or_insert"]
            ),
            "exact_reconstruction_selected": int(
                counts["exact_reconstruction_selected"]
            ),
            "selected_declared_span": int(counts["declared_span"]),
            "selected_unique_reconstruction_fallback": int(
                counts["unique_reconstruction_fallback"]
            ),
            "lexical_token_role_valid_rows": int(counts["token_role_valid_rows"]),
            "lexical_token_role_invalid_rows": int(invalid_rows),
            "runtime_token_role_valid_rows": int(
                runtime["runtime_token_role_valid_rows"]
            ),
            "runtime_token_role_invalid_rows": int(
                runtime["runtime_token_role_invalid_rows"]
            ),
            "bert_alignment_or_truncation_rejections": int(
                runtime["bert_alignment_or_truncation_rejections"]
            ),
            "changed_target_token_total_on_valid_rows": int(
                counts["changed_target_token_total"]
            ),
            "changed_wordpiece_total_on_runtime_valid_rows": int(
                runtime["changed_wordpiece_total"]
            ),
            "maximum_padded_input_tokens": int(
                runtime["maximum_padded_input_tokens"]
            ),
        },
        "failure_counts": {
            reason: int(failure_counts[reason]) for reason in FAILURE_REASONS
        },
        "histograms": {
            "reconstructing_span_count": {
                str(key): int(value)
                for key, value in sorted(reconstructing_span_histogram.items())
            },
            "changed_target_token_count_on_valid_rows": {
                str(key): int(value)
                for key, value in sorted(changed_count_histogram.items())
            },
        },
        "ordered_stream_sha256": {
            "role_decisions": decision_digest.hexdigest(),
            "valid_sample_ids": valid_sample_digest.hexdigest(),
            "invalid_sample_ids": invalid_sample_digest.hexdigest(),
            "runtime_valid_sample_ids": runtime[
                "runtime_valid_sample_ids_sha256"
            ],
            "runtime_invalid_sample_ids": runtime[
                "runtime_invalid_sample_ids_sha256"
            ],
        },
        "invariants": {
            "all_rows_have_one_well_formed_trace": True,
            "all_rows_preserve_proposal_covered_scope": True,
            "all_rows_preserve_global_tn_verified_false": True,
            "sample_ids_are_unique": len(seen_sample_ids) == counts["rows"],
            "valid_invalid_accounting_closes": (
                counts["token_role_valid_rows"] + invalid_rows == counts["rows"]
            ),
            "span_selection_accounting_closes": (
                counts["declared_span"]
                + counts["unique_reconstruction_fallback"]
                == counts["exact_reconstruction_selected"]
            ),
            "runtime_valid_is_a_subset_of_lexical_valid": (
                runtime["runtime_token_role_valid_rows"]
                <= runtime["lexical_valid_rows"]
            ),
        },
    }


def _render_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing receipt differs; do not rewrite it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    payload = audit(Path(args.source).expanduser())
    content = _render_json(payload)
    if args.check:
        observed = output_path.read_bytes()
        if observed != content:
            raise SystemExit(f"receipt drifted: {output_path}")
        action = "verified"
    else:
        _write_atomic(output_path, content)
        action = "wrote"
    print(
        f"{action} {output_path} "
        f"sha256={_sha256_bytes(content)} "
        f"valid={payload['counts']['runtime_token_role_valid_rows']}/"
        f"{payload['source']['rows']}"
    )


if __name__ == "__main__":
    main()
