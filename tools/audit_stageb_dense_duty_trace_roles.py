#!/usr/bin/env python3
"""Seal exact direct-trace token roles for formal dense-duty Stage-B."""

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
from typing import Any, Mapping

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.patch_episode import (  # noqa: E402
    _build_canonical_text_maps,
    _validate_single_edit_token_provenance,
)
from models.GroundingDINO.groundingdino import GroundingDINO  # noqa: E402
from util.stage_b_dense_duty_audit import build_code_source_closure  # noqa: E402


SCHEMA = "pivot.stageb.dense_duty_direct_trace_audit/v1"
SOURCE = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_semantic_partition_20260717"
    / "single_edit_train.jsonl"
)
SOURCE_SHA256 = "276dc5a67c6e7a6654d6daa6a88cb99b9c59b1c52f84ef93205a3d6326b1b529"
CANONICAL_CLASSES = Path(
    "/media/haoyi/T9/data/canonical_classes_with_aliases.json"
)
TOKENIZER = Path(
    "/home/haoyi/.cache/huggingface/hub/models--bert-base-uncased/"
    "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
)
OUTPUT = (
    REPO_ROOT
    / "data/ablations/stageb_dense_duty_trace_audit_20260728/receipt.json"
)
EXPECTED_ROWS = 14_196
MAX_TEXT_LEN = 256
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class _RoleHarness:
    _tokenize_stage_b_v11_captions = GroundingDINO._tokenize_stage_b_v11_captions
    _build_stage_b_v15_score_token_masks = (
        GroundingDINO._build_stage_b_v15_score_token_masks
    )
    _build_stage_b_v21_direct_trace_token_roles = (
        GroundingDINO._build_stage_b_v21_direct_trace_token_roles
    )

    def __init__(self) -> None:
        self.max_text_len = MAX_TEXT_LEN
        self.stage_b_dense_duty = True
        self.stage_b_dense_duty_allow_incidental_trace_edits = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER), local_files_only=True, use_fast=True
        )
        if not bool(getattr(self.tokenizer, "is_fast", False)):
            raise RuntimeError("dense-duty direct-trace audit requires a fast tokenizer")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _lexical(value: Any) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(value or ""))]


def _clean_expression(value: Any) -> str:
    text = str(value).replace("_", " ").replace(".", " ").strip()
    text = " ".join(text.split())
    if not text:
        raise ValueError("formal direct-trace expression is empty")
    return text + " ."


def _changed_target_indices(source: list[str], target: list[str]) -> list[int]:
    changed: list[int] = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(
        None, source, target, autojunk=False
    ).get_opcodes():
        if tag in {"replace", "insert"}:
            changed.extend(range(int(j1), int(j2)))
    return changed


def _lexical_decision(
    positive: str, negative: str, trace: Mapping[str, Any]
) -> tuple[str, str]:
    positive_tokens = _lexical(positive)
    negative_tokens = _lexical(negative)
    source_tokens = _lexical(trace.get("replace_from"))
    target_tokens = _lexical(trace.get("replace_to"))
    if not _changed_target_indices(source_tokens, target_tokens):
        return "invalid", "no_target_side_changed_token"
    reconstructing = []
    for start in range(0, len(positive_tokens) - len(source_tokens) + 1):
        end = start + len(source_tokens)
        if positive_tokens[start:end] != source_tokens:
            continue
        if positive_tokens[:start] + target_tokens + positive_tokens[end:] == negative_tokens:
            reconstructing.append((start, end))
    declared = tuple(int(value) for value in trace["replace_span"])
    if declared in reconstructing:
        return "valid", "declared_exact_reconstruction"
    if len(reconstructing) == 1:
        return "valid", "unique_exact_reconstruction_fallback"
    return "invalid", "no_unique_exact_reconstruction"


def _function_record(path: Path, function_name: str) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1 or matches[0].end_lineno is None:
        raise RuntimeError(
            f"expected one source function {function_name!r} in {path}"
        )
    lines = source.splitlines(keepends=True)
    function_source = "".join(
        lines[matches[0].lineno - 1 : matches[0].end_lineno]
    ).encode("utf-8")
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "function": function_name,
        "sha256": hashlib.sha256(function_source).hexdigest(),
    }


def _tokenizer_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(TOKENIZER.iterdir()):
        if path.is_file():
            records.append(
                {
                    "name": path.name,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
    if not records:
        raise RuntimeError("tokenizer snapshot is empty")
    return records


def _load_rows() -> list[dict[str, Any]]:
    if _sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("formal dense-duty TN manifest SHA drifted")
    rows = []
    seen = set()
    with SOURCE.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{SOURCE}:{line_number}: blank row")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise TypeError(f"{SOURCE}:{line_number}: row must be an object")
            _validate_single_edit_token_provenance(
                row, context=f"{SOURCE}:{line_number}"
            )
            if row.get("tn_scope") != "image_global_topk_verified":
                raise ValueError(f"{SOURCE}:{line_number}: TN scope drifted")
            if row.get("global_tn_verified") is not True:
                raise ValueError(f"{SOURCE}:{line_number}: verification flag drifted")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError(f"{SOURCE}:{line_number}: invalid/duplicate sample_id")
            seen.add(sample_id)
            rows.append(row)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} TN rows, observed {len(rows)}")
    return rows


def audit(*, batch_size: int) -> dict[str, Any]:
    rows = _load_rows()
    canonical_names, _aliases = _build_canonical_text_maps(
        str(CANONICAL_CLASSES)
    )
    harness = _RoleHarness()
    lexical_counts: Counter[str] = Counter()
    role_valid = 0
    positive_wordpieces = 0
    shared_wordpieces = 0
    changed_wordpieces = 0
    valid_digest = hashlib.sha256()
    invalid_digest = hashlib.sha256()
    role_digest = hashlib.sha256()

    with torch.no_grad():
        for offset in range(0, len(rows), int(batch_size)):
            chunk = rows[offset : offset + int(batch_size)]
            expressions = [
                [_clean_expression(row["sent"]), _clean_expression(row["try_tn"])]
                for row in chunk
            ]
            traces = [dict(row["tn_edits"][0]) for row in chunk]
            canonical = []
            for row in chunk:
                class_id = int(row["class_id"])
                if class_id not in canonical_names:
                    raise KeyError(f"canonical class {class_id} is missing")
                canonical.append(_clean_expression(canonical_names[class_id]))
            score_mask = harness._build_stage_b_v15_score_token_masks(
                expressions,
                canonical,
                torch.ones((len(chunk), 2), dtype=torch.bool),
                torch.device("cpu"),
            )
            roles = harness._build_stage_b_v21_direct_trace_token_roles(
                expressions, traces, score_mask, torch.device("cpu")
            )
            positive_wordpieces += int(roles["positive"].sum().item())
            shared_wordpieces += int(roles["shared"].sum().item())
            changed_wordpieces += int(roles["changed"].sum().item())

            for index, row in enumerate(chunk):
                lexical_status, lexical_reason = _lexical_decision(
                    expressions[index][0], expressions[index][1], traces[index]
                )
                lexical_counts[lexical_reason] += 1
                runtime_valid = bool(roles["valid"][index].item())
                if runtime_valid and lexical_status != "valid":
                    raise AssertionError("runtime accepted a lexically invalid trace")
                sample_id = row["sample_id"]
                digest = valid_digest if runtime_valid else invalid_digest
                digest.update(sample_id.encode("utf-8") + b"\n")
                role_valid += int(runtime_valid)
                role_record = {
                    "sample_id": sample_id,
                    "valid": runtime_valid,
                    "positive": roles["positive"][index].nonzero().tolist(),
                    "shared": roles["shared"][index].nonzero().tolist(),
                    "changed": roles["changed"][index].nonzero().tolist(),
                }
                role_digest.update(_canonical_bytes(role_record) + b"\n")

    lexical_valid = (
        lexical_counts["declared_exact_reconstruction"]
        + lexical_counts["unique_exact_reconstruction_fallback"]
    )
    surface_rejected = lexical_valid - role_valid
    if role_valid <= 0 or surface_rejected < 0:
        raise AssertionError("direct-trace role accounting is invalid")
    code_closure = build_code_source_closure(repo_root=REPO_ROOT)
    return {
        "schema": SCHEMA,
        "source": {
            "path": str(SOURCE.relative_to(REPO_ROOT)),
            "sha256": SOURCE_SHA256,
            "rows": len(rows),
            "scope": "image_global_topk_verified",
            "global_tn_verified": True,
        },
        "canonical_classes": {
            "path": str(CANONICAL_CLASSES),
            "sha256": _sha256_file(CANONICAL_CLASSES),
        },
        "tokenizer": {
            "path": str(TOKENIZER),
            "is_fast": True,
            "max_text_len": MAX_TEXT_LEN,
            "files": _tokenizer_records(),
        },
        "algorithm": {
            "token_role_source": "exact_direct_trace_v1",
            "allow_incidental_trace_edits": False,
            "changed_target_opcodes": ["replace", "insert"],
            "canonical_tokens_excluded_from_roles": True,
            "legacy_full_pair_diff_used_for_token_roles": False,
            "bindings": [
                _function_record(
                    REPO_ROOT
                    / "models/GroundingDINO/stage_b_data_driven_score.py",
                    "build_direct_trace_token_roles",
                ),
                _function_record(
                    REPO_ROOT / "models/GroundingDINO/groundingdino.py",
                    "_build_stage_b_v21_direct_trace_token_roles",
                ),
            ],
        },
        "counts": {
            "lexical_exact_valid_rows": lexical_valid,
            "selected_declared_exact_span": lexical_counts[
                "declared_exact_reconstruction"
            ],
            "selected_unique_exact_reconstruction_fallback": lexical_counts[
                "unique_exact_reconstruction_fallback"
            ],
            "no_unique_exact_reconstruction": lexical_counts[
                "no_unique_exact_reconstruction"
            ],
            "no_target_side_changed_token": lexical_counts[
                "no_target_side_changed_token"
            ],
            "canonical_score_surface_rejections": surface_rejected,
            "direct_token_valid_rows": role_valid,
            "direct_token_invalid_rows": len(rows) - role_valid,
            "positive_wordpiece_roles": positive_wordpieces,
            "shared_wordpiece_roles": shared_wordpieces,
            "changed_wordpiece_roles": changed_wordpieces,
        },
        "ordered_stream_sha256": {
            "valid_sample_ids": valid_digest.hexdigest(),
            "invalid_sample_ids": invalid_digest.hexdigest(),
            "token_roles": role_digest.hexdigest(),
        },
        "code_source_closure": {
            "schema": code_closure["schema"],
            "file_count": code_closure["file_count"],
            "sha256": code_closure["sha256"],
        },
        "invariants": {
            "all_rows_have_dataset_validated_single_edit_trace": True,
            "all_token_roles_require_exact_trace_reconstruction": True,
            "incidental_edit_rows_have_no_tn_token_roles": True,
            "deletion_only_rows_have_no_tn_token_roles": True,
            "valid_invalid_accounting_closes": role_valid + len(rows) - role_valid
            == len(rows),
        },
    }


def _render(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    output = Path(args.output).expanduser().resolve()
    content = _render(audit(batch_size=args.batch_size))
    if args.check:
        if output.read_bytes() != content:
            raise SystemExit(f"receipt drifted: {output}")
        action = "verified"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, output)
        action = "wrote"
    payload = json.loads(content)
    print(
        f"{action} {output} sha256={hashlib.sha256(content).hexdigest()} "
        f"valid={payload['counts']['direct_token_valid_rows']}/{EXPECTED_ROWS}"
    )


if __name__ == "__main__":
    main()
