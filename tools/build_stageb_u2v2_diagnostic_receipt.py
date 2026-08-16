#!/usr/bin/env python3
"""Bind the U2-v2 diagnostic selection, Ref8, and C100-parity result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ReceiptError(RuntimeError):
    pass


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{path} is not a JSON object")
    return value


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _micro(rows) -> float:
    total = sum(int(row["num_expressions"]) for row in rows)
    return sum(float(row["acc50"]) * int(row["num_expressions"]) for row in rows) / total


def _records(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = summary.get("tn")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ReceiptError("strict summary must contain exactly one TN row")
    path = Path(str(rows[0].get("records_jsonl", "")))
    if not path.is_absolute():
        path = Path.cwd() / path
    result = {}
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in result:
                raise ReceiptError(f"invalid record universe in {path}")
            result[sample_id] = row
    return result


def _confidence_parity(candidate, reference) -> dict[str, Any]:
    candidate_records = _records(candidate)
    reference_records = _records(reference)
    fields = ("pos_score", "neg_score", "pos_iou", "neg_iou")
    same_universe = set(candidate_records) == set(reference_records)
    field_equal = {
        field: same_universe and all(
            candidate_records[key].get(field) == reference_records[key].get(field)
            for key in candidate_records
        )
        for field in fields
    }
    return {
        "rows": len(candidate_records),
        "same_sample_universe": same_universe,
        "field_bitwise_equal": field_equal,
        "all_equal": same_universe and all(field_equal.values()),
    }


def build(args) -> dict[str, Any]:
    final2031 = _load(Path(args.final_2031))
    final1607 = _load(Path(args.final_1607))
    b58 = _load(Path(args.b58_ref8))
    c100_2031 = _load(Path(args.c100_2031))
    c100_1607 = _load(Path(args.c100_1607))
    legacy_rows = []
    for path in (args.legacy_val3, args.legacy_test5):
        legacy_rows.extend(_load(Path(path))["refcoco"])
    final_rows = final2031.get("refcoco")
    baseline_rows = b58.get("refcoco")
    if not isinstance(final_rows, list) or len(final_rows) != 8:
        raise ReceiptError("final summary must contain Ref8")
    baseline_by_split = {row["dataset"]: row for row in baseline_rows}
    ref_rows = []
    for row in final_rows:
        split = row["dataset"]
        causal = row.get("stage_b_u2v2_causal_ref_routes")
        if not isinstance(causal, Mapping) or set(causal) != {
            "b58_base", "raw_r100", "patch_r100", "patch_residual"
        }:
            raise ReceiptError(f"{split} lacks the four causal routes")
        ref_rows.append(
            {
                "split": split,
                "b58": float(baseline_by_split[split]["acc50"]),
                "u2v2": float(row["acc50"]),
                "strict_b58_win": float(row["acc50"]) > float(baseline_by_split[split]["acc50"]),
                "causal_acc50": {
                    name: float(value["acc50"]) for name, value in causal.items()
                },
            }
        )
    final_micro = _micro(final_rows)
    b58_micro = _micro(baseline_rows)
    legacy_micro = _micro(legacy_rows)
    parity1607 = _confidence_parity(final1607, c100_1607)
    parity2031 = _confidence_parity(final2031, c100_2031)
    fpr1607 = float(final1607["tn"][0]["fpr95tpr"])
    fpr2031 = float(final2031["tn"][0]["fpr95tpr"])
    c100_fpr1607 = float(c100_1607["tn"][0]["fpr95tpr"])
    c100_fpr2031 = float(c100_2031["tn"][0]["fpr95tpr"])
    all_ref_win = all(row["strict_b58_win"] for row in ref_rows)
    legacy_promotion = final_micro > legacy_micro
    confidence_parity = parity1607["all_equal"] and parity2031["all_equal"]
    formal_authorized = all_ref_win and legacy_promotion and confidence_parity
    return {
        "schema": "pivot.stageb.u2v2_diagnostic_final/v1",
        "status": "passed" if formal_authorized else "diagnostic_rejected",
        "rejection_reasons": [] if formal_authorized else [
            reason for reason, failed in (
                ("ref8_not_8_of_8_strict_b58", not all_ref_win),
                ("ref8_micro_not_above_legacy_u2", not legacy_promotion),
                ("confidence_not_bitwise_c100", not confidence_parity),
            ) if failed
        ],
        "formal_clean_training_authorized": formal_authorized,
        "checkpoint": _file(Path(args.checkpoint)),
        "inputs": {
            name: _file(Path(value)) for name, value in (
                ("selection", args.selection),
                ("final_2031", args.final_2031),
                ("final_1607", args.final_1607),
                ("b58_ref8", args.b58_ref8),
                ("legacy_val3", args.legacy_val3),
                ("legacy_test5", args.legacy_test5),
                ("c100_2031", args.c100_2031),
                ("c100_1607", args.c100_1607),
            )
        },
        "ref8": {
            "splits": ref_rows,
            "strict_b58_wins": sum(row["strict_b58_win"] for row in ref_rows),
            "micro": final_micro,
            "b58_micro": b58_micro,
            "delta_vs_b58": final_micro - b58_micro,
            "legacy_u2_micro": legacy_micro,
            "delta_vs_legacy_u2": final_micro - legacy_micro,
        },
        "confidence": {
            "strict1607": {"fpr95": fpr1607, "c100_fpr95": c100_fpr1607, "parity": parity1607},
            "strict2031": {"fpr95": fpr2031, "c100_fpr95": c100_fpr2031, "parity": parity2031},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "selection", "final_2031", "final_1607", "b58_ref8", "legacy_val3", "legacy_test5", "c100_2031", "c100_1607", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise ReceiptError(f"refusing to overwrite {output}")
    result = build(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ReceiptError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
