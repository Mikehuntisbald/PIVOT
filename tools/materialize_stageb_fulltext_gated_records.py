#!/usr/bin/env python3
"""Materialize a frozen full-text route from paired formal source records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_fulltext_route_gate_artifact import (  # noqa: E402
    DEFAULT_DESCRIPTOR_ID,
    GATED_CAPTION,
    FullTextRouteGateArtifactError,
    full_expression_token_count,
    load_and_verify_fulltext_route_gate_artifact,
)


SCHEMA = "stageb-fulltext-gated-record-materialization-v1"
REF_SPLITS = (
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_val",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_val",
    "refcocog_test",
)
EXPECTED_MANIFEST_N = {
    "refcoco_val": 10834,
    "refcoco_testA": 5657,
    "refcoco_testB": 5095,
    "refcocop_val": 10758,
    "refcocop_testA": 5726,
    "refcocop_testB": 4889,
    "refcocog_val": 4896,
    "refcocog_test": 9602,
}


class FullTextGatedRecordError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FullTextGatedRecordError(f"missing input file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_jsonl(path: Path, *, label: str, expected_n: int) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FullTextGatedRecordError(
                        f"{label}:{line_number} must be a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise FullTextGatedRecordError(f"could not parse {label}: {error}") from error
    if len(rows) != expected_n:
        raise FullTextGatedRecordError(
            f"{label} has {len(rows)} rows, expected {expected_n}"
        )
    return rows


def _parse_split_paths(values: Iterable[str], *, label: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise FullTextGatedRecordError(f"{label} must use SPLIT=PATH")
        split, raw_path = value.split("=", 1)
        if split not in REF_SPLITS or split in parsed or not raw_path:
            raise FullTextGatedRecordError(f"invalid or duplicate {label}: {split!r}")
        parsed[split] = Path(raw_path).expanduser().resolve()
    if set(parsed) != set(REF_SPLITS):
        raise FullTextGatedRecordError(f"{label} require the exact Ref8 splits")
    return parsed


def _require_identity(
    row: Mapping[str, Any],
    *,
    split: str,
    index: int,
    manifest_sha256: str,
    expected_n: int,
    label: str,
) -> None:
    checks = {
        "schema": "stageb-eval-record-v1",
        "task": "ref",
        "split": split,
        "manifest_index": index,
        "manifest_sha256": manifest_sha256,
        "manifest_n": expected_n,
        "valid": True,
    }
    for key, expected in checks.items():
        if row.get(key) != expected:
            raise FullTextGatedRecordError(
                f"{split} {label} record {index} drifted for {key}"
            )
    if type(row.get("correct50")) is not bool:
        raise FullTextGatedRecordError(
            f"{split} {label} record {index} lacks correct50"
        )
    top1 = row.get("top1_iou")
    best = row.get("all_query_best_iou")
    for key, value in (("top1_iou", top1), ("all_query_best_iou", best)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise FullTextGatedRecordError(
                f"{split} {label} record {index} lacks finite {key}"
            )


def _materialized_record(
    *,
    base: Mapping[str, Any],
    routed: Mapping[str, Any],
    raw_phrase: str,
    token_count: int,
    max_tokens: int,
    gate_payload: Mapping[str, Any],
    gate_file: Mapping[str, Any],
    run_id: str,
) -> Dict[str, Any]:
    caption = str(routed["canonical_class_norm"])
    predicate_matched = caption == GATED_CAPTION and token_count <= max_tokens
    fallback_matched = caption == GATED_CAPTION and not predicate_matched
    source = base if fallback_matched else routed
    record = dict(routed)
    for key in (
        "top1_iou",
        "correct50",
        "valid",
        "all_query_best_iou",
    ):
        record[key] = source[key]
    record["correct25"] = bool(float(source["top1_iou"]) >= 0.25)
    if fallback_matched:
        for key in (
            "selected_box",
            "selected_box_format",
            "winner_candidate_index",
            "winner_patch_query_index",
        ):
            record.pop(key, None)
        record["matched_external_query_index"] = None
    selected_descriptor = (
        DEFAULT_DESCRIPTOR_ID
        if fallback_matched
        else str(routed["caption_route_descriptor_id"])
    )
    record.update(
        {
            "run_id": run_id,
            "fulltext_route_gate_contract_version": 1,
            "fulltext_route_gate_artifact_path": gate_file["path"],
            "fulltext_route_gate_artifact_file_sha256": gate_file["sha256"],
            "fulltext_route_gate_artifact_size_bytes": gate_file["size_bytes"],
            "fulltext_route_gate_artifact_identity_sha256": gate_payload[
                "artifact_identity"
            ]["sha256"],
            "fulltext_route_gate_raw_expression": raw_phrase,
            "fulltext_route_gate_token_count": token_count,
            "fulltext_route_gate_max_tokens": max_tokens,
            "fulltext_route_gate_predicate_matched": predicate_matched,
            "fulltext_route_gate_fallback_matched": fallback_matched,
            "fulltext_route_gate_source": (
                "external_base_direct" if fallback_matched else "formal_routed_v2"
            ),
            "fulltext_route_gate_selected_descriptor_id": selected_descriptor,
        }
    )
    return record


def materialize_fulltext_gated_records(
    *,
    gate_artifact_path: str | Path,
    base_record_paths: Mapping[str, str | Path],
    routed_record_paths: Mapping[str, str | Path],
    manifest_paths: Mapping[str, str | Path],
    output_dir: str | Path,
) -> Dict[str, Any]:
    for label, paths in (
        ("base records", base_record_paths),
        ("routed records", routed_record_paths),
        ("manifests", manifest_paths),
    ):
        if set(paths) != set(REF_SPLITS):
            raise FullTextGatedRecordError(f"{label} require the exact Ref8 splits")
    gate_path = Path(gate_artifact_path).expanduser().resolve()
    gate_file = _file_record(gate_path)
    try:
        gate = load_and_verify_fulltext_route_gate_artifact(gate_path)
    except FullTextRouteGateArtifactError as error:
        raise FullTextGatedRecordError(str(error)) from error
    conditional = gate["route"]["conditional_overrides"][GATED_CAPTION]
    max_tokens = int(conditional["predicate"]["max_tokens"])
    route_identity = gate["caption_route_artifact_identity"]["sha256"]
    formal_identity = gate["formal_routed_artifact_identity"]["sha256"]
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FullTextGatedRecordError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    records_dir = output / "per_example_records"
    records_dir.mkdir()
    run_id = f"fulltext_route_projection={gate['artifact_identity']['sha256'][:16]}"
    summaries: Dict[str, Any] = {}
    created: list[Path] = []
    try:
        for split in REF_SPLITS:
            expected_n = EXPECTED_MANIFEST_N[split]
            base_path = Path(base_record_paths[split]).expanduser().resolve()
            routed_path = Path(routed_record_paths[split]).expanduser().resolve()
            manifest_path = Path(manifest_paths[split]).expanduser().resolve()
            base_file = _file_record(base_path)
            routed_file = _file_record(routed_path)
            manifest_file = _file_record(manifest_path)
            base_rows = _load_jsonl(
                base_path, label=f"{split} base records", expected_n=expected_n
            )
            routed_rows = _load_jsonl(
                routed_path, label=f"{split} routed records", expected_n=expected_n
            )
            manifest_rows = _load_jsonl(
                manifest_path, label=f"{split} manifest", expected_n=expected_n
            )
            destination = records_dir / f"{run_id}__{split}.records.jsonl"
            correct50 = 0
            correct25 = 0
            iou_sum = 0.0
            routed_count = 0
            fallback_count = 0
            with destination.open("x", encoding="utf-8") as handle:
                created.append(destination)
                for index, (base, routed, manifest) in enumerate(
                    zip(base_rows, routed_rows, manifest_rows)
                ):
                    _require_identity(
                        base,
                        split=split,
                        index=index,
                        manifest_sha256=manifest_file["sha256"],
                        expected_n=expected_n,
                        label="base",
                    )
                    _require_identity(
                        routed,
                        split=split,
                        index=index,
                        manifest_sha256=manifest_file["sha256"],
                        expected_n=expected_n,
                        label="routed",
                    )
                    for key in ("sample_id", "image_id", "ann_id", "ref_id", "sent_id"):
                        if base.get(key) != routed.get(key):
                            raise FullTextGatedRecordError(
                                f"{split} paired record {index} drifted for {key}"
                            )
                    if base["all_query_best_iou"] != routed["all_query_best_iou"]:
                        raise FullTextGatedRecordError(
                            f"{split} paired record {index} changed all_query_best_iou"
                        )
                    if (
                        routed.get("caption_route_selection_artifact_identity_sha256")
                        != route_identity
                        or routed.get("external_transfer_artifact_sha256")
                        != formal_identity
                    ):
                        raise FullTextGatedRecordError(
                            f"{split} routed record {index} artifact identity drifted"
                        )
                    instances = manifest.get("instances") if isinstance(manifest, dict) else None
                    if (
                        not isinstance(instances, list)
                        or len(instances) != 1
                        or not isinstance(instances[0], dict)
                    ):
                        raise FullTextGatedRecordError(
                            f"{split} manifest row {index} must contain one instance"
                        )
                    raw_phrase = instances[0].get("raw_phrase")
                    token_count = full_expression_token_count(raw_phrase)
                    record = _materialized_record(
                        base=base,
                        routed=routed,
                        raw_phrase=raw_phrase,
                        token_count=token_count,
                        max_tokens=max_tokens,
                        gate_payload=gate,
                        gate_file=gate_file,
                        run_id=run_id,
                    )
                    handle.write(
                        json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
                    )
                    top1 = float(record["top1_iou"])
                    correct50 += int(record["correct50"])
                    correct25 += int(record["correct25"])
                    iou_sum += top1
                    routed_count += int(record["fulltext_route_gate_predicate_matched"])
                    fallback_count += int(record["fulltext_route_gate_fallback_matched"])
            destination_file = _file_record(destination)
            with destination.open("r", encoding="utf-8") as readback:
                observed_lines = sum(1 for _ in readback)
            if observed_lines != expected_n:
                raise FullTextGatedRecordError(f"{split} materialized record count drifted")
            summaries[split] = {
                "n": expected_n,
                "correct50": correct50,
                "acc50": correct50 / expected_n,
                "correct25": correct25,
                "acc25": correct25 / expected_n,
                "mean_iou_top1": iou_sum / expected_n,
                "gated_caption_routed_count": routed_count,
                "gated_caption_fallback_count": fallback_count,
                "records": destination_file,
                "source_base_records": base_file,
                "source_routed_records": routed_file,
                "manifest": manifest_file,
            }
        summary: Dict[str, Any] = {
            "schema": SCHEMA,
            "run_id": run_id,
            "gate_artifact": gate_file,
            "gate_artifact_identity": dict(gate["artifact_identity"]),
            "route": dict(gate["route"]),
            "splits": summaries,
        }
        summary_path = output / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        created.append(summary_path)
        return summary
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        try:
            records_dir.rmdir()
            output.rmdir()
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize Ref8 records from one frozen full-text route gate"
    )
    parser.add_argument("--gate-artifact", required=True)
    parser.add_argument("--base-record", action="append", required=True)
    parser.add_argument("--routed-record", action="append", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        summary = materialize_fulltext_gated_records(
            gate_artifact_path=args.gate_artifact,
            base_record_paths=_parse_split_paths(
                args.base_record, label="base records"
            ),
            routed_record_paths=_parse_split_paths(
                args.routed_record, label="routed records"
            ),
            manifest_paths=_parse_split_paths(args.manifest, label="manifests"),
            output_dir=args.output_dir,
        )
    except (FullTextGatedRecordError, FullTextRouteGateArtifactError) as error:
        print(f"[FAIL] {error}")
        return 2
    print(json.dumps({key: row["acc50"] for key, row in summary["splits"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
