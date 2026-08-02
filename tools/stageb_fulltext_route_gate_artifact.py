#!/usr/bin/env python3
"""Freeze a validation-only full-expression gate for a routed Stage-B score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_canonical_caption_route_artifact import (
    DEFAULT_DESCRIPTOR_ID,
    load_and_verify_caption_route_artifact,
)
from util.stageb_exact_topk_contract import canonical_sha256


SCHEMA = "stageb-fulltext-route-gate-selection-v2"
KIND = "formal_validation_frozen_fulltext_route_gate"
IDENTITY_ALGORITHM = "sha256-canonical-json-excluding-artifact-identity-v1"
VAL_SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")
EXPECTED_MANIFEST_N = {
    "refcoco_val": 10834,
    "refcocop_val": 10758,
    "refcocog_val": 4896,
}
GATED_CAPTION = "person"
GATED_DESCRIPTOR_ID = "max_patch_p025_w003125_v1"
TOKEN_COUNT_CONTRACT = {
    "schema": "ascii-lower-alnum-lexical-token-count-v1",
    "normalization": "lowercase_then_regex_[a-z0-9]+",
    "selection_source": "manifest.instances[0].raw_phrase",
    "runtime_source": "validated_full_expression_target.caption",
    "equivalence": "formal caption provenance requires cleaned raw_phrase == target.caption",
}
THRESHOLD_CANDIDATES = tuple(range(1, 65))
ITERATION_PROVENANCE = {
    "test_blind_holdout_claim": False,
    "selection_metric_inputs": list(VAL_SPLITS),
    "note": (
        "The full-text gate hypothesis and pooled-primary v2 rule were introduced "
        "after inspecting routed-v2 and t8 adaptive failures. Threshold metrics are "
        "computed only from the bound validation records, but reused Ref8 test splits "
        "are not an independent holdout."
    ),
}
SELECTION_CONTRACT = {
    "schema": "stageb-fulltext-route-gate-selection-contract-v1",
    "input_splits": list(VAL_SPLITS),
    "full_validation_only": True,
    "gated_caption": GATED_CAPTION,
    "gated_descriptor_id": GATED_DESCRIPTOR_ID,
    "fallback_descriptor_id": DEFAULT_DESCRIPTOR_ID,
    "threshold_candidates": list(THRESHOLD_CANDIDATES),
    "predicate": "full_expression_lexical_token_count <= max_tokens",
    "every_validation_split_must_strictly_improve": True,
    "primary": "maximum pooled correct gain",
    "secondary_for_exact_primary_tie": (
        "maximum worst validation-split gain rate over all caption examples"
    ),
    "tertiary_tiebreak": "smaller max_tokens",
    "runtime_routing_inputs": [
        "patch_outputs.stage_a_captions",
        "validated_full_expression_target.caption",
    ],
    "forbidden_runtime_routing_inputs": [
        "targets.labels",
        "targets.boxes",
        "manifest.category",
        "refcoco_category_id",
        "dataset",
        "split",
    ],
    "token_count_contract": dict(TOKEN_COUNT_CONTRACT),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FullTextRouteGateArtifactError(RuntimeError):
    pass


def full_expression_token_count(value: Any) -> int:
    if not isinstance(value, str) or not value.strip():
        raise FullTextRouteGateArtifactError("full expression must be a non-empty string")
    count = len(_TOKEN_RE.findall(value.lower()))
    if count <= 0:
        raise FullTextRouteGateArtifactError("full expression has no lexical tokens")
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FullTextRouteGateArtifactError(f"missing bound file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _require_file_unchanged(record: Mapping[str, Any], *, label: str) -> None:
    observed = _file_record(Path(str(record.get("path", ""))))
    expected = {
        "path": str(Path(str(record.get("path", ""))).expanduser().resolve()),
        "size_bytes": record.get("size_bytes"),
        "sha256": record.get("sha256"),
    }
    if observed != expected:
        raise FullTextRouteGateArtifactError(f"{label} changed during artifact build")


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullTextRouteGateArtifactError(f"could not parse {label}: {error}") from error


def _load_jsonl(path: Path, *, label: str, expected_n: int) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FullTextRouteGateArtifactError(
                        f"{label}:{line_number} must be a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise FullTextRouteGateArtifactError(f"could not parse {label}: {error}") from error
    if len(rows) != expected_n:
        raise FullTextRouteGateArtifactError(
            f"{label} has {len(rows)} rows, expected {expected_n}"
        )
    return rows


def _formal_route_binding(path: Path, route_artifact: Mapping[str, Any]) -> Dict[str, Any]:
    record = _file_record(path)
    payload = _load_json(path, label="formal routed-transfer artifact")
    if not isinstance(payload, dict):
        raise FullTextRouteGateArtifactError("formal routed-transfer artifact must be an object")
    if payload.get("schema") != "stageb-external-gdino-rank-transfer-formal-artifact-v2":
        raise FullTextRouteGateArtifactError("full-text gate requires the routed v2 artifact")
    identity = payload.get("artifact_identity")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_identity"}
    expected_identity = {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(unsigned),
    }
    if identity != expected_identity:
        raise FullTextRouteGateArtifactError("formal routed-transfer identity mismatch")
    routing = payload.get("inference_contract", {}).get("routing", {})
    route_identity = route_artifact.get("artifact_identity", {})
    if routing.get("artifact_identity") != route_identity:
        raise FullTextRouteGateArtifactError(
            "formal routed-transfer artifact does not bind the supplied caption route"
        )
    baseline = payload.get("components", {}).get("merged_external", {}).get(
        "base_baseline_checkpoint"
    )
    if not isinstance(baseline, dict):
        raise FullTextRouteGateArtifactError(
            "formal routed-transfer artifact lacks its baseline checkpoint"
        )
    observed_baseline = _file_record(Path(str(baseline.get("path", ""))))
    expected_baseline = {
        "path": str(Path(str(baseline.get("path", ""))).expanduser().resolve()),
        "size_bytes": baseline.get("size_bytes"),
        "sha256": baseline.get("sha256"),
    }
    if observed_baseline != expected_baseline:
        raise FullTextRouteGateArtifactError("baseline checkpoint identity drifted")
    return {
        "artifact": record,
        "artifact_identity": dict(identity),
        "baseline_checkpoint": observed_baseline,
    }


def _validate_baseline_summary(
    path: Path,
    *,
    base_record_paths: Mapping[str, Path],
    baseline_checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    record = _file_record(path)
    payload = _load_json(path, label="baseline Ref8 summary")
    rows = payload.get("refcoco") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise FullTextRouteGateArtifactError("baseline summary lacks refcoco rows")
    by_split: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("dataset") in VAL_SPLITS:
            split = str(row["dataset"])
            if split in by_split:
                raise FullTextRouteGateArtifactError(
                    f"baseline summary duplicates {split}"
                )
            by_split[split] = row
    if set(by_split) != set(VAL_SPLITS):
        raise FullTextRouteGateArtifactError(
            "baseline summary lacks exact validation split coverage"
        )
    checkpoint_path = str(
        Path(str(baseline_checkpoint["path"])).expanduser().resolve()
    )
    for split in VAL_SPLITS:
        row = by_split[split]
        expected_n = EXPECTED_MANIFEST_N[split]
        base_rows = _load_jsonl(
            base_record_paths[split],
            label=f"{split} baseline summary records",
            expected_n=expected_n,
        )
        checks = {
            "checkpoint": checkpoint_path,
            "dataset": split,
            "num_expressions": expected_n,
            "manifest_n": expected_n,
            "batch_size": 32,
            "num_workers": 4,
            "max_batches": 0,
            "invalid_records": 0,
        }
        for key, expected in checks.items():
            observed = row.get(key)
            if key == "checkpoint" and isinstance(observed, str):
                observed = str(Path(observed).expanduser().resolve())
            if observed != expected:
                raise FullTextRouteGateArtifactError(
                    f"baseline summary {split} drifted for {key}"
                )
        observed_records = str(
            Path(str(row.get("records_jsonl", ""))).expanduser().resolve()
        )
        if observed_records != str(base_record_paths[split].resolve()):
            raise FullTextRouteGateArtifactError(
                f"baseline summary {split} records path drifted"
            )
        run_ids = {item.get("run_id") for item in base_rows}
        manifest_hashes = {item.get("manifest_sha256") for item in base_rows}
        if run_ids != {row.get("run_id")}:
            raise FullTextRouteGateArtifactError(
                f"baseline summary {split} run_id drifted"
            )
        if manifest_hashes != {row.get("manifest_sha256")}:
            raise FullTextRouteGateArtifactError(
                f"baseline summary {split} manifest identity drifted"
            )
        recomputed_acc50 = sum(int(item.get("correct50") is True) for item in base_rows) / expected_n
        observed_acc50 = row.get("acc50")
        if (
            isinstance(observed_acc50, bool)
            or not isinstance(observed_acc50, (int, float))
            or not math.isclose(
                float(observed_acc50), recomputed_acc50, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise FullTextRouteGateArtifactError(
                f"baseline summary {split} acc50 drifted"
            )
    return record


def _require_record_identity(
    row: Mapping[str, Any],
    *,
    split: str,
    index: int,
    manifest_sha256: str,
    expected_n: int,
    label: str,
) -> None:
    expected = {
        "schema": "stageb-eval-record-v1",
        "task": "ref",
        "split": split,
        "manifest_index": index,
        "manifest_sha256": manifest_sha256,
        "manifest_n": expected_n,
        "valid": True,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise FullTextRouteGateArtifactError(
                f"{label} record {index} drifted for {key}"
            )
    if type(row.get("correct50")) is not bool:
        raise FullTextRouteGateArtifactError(f"{label} record {index} lacks correct50")
    best_iou = row.get("all_query_best_iou")
    if (
        isinstance(best_iou, bool)
        or not isinstance(best_iou, (int, float))
        or not math.isfinite(float(best_iou))
        or not 0.0 <= float(best_iou) <= 1.0
    ):
        raise FullTextRouteGateArtifactError(
            f"{label} record {index} lacks all_query_best_iou"
        )


def _split_evidence(
    *,
    split: str,
    base_record_path: Path,
    routed_record_path: Path,
    manifest_path: Path,
    route_artifact: Mapping[str, Any],
    formal_identity_sha256: str,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    expected_n = EXPECTED_MANIFEST_N[split]
    base_file = _file_record(base_record_path)
    routed_file = _file_record(routed_record_path)
    manifest_file = _file_record(manifest_path)
    base_rows = _load_jsonl(base_record_path, label=f"{split} base records", expected_n=expected_n)
    routed_rows = _load_jsonl(
        routed_record_path, label=f"{split} routed records", expected_n=expected_n
    )
    manifest_rows = _load_jsonl(
        manifest_path, label=f"{split} manifest", expected_n=expected_n
    )
    route_identity = route_artifact["artifact_identity"]["sha256"]
    routed_overrides = route_artifact["route"]["overrides"]
    if routed_overrides.get(GATED_CAPTION) != GATED_DESCRIPTOR_ID:
        raise FullTextRouteGateArtifactError(
            "upstream route does not map person to the required descriptor"
        )

    evidence: list[Dict[str, Any]] = []
    for index, (base, routed, manifest) in enumerate(
        zip(base_rows, routed_rows, manifest_rows)
    ):
        _require_record_identity(
            base,
            split=split,
            index=index,
            manifest_sha256=manifest_file["sha256"],
            expected_n=expected_n,
            label="base",
        )
        _require_record_identity(
            routed,
            split=split,
            index=index,
            manifest_sha256=manifest_file["sha256"],
            expected_n=expected_n,
            label="routed",
        )
        for key in ("sample_id", "image_id", "ann_id", "ref_id", "sent_id"):
            if base.get(key) != routed.get(key):
                raise FullTextRouteGateArtifactError(
                    f"{split} paired record {index} drifted for {key}"
                )
        if base.get("all_query_best_iou") != routed.get("all_query_best_iou"):
            raise FullTextRouteGateArtifactError(
                f"{split} paired record {index} changed all_query_best_iou"
            )
        if (
            routed.get("caption_route_selection_artifact_identity_sha256")
            != route_identity
            or routed.get("external_transfer_artifact_sha256")
            != formal_identity_sha256
        ):
            raise FullTextRouteGateArtifactError(
                f"{split} routed record {index} has the wrong artifact identity"
            )
        instances = manifest.get("instances") if isinstance(manifest, dict) else None
        if (
            not isinstance(instances, list)
            or len(instances) != 1
            or not isinstance(instances[0], dict)
        ):
            raise FullTextRouteGateArtifactError(
                f"{split} manifest row {index} must contain one instance"
            )
        raw_phrase = instances[0].get("raw_phrase")
        token_count = full_expression_token_count(raw_phrase)
        caption = routed.get("canonical_class_norm")
        if not isinstance(caption, str) or not caption:
            raise FullTextRouteGateArtifactError(
                f"{split} routed record {index} lacks canonical_class_norm"
            )
        descriptor_id = routed.get("caption_route_descriptor_id")
        expected_descriptor = routed_overrides.get(caption, DEFAULT_DESCRIPTOR_ID)
        if descriptor_id != expected_descriptor:
            raise FullTextRouteGateArtifactError(
                f"{split} routed record {index} contradicts the upstream route"
            )
        evidence.append(
            {
                "caption": caption,
                "token_count": token_count,
                "base_correct50": bool(base["correct50"]),
                "routed_correct50": bool(routed["correct50"]),
            }
        )

    for label, record in (
        ("base records", base_file),
        ("routed records", routed_file),
        ("manifest", manifest_file),
    ):
        _require_file_unchanged(record, label=f"{split} {label}")
    return {
        "base_records": base_file,
        "routed_records": routed_file,
        "manifest": manifest_file,
        "manifest_n": expected_n,
    }, evidence


def _threshold_stats(
    threshold: int, evidence_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    per_split: Dict[str, Any] = {}
    pooled_caption_n = 0
    pooled_delta = 0
    gain_rates: list[float] = []
    for split in VAL_SPLITS:
        people = [row for row in evidence_by_split[split] if row["caption"] == GATED_CAPTION]
        if not people:
            raise FullTextRouteGateArtifactError(f"{split} contains no person examples")
        gated = [row for row in people if int(row["token_count"]) <= threshold]
        delta = sum(
            int(row["routed_correct50"]) - int(row["base_correct50"])
            for row in gated
        )
        caption_n = len(people)
        pooled_caption_n += caption_n
        pooled_delta += delta
        gain_rates.append(delta / caption_n)
        per_split[split] = {
            "caption_n": caption_n,
            "gated_n": len(gated),
            "delta_correct": delta,
            "delta_rate_over_all_caption_examples": delta / caption_n,
            "strictly_improves": delta > 0,
        }
    return {
        "max_tokens": threshold,
        "per_split": per_split,
        "pooled_caption_n": pooled_caption_n,
        "pooled_delta_correct": pooled_delta,
        "worst_split_gain_rate": min(gain_rates),
        "eligible": all(row["strictly_improves"] for row in per_split.values()),
    }


def _select_threshold(candidates: Sequence[Mapping[str, Any]]) -> int:
    eligible = [row for row in candidates if row.get("eligible") is True]
    if not eligible:
        raise FullTextRouteGateArtifactError(
            "no full-expression threshold strictly improves every validation split"
        )
    maximum_gain = max(int(row["pooled_delta_correct"]) for row in eligible)
    maximum = [
        row
        for row in eligible
        if int(row["pooled_delta_correct"]) == maximum_gain
    ]
    maximum.sort(
        key=lambda row: (-float(row["worst_split_gain_rate"]), int(row["max_tokens"]))
    )
    return int(maximum[0]["max_tokens"])


def _route_summary(
    threshold: int, evidence_by_split: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for split in VAL_SPLITS:
        rows = evidence_by_split[split]
        base_correct = sum(int(row["base_correct50"]) for row in rows)
        routed_correct = 0
        gated_count = 0
        fallback_count = 0
        for row in rows:
            use_routed = row["caption"] != GATED_CAPTION or int(row["token_count"]) <= threshold
            if row["caption"] == GATED_CAPTION:
                gated_count += int(use_routed)
                fallback_count += int(not use_routed)
            routed_correct += int(
                row["routed_correct50"] if use_routed else row["base_correct50"]
            )
        n = len(rows)
        delta = routed_correct - base_correct
        if delta <= 0:
            raise FullTextRouteGateArtifactError(
                f"selected full-text route does not strictly improve {split}"
            )
        summary[split] = {
            "n": n,
            "base_correct": base_correct,
            "base_acc50": base_correct / n,
            "route_correct": routed_correct,
            "route_acc50": routed_correct / n,
            "delta_correct": delta,
            "delta_acc50": delta / n,
            "gated_caption_routed_count": gated_count,
            "gated_caption_fallback_count": fallback_count,
        }
    return summary


def build_fulltext_route_gate_artifact(
    *,
    base_record_paths: Mapping[str, str | Path],
    routed_record_paths: Mapping[str, str | Path],
    manifest_paths: Mapping[str, str | Path],
    baseline_summary_path: str | Path,
    caption_route_artifact_path: str | Path,
    formal_routed_artifact_path: str | Path,
) -> Dict[str, Any]:
    for label, paths in (
        ("base records", base_record_paths),
        ("routed records", routed_record_paths),
        ("manifests", manifest_paths),
    ):
        if set(paths) != set(VAL_SPLITS):
            raise FullTextRouteGateArtifactError(
                f"{label} require the exact three validation splits"
            )
    route_path = Path(caption_route_artifact_path).expanduser().resolve()
    route_record = _file_record(route_path)
    try:
        route_artifact = load_and_verify_caption_route_artifact(route_path)
    except Exception as error:
        raise FullTextRouteGateArtifactError(str(error)) from error
    formal_path = Path(formal_routed_artifact_path).expanduser().resolve()
    formal_binding = _formal_route_binding(formal_path, route_artifact)
    formal_identity = formal_binding["artifact_identity"]["sha256"]
    resolved_base_paths = {
        split: Path(base_record_paths[split]).expanduser().resolve()
        for split in VAL_SPLITS
    }
    baseline_summary = _validate_baseline_summary(
        Path(baseline_summary_path).expanduser().resolve(),
        base_record_paths=resolved_base_paths,
        baseline_checkpoint=formal_binding["baseline_checkpoint"],
    )

    selection_inputs: Dict[str, Any] = {}
    evidence_by_split: Dict[str, list[Dict[str, Any]]] = {}
    for split in VAL_SPLITS:
        selection_inputs[split], evidence_by_split[split] = _split_evidence(
            split=split,
            base_record_path=resolved_base_paths[split],
            routed_record_path=Path(routed_record_paths[split]).expanduser().resolve(),
            manifest_path=Path(manifest_paths[split]).expanduser().resolve(),
            route_artifact=route_artifact,
            formal_identity_sha256=formal_identity,
        )

    candidates = [
        _threshold_stats(value, evidence_by_split) for value in THRESHOLD_CANDIDATES
    ]
    selected_threshold = _select_threshold(candidates)
    upstream_overrides = dict(route_artifact["route"]["overrides"])
    del upstream_overrides[GATED_CAPTION]
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "kind": KIND,
        "selection_inputs": selection_inputs,
        "caption_route_artifact": route_record,
        "caption_route_artifact_identity": dict(route_artifact["artifact_identity"]),
        "formal_routed_artifact": formal_binding["artifact"],
        "formal_routed_artifact_identity": formal_binding["artifact_identity"],
        "baseline_checkpoint": formal_binding["baseline_checkpoint"],
        "baseline_summary": baseline_summary,
        "iteration_provenance": dict(ITERATION_PROVENANCE),
        "selection_contract": dict(SELECTION_CONTRACT),
        "selection_contract_sha256": canonical_sha256(SELECTION_CONTRACT),
        "threshold_candidates": {
            str(row["max_tokens"]): row for row in candidates
        },
        "route": {
            "default_descriptor_id": DEFAULT_DESCRIPTOR_ID,
            "unconditional_overrides": dict(sorted(upstream_overrides.items())),
            "conditional_overrides": {
                GATED_CAPTION: {
                    "descriptor_id": GATED_DESCRIPTOR_ID,
                    "fallback_descriptor_id": DEFAULT_DESCRIPTOR_ID,
                    "predicate": {
                        "kind": "full_expression_lexical_token_count_lte",
                        "max_tokens": selected_threshold,
                        "token_count_contract": dict(TOKEN_COUNT_CONTRACT),
                    },
                }
            },
        },
        "validation_route_summary": _route_summary(
            selected_threshold, evidence_by_split
        ),
    }
    payload["artifact_identity"] = {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(payload),
    }
    _require_file_unchanged(route_record, label="caption route artifact")
    _require_file_unchanged(
        formal_binding["artifact"], label="formal routed-transfer artifact"
    )
    _require_file_unchanged(baseline_summary, label="baseline Ref8 summary")
    return payload


def create_fulltext_route_gate_artifact(
    output: str | Path,
    **kwargs: Any,
) -> Dict[str, Any]:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FullTextRouteGateArtifactError(f"refusing to overwrite artifact: {output_path}")
    payload = build_fulltext_route_gate_artifact(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
        verified = load_and_verify_fulltext_route_gate_artifact(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    if verified != payload:
        output_path.unlink(missing_ok=True)
        raise FullTextRouteGateArtifactError("serialized artifact failed exact verification")
    return payload


def load_and_verify_fulltext_route_gate_artifact(path: str | Path) -> Dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    artifact_record = _file_record(artifact_path)
    payload = _load_json(artifact_path, label="full-text route gate artifact")
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("kind") != KIND:
        raise FullTextRouteGateArtifactError("unsupported full-text route gate artifact")
    identity = payload.get("artifact_identity")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_identity"}
    if identity != {
        "algorithm": IDENTITY_ALGORITHM,
        "sha256": canonical_sha256(unsigned),
    }:
        raise FullTextRouteGateArtifactError("full-text route gate identity mismatch")
    inputs = payload.get("selection_inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(VAL_SPLITS):
        raise FullTextRouteGateArtifactError("full-text route gate inputs drifted")
    rebuilt = build_fulltext_route_gate_artifact(
        base_record_paths={
            split: inputs[split]["base_records"]["path"] for split in VAL_SPLITS
        },
        routed_record_paths={
            split: inputs[split]["routed_records"]["path"] for split in VAL_SPLITS
        },
        manifest_paths={split: inputs[split]["manifest"]["path"] for split in VAL_SPLITS},
        baseline_summary_path=payload["baseline_summary"]["path"],
        caption_route_artifact_path=payload["caption_route_artifact"]["path"],
        formal_routed_artifact_path=payload["formal_routed_artifact"]["path"],
    )
    if rebuilt != payload:
        raise FullTextRouteGateArtifactError(
            "full-text route gate no longer matches its bound validation evidence"
        )
    _require_file_unchanged(artifact_record, label="full-text route gate artifact")
    return payload


def _parse_split_paths(values: Iterable[str], *, label: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise FullTextRouteGateArtifactError(f"{label} must use SPLIT=PATH")
        split, raw_path = value.split("=", 1)
        if split not in VAL_SPLITS or split in parsed or not raw_path:
            raise FullTextRouteGateArtifactError(f"invalid or duplicate {label}: {split!r}")
        parsed[split] = Path(raw_path).expanduser().resolve()
    if set(parsed) != set(VAL_SPLITS):
        raise FullTextRouteGateArtifactError(
            f"{label} require the exact three validation splits"
        )
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or verify a validation-only full-expression route gate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--base-record", action="append", required=True)
    create.add_argument("--routed-record", action="append", required=True)
    create.add_argument("--manifest", action="append", required=True)
    create.add_argument("--baseline-summary", required=True)
    create.add_argument("--caption-route-artifact", required=True)
    create.add_argument("--formal-routed-artifact", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            payload = create_fulltext_route_gate_artifact(
                args.output,
                base_record_paths=_parse_split_paths(
                    args.base_record, label="base records"
                ),
                routed_record_paths=_parse_split_paths(
                    args.routed_record, label="routed records"
                ),
                manifest_paths=_parse_split_paths(args.manifest, label="manifests"),
                baseline_summary_path=args.baseline_summary,
                caption_route_artifact_path=args.caption_route_artifact,
                formal_routed_artifact_path=args.formal_routed_artifact,
            )
        else:
            payload = load_and_verify_fulltext_route_gate_artifact(args.artifact)
    except FullTextRouteGateArtifactError as error:
        print(f"[FAIL] {error}")
        return 2
    print(json.dumps(payload["artifact_identity"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
