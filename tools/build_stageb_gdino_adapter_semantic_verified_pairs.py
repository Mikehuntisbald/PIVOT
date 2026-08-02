#!/usr/bin/env python3
"""Build audited semantic-TN pairs for the pure-GDINO confidence adapter.

The source rows verify the target and every cached top-k proposal with a VLM.
They do not label all 900 GDINO decoder queries.  The emitted scope therefore
uses the existing ``image_global_topk_verified`` contract and keeps that limit
explicit in both the row schema and the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from tools.build_stageb_vlm_strict_tn_manifest import (
    _build_g_maps,
    _build_plus_map,
    normalize_sentence,
    proposal_coverage,
    sha256_file,
)
from util.path_compat import remap_legacy_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/home/user/datasets/pivot_data"))
SOURCE_DIR = (
    REPO_ROOT / "data/ablations/stageb_v15_global_verified_train_20260711"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_semantic_verified_20260711"
)
SCHEMA = "stage-b-gdino-adapter-semantic-verified-pairs-v1"
PAIR_SCHEMA = "stage-b-gdino-adapter-semantic-verified-pair-v1"
TN_SCOPE = "image_global_topk_verified"
SOURCE_SCHEMA = "stageb_v15_global_verified_train_v1"
EXPECTED_ROWS = {"refcocoplus": 10_855, "refcocog": 6_974}
_WS = re.compile(r"\s+")


class SemanticPairError(RuntimeError):
    pass


def _clean_text(value: Any) -> str:
    return _WS.sub(
        " ", str(value or "").replace("_", " ").replace(".", " ").strip()
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticPairError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SemanticPairError(f"expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if not path.is_file():
        raise SemanticPairError(f"missing JSONL: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SemanticPairError(f"blank row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SemanticPairError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise SemanticPairError(f"non-object row at {path}:{line_number}")
            yield line_number, row


def _valid_xywh(value: Any, *, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SemanticPairError(f"invalid target bbox at {context}: {value!r}")
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise SemanticPairError(f"non-numeric target bbox at {context}") from error
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise SemanticPairError(f"non-positive target bbox at {context}: {box!r}")
    return box


def _identity(row: Mapping[str, Any], *, dataset: str, context: str) -> tuple:
    values = []
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        try:
            values.append(int(row[key]))
        except (KeyError, TypeError, ValueError) as error:
            raise SemanticPairError(f"invalid {key} at {context}") from error
    return (dataset, *values)


def _validate_source_audit(
    audit_path: Path, source_paths: Mapping[str, Path]
) -> Dict[str, Dict[str, Any]]:
    audit = _read_json(audit_path)
    if audit.get("schema") != SOURCE_SCHEMA:
        raise SemanticPairError(
            f"unexpected source audit schema: {audit.get('schema')!r}"
        )
    files = audit.get("files")
    if not isinstance(files, list) or len(files) != len(source_paths):
        raise SemanticPairError("source audit must describe exactly both frozen files")
    by_destination = {}
    for record in files:
        if not isinstance(record, dict) or not record.get("destination"):
            raise SemanticPairError("malformed source audit file record")
        by_destination[Path(record["destination"]).resolve()] = record
    result = {}
    for dataset, path in source_paths.items():
        resolved = path.resolve()
        record = by_destination.get(resolved)
        if record is None:
            raise SemanticPairError(f"source audit does not cover {resolved}")
        observed_hash = sha256_file(resolved)
        if record.get("sha256") != observed_hash:
            raise SemanticPairError(f"source hash drift: {resolved}")
        if int(record.get("kept_rows", -1)) != EXPECTED_ROWS[dataset]:
            raise SemanticPairError(f"source row-count drift: {resolved}")
        result[dataset] = {
            "path": str(resolved),
            "rows": int(record["kept_rows"]),
            "sha256": observed_hash,
            "upstream_source": str(record.get("source", "")),
            "upstream_source_rows": int(record.get("source_rows", -1)),
        }
    if int(audit.get("total_kept_rows", -1)) != sum(EXPECTED_ROWS.values()):
        raise SemanticPairError("source audit total_kept_rows drifted")
    return result


def _validate_verification(row: Mapping[str, Any], *, context: str) -> Dict[str, Any]:
    if row.get("visual_filter_status") != "accept":
        raise SemanticPairError(f"row is not accepted at {context}")
    if row.get("visual_filter_reason") != "verified_negative":
        raise SemanticPairError(f"row is not a verified negative at {context}")
    coverage = proposal_coverage(row, context=context)
    if not coverage["all_proposals_all_no"]:
        raise SemanticPairError(
            f"target plus all cached proposals are not verified no at {context}"
        )
    if not coverage["target_plus_proposal_covered"]:
        raise SemanticPairError(f"strict target-plus-proposal coverage failed at {context}")
    if row.get("tn_scope") != "image_global_proposal_verified":
        raise SemanticPairError(f"unexpected frozen source scope at {context}")
    if row.get("global_tn_verified") is not True:
        raise SemanticPairError(f"frozen source verification flag is not exact true at {context}")
    return coverage


def _make_pair(
    row: Mapping[str, Any],
    *,
    dataset: str,
    source_path: Path,
    source_line: int,
    coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    context = f"{source_path}:{source_line}"
    identity = _identity(row, dataset=dataset, context=context)
    _, image_id, ann_id, ref_id, sent_id = identity
    positive = _clean_text(row.get("sent"))
    negative = _clean_text(row.get("try_tn"))
    if not positive or not negative or positive.casefold() == negative.casefold():
        raise SemanticPairError(f"invalid positive/TN expression pair at {context}")
    try:
        class_id = int(row["class_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticPairError(f"invalid class_id at {context}") from error
    bbox = _valid_xywh(row.get("target_bbox_used"), context=context)
    output = {
        "adapter_pair_schema": PAIR_SCHEMA,
        "source": "stage_b_gdino_adapter_semantic_verified",
        "dataset": dataset,
        "pair_source": row.get("pair_source"),
        "split": "train",
        "image_id": image_id,
        "ann_id": ann_id,
        "ref_id": ref_id,
        "sent_id": sent_id,
        "sample_id": (
            f"semantic-topk:{dataset}:{image_id}:{ann_id}:{ref_id}:{sent_id}"
        ),
        "file_name": f"COCO_train2014_{image_id:012d}.jpg",
        "target_bbox_used": bbox,
        "class_id": class_id,
        "category_name": row.get("category_name"),
        "class_norm_name": row.get("class_norm_name"),
        "sent": positive,
        "try_tn": negative,
        "try_tn_head": row.get("try_tn_head"),
        "try_tn_head_phrase": positive,
        "replace_from": row.get("replace_from"),
        "replace_to": row.get("replace_to"),
        "replace_category": row.get("replace_category"),
        "replace_span": row.get("replace_span"),
        "tn_edits": row.get("tn_edits"),
        "semantic_verified_negative": True,
        "visual_verified_negative": True,
        "coverage_policy": "all_proposals_all_no",
        "proposal_count": int(coverage["proposal_count"]),
        "verification_contract": "target_plus_all_cached_proposals_no",
        "cached_proposal_coverage_only": True,
        "all_900_gdino_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": True,
        "source_tn_scope": row.get("tn_scope"),
        "tn_scope": TN_SCOPE,
        "global_tn_verified": True,
        "proposalset_proxy_verified": False,
        "source_file": str(source_path.resolve()),
        "source_line": int(source_line),
        "source_row_sha256": _canonical_sha256(row),
    }
    return output


def _manifest_record(path: Path) -> Tuple[Dict[str, Any], set[int]]:
    images = set()
    rows = 0
    for _, row in _iter_jsonl(path):
        try:
            images.add(int(row["image_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise SemanticPairError(f"manifest has invalid image_id: {path}") from error
        rows += 1
    return (
        {
            "path": str(path.resolve()),
            "rows": rows,
            "unique_images": len(images),
            "sha256": sha256_file(path),
        },
        images,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    source_paths = {
        "refcocoplus": Path(args.plus_input).resolve(),
        "refcocog": Path(args.g_input).resolve(),
    }
    source_records = _validate_source_audit(
        Path(args.source_audit).resolve(), source_paths
    )
    plus_map, plus_duplicates = _build_plus_map(Path(args.plus_refs).resolve())
    (
        g_umd_map,
        _,
        g_google_map,
        g_umd_duplicates,
        g_google_duplicates,
    ) = _build_g_maps(
        Path(args.g_umd_refs).resolve(), Path(args.g_google_refs).resolve()
    )

    strict2031_record, strict2031_images = _manifest_record(
        Path(args.strict2031).resolve()
    )
    strict1607_record, strict1607_images = _manifest_record(
        Path(args.strict1607).resolve()
    )

    pairs = []
    identities = set()
    dataset_images: Dict[str, set[int]] = {}
    dataset_counts: Dict[str, Counter[str]] = {}
    proposal_histograms: Dict[str, Counter[int]] = {}
    for dataset, source_path in source_paths.items():
        counts: Counter[str] = Counter()
        proposal_histogram: Counter[int] = Counter()
        images: set[int] = set()
        for source_line, row in _iter_jsonl(source_path):
            context = f"{source_path}:{source_line}"
            if row.get("dataset") != dataset or row.get("split") != "train":
                raise SemanticPairError(f"dataset/split mismatch at {context}")
            identity = _identity(row, dataset=dataset, context=context)
            if identity in identities:
                raise SemanticPairError(f"duplicate identity at {context}: {identity}")
            identities.add(identity)
            _, image_id, ann_id, ref_id, sent_id = identity
            if dataset == "refcocoplus":
                split = plus_map.get((ref_id, ann_id, sent_id))
                if split != "train":
                    raise SemanticPairError(
                        f"row is not official RefCOCO+ UNC train at {context}: {split!r}"
                    )
                counts["official_unc_train"] += 1
            else:
                google_split = g_google_map.get((ref_id, ann_id, image_id))
                umd_split = g_umd_map.get(
                    (image_id, ann_id, normalize_sentence(row.get("sent")))
                )
                if google_split != "train" or umd_split != "train":
                    raise SemanticPairError(
                        "row is not official RefCOCOg Google+UMD train at "
                        f"{context}: google={google_split!r}, umd={umd_split!r}"
                    )
                counts["official_google_train"] += 1
                counts["official_umd_train"] += 1
            coverage = _validate_verification(row, context=context)
            proposal_histogram[int(coverage["proposal_count"])] += 1
            counts["target_verified_no"] += 1
            counts["all_cached_proposals_verified_no"] += 1
            counts["rows"] += 1
            images.add(image_id)
            pairs.append(
                _make_pair(
                    row,
                    dataset=dataset,
                    source_path=source_path,
                    source_line=source_line,
                    coverage=coverage,
                )
            )
        if counts["rows"] != EXPECTED_ROWS[dataset]:
            raise SemanticPairError(
                f"unexpected {dataset} rows: {counts['rows']} != {EXPECTED_ROWS[dataset]}"
            )
        dataset_images[dataset] = images
        dataset_counts[dataset] = counts
        proposal_histograms[dataset] = proposal_histogram

    expected_total = int(args.expected_rows)
    if len(pairs) != expected_total or len(identities) != expected_total:
        raise SemanticPairError(
            f"expected {expected_total} unique pairs, got {len(pairs)} rows and "
            f"{len(identities)} identities"
        )
    all_images = set().union(*dataset_images.values())
    strict1607_overlap = all_images.intersection(strict1607_images)
    if strict1607_overlap:
        raise SemanticPairError(
            "semantic strict1607 must remain image-disjoint, found overlap: "
            f"{sorted(strict1607_overlap)[:16]}"
        )

    pairs.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["image_id"]),
            int(row["ann_id"]),
            int(row["ref_id"]),
            int(row["sent_id"]),
        )
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in pairs:
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
    os.replace(temporary, output)

    for dataset, record in source_records.items():
        record.update(
            {
                "official_split_counts": dict(sorted(dataset_counts[dataset].items())),
                "unique_images": len(dataset_images[dataset]),
                "proposal_count_histogram": {
                    str(key): value
                    for key, value in sorted(proposal_histograms[dataset].items())
                },
                "strict2031_image_overlap": len(
                    dataset_images[dataset].intersection(strict2031_images)
                ),
                "strict1607_image_overlap": 0,
            }
        )
    audit = {
        "schema": SCHEMA,
        "rows": len(pairs),
        "unique_identities": len(identities),
        "unique_images": len(all_images),
        "tn_scope": TN_SCOPE,
        "proposalset_proxy_verified": False,
        "cached_proposal_coverage_only": True,
        "all_900_gdino_queries_verified": False,
        "global_max_label_is_semantic_extrapolation": True,
        "source_audit": {
            "path": str(Path(args.source_audit).resolve()),
            "sha256": sha256_file(Path(args.source_audit).resolve()),
            "schema": SOURCE_SCHEMA,
        },
        "sources": source_records,
        "official_refs": {
            "refcocoplus_unc": {
                "path": str(Path(args.plus_refs).resolve()),
                "sha256": sha256_file(Path(args.plus_refs).resolve()),
                "mapping_duplicates": plus_duplicates,
            },
            "refcocog_umd": {
                "path": str(Path(args.g_umd_refs).resolve()),
                "sha256": sha256_file(Path(args.g_umd_refs).resolve()),
                "mapping_duplicates": g_umd_duplicates,
            },
            "refcocog_google": {
                "path": str(Path(args.g_google_refs).resolve()),
                "sha256": sha256_file(Path(args.g_google_refs).resolve()),
                "mapping_duplicates": g_google_duplicates,
            },
        },
        "overlap_audit": {
            "cross_dataset_image_overlap": len(
                dataset_images["refcocoplus"].intersection(dataset_images["refcocog"])
            ),
            "strict2031": {
                **strict2031_record,
                "train_image_overlap": len(all_images.intersection(strict2031_images)),
                "claim": "overlap disclosed; strict2031 remains the authoritative full gate",
            },
            "strict1607": {
                **strict1607_record,
                "train_image_overlap": 0,
                "claim": "exactly image-disjoint from this semantic probe source",
            },
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
        "claims": {
            "positive": (
                "Official train rows only; target VLM answer=no; non-empty cached "
                "proposal set; every cached proposal has a matching VLM answer=no."
            ),
            "scope_limit": (
                "image_global_topk_verified means target-plus-cached-top-k/proposal "
                "coverage. It does not mean that all 900 GDINO decoder queries were "
                "individually verified."
            ),
            "global_max_supervision": (
                "The confidence loss applies the semantic negative label to the model's "
                "all-query maximum as a generalization objective."
            ),
            "proposalset_proxy_false": (
                "Rows carry a semantic image-expression negative label after coverage "
                "verification; they are not old proposal-index score targets."
            ),
        },
    }
    _write_json(Path(args.audit).resolve(), audit)
    return audit


def verify(args: argparse.Namespace) -> Dict[str, Any]:
    audit = _read_json(Path(args.audit).resolve())
    if audit.get("schema") != SCHEMA or audit.get("tn_scope") != TN_SCOPE:
        raise SemanticPairError("semantic pair audit schema/scope mismatch")
    output = Path(args.output).resolve()
    recorded_output = remap_legacy_path(
        audit.get("output", ""), repo_root=REPO_ROOT, data_root=DATA_ROOT
    ).resolve()
    if recorded_output != output or audit.get("output_sha256") != sha256_file(output):
        raise SemanticPairError("semantic pair output hash/path drifted")
    rows = 0
    identities = set()
    for line_number, row in _iter_jsonl(output):
        if (
            row.get("adapter_pair_schema") != PAIR_SCHEMA
            or row.get("tn_scope") != TN_SCOPE
            or row.get("global_tn_verified") is not True
            or row.get("proposalset_proxy_verified") is not False
            or row.get("semantic_verified_negative") is not True
            or row.get("coverage_policy") != "all_proposals_all_no"
            or row.get("cached_proposal_coverage_only") is not True
            or row.get("all_900_gdino_queries_verified") is not False
            or row.get("global_max_label_is_semantic_extrapolation") is not True
        ):
            raise SemanticPairError(f"invalid pair contract at {output}:{line_number}")
        identity = _identity(
            row, dataset=str(row.get("dataset", "")), context=f"{output}:{line_number}"
        )
        if identity in identities:
            raise SemanticPairError(f"duplicate output identity: {identity}")
        identities.add(identity)
        rows += 1
    if rows != int(audit.get("rows", -1)) or rows != int(args.expected_rows):
        raise SemanticPairError("semantic pair output row count drifted")
    for record in audit.get("sources", {}).values():
        path = remap_legacy_path(
            record["path"], repo_root=REPO_ROOT, data_root=DATA_ROOT
        )
        if sha256_file(path) != record.get("sha256"):
            raise SemanticPairError(f"semantic source hash drifted: {path}")
    return {
        "schema": SCHEMA,
        "rows": rows,
        "output": str(output),
        "output_sha256": audit["output_sha256"],
        "verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plus-input",
        default=SOURCE_DIR / "refcocoplus_verified_train.global_verified.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--g-input",
        default=SOURCE_DIR / "refcocog_verified_train.global_verified.jsonl",
        type=Path,
    )
    parser.add_argument("--source-audit", default=SOURCE_DIR / "audit.json", type=Path)
    parser.add_argument(
        "--plus-refs", default=DATA_ROOT / "COCO/refcoco+/refs(unc).p", type=Path
    )
    parser.add_argument(
        "--g-umd-refs", default=DATA_ROOT / "COCO/refcocog/refs(umd).p", type=Path
    )
    parser.add_argument(
        "--g-google-refs",
        default=DATA_ROOT / "COCO/refcocog/refs(google).p",
        type=Path,
    )
    strict_dir = (
        REPO_ROOT
        / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
    )
    parser.add_argument(
        "--strict2031", default=strict_dir / "eval_manifest.jsonl", type=Path
    )
    parser.add_argument(
        "--strict1607",
        default=strict_dir / "semantic_stageb_union_image_disjoint_manifest.jsonl",
        type=Path,
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR / "semantic_verified_pairs.jsonl", type=Path
    )
    parser.add_argument("--audit", default=OUTPUT_DIR / "audit.json", type=Path)
    parser.add_argument("--expected-rows", type=int, default=sum(EXPECTED_ROWS.values()))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if int(args.expected_rows) != sum(EXPECTED_ROWS.values()):
        parser.error(f"--expected-rows must remain {sum(EXPECTED_ROWS.values())}")
    return args


def main() -> None:
    args = parse_args()
    result = verify(args) if args.verify_only else build(args)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
