#!/usr/bin/env python3
"""Append-only unused-field erratum for FineCops readout records.

The frozen v1 evaluator copied annotation `level` into negative_edit_level.
Official edit difficulty is `negative_level`, which may be absent. This tool
exports new v2 records, preserving every original field except that unused
diagnostic and adding explicit raw/revision fields. No original file, model,
score, metric, pairing, positive difficulty or winner diagnostic is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path

ROW_SCHEMA = "arrow.confidence_readout.record/v2"
REVISION = "negative_edit_level_from_official_annotation/v1"
SEEDS = ("17", "42", "73")
ALLOWED = frozenset({"schema", "metadata_revision", "negative_edit_level", "raw_annotation_level"})


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def verify(binding):
    path = Path(binding["path"]).resolve(strict=True)
    if sha(path) != binding["sha256"]:
        raise ValueError("source SHA drift: " + str(path))
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_json(text):
    value = json.loads(text, object_pairs_hook=_unique_object)
    # Reject NaN/Infinity accepted by Python's default JSON decoder.
    json.dumps(value, allow_nan=False)
    return value


def read_binding(binding):
    return parse_json(verify(binding).read_text())


def records(path):
    with Path(path).open() as handle:
        return [parse_json(line) for line in handle if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def protected_projection(row):
    """All existing scientific/identity values, not a narrow score-only check."""
    return {key: value for key, value in row.items() if key not in ALLOWED}


def _integer(value, name, allow_none=False):
    if value is None and allow_none:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer required: " + name)
    return value


def official_lookup(annotation, index_rows):
    annotations = annotation.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("COCO-format official annotation required")
    by_ann = {}
    for row in annotations:
        aid = _integer(row["id"], "annotation ID")
        if aid in by_ann:
            raise ValueError("duplicate official annotation ID")
        by_ann[aid] = row
    source, by_id = {}, {}
    for item in index_rows:
        sid = item["sample_id"]
        aid = _integer(item["annotation_id"], "index annotation ID")
        if not isinstance(sid, str) or not sid or sid in source or aid in by_id:
            raise ValueError("cache index identities must be unique")
        if sid != "finecops-val:" + str(aid):
            raise ValueError("only the sealed FineCops val sample-ID contract is supported")
        if aid not in by_ann:
            raise ValueError("cache index annotation is absent from the official source")
        ann = by_ann[aid]
        kind = "text" if ann.get("negative_type") is not None else "positive"
        raw_level = _integer(ann.get("level"), "raw annotation level")
        negative_level = _integer(ann.get("negative_level"), "official negative_level", allow_none=True)
        expected_parent = _integer(ann.get("positive_id") if kind == "text" else aid, "parent annotation ID")
        if (item["kind"] != kind or item["level"] != raw_level
                or item["parent_positive_id"] != expected_parent):
            raise ValueError("official annotation and sealed cache index disagree")
        source[sid] = {"annotation_id": aid, "kind": kind, "cluster_id": str(item["cluster_image_id"]),
                       "raw_annotation_level": raw_level, "negative_edit_level": negative_level if kind == "text" else None,
                       "parent_annotation_id": expected_parent}
        by_id[aid] = sid
    if set(by_id) != set(by_ann):
        raise ValueError("cache index must cover every official validation annotation")
    for sid, row in source.items():
        if row["kind"] == "text":
            parent_sid = by_id.get(row["parent_annotation_id"])
            parent = source.get(parent_sid)
            if parent is None or parent["kind"] != "positive" or parent["cluster_id"] != row["cluster_id"]:
                raise ValueError("negative must retain its real same-image positive parent")
            row["parent_positive_id"] = parent_sid
            row["parent_positive_level"] = parent["raw_annotation_level"]
        else:
            row["parent_positive_id"] = None
            row["parent_positive_level"] = None
    return source


def correct_record(row, source):
    sid = row.get("sample_id")
    if sid not in source:
        raise ValueError("record identity absent from the sealed annotation/index")
    if "metadata_revision" in row or "raw_annotation_level" in row or "schema" in row:
        raise ValueError("only original v1 evaluator records may receive this erratum")
    official = source[sid]
    expected_level = official["raw_annotation_level"] if official["kind"] == "positive" else None
    expected_old_edit = official["raw_annotation_level"] if official["kind"] == "text" else None
    for key, expected in ("kind", official["kind"]), ("cluster_id", official["cluster_id"]), (
        "parent_positive_id", official["parent_positive_id"]), ("parent_positive_level", official["parent_positive_level"]), (
        "level", expected_level), ("negative_edit_level", expected_old_edit):
        if row.get(key, "__missing__") != expected:
            raise ValueError("record is not the known frozen-v1 metadata pattern: " + key)
    if row.get("split") != "val" or row.get("stratum") != "validation":
        raise ValueError("only FineCops validation evaluation records are in scope")
    for field in ("scores", "readout_diagnostics", "native_score", "correct"):
        if field not in row:
            raise ValueError("complete evaluated record required: " + field)
    result = copy.deepcopy(row)
    result.update(schema=ROW_SCHEMA, metadata_revision=REVISION,
                  negative_edit_level=official["negative_edit_level"],
                  raw_annotation_level=official["raw_annotation_level"])
    if canonical(protected_projection(result)) != canonical(protected_projection(row)):
        raise ValueError("erratum changed a protected scientific field")
    return result


def reconcile_rows(rows, source):
    seen, output, before, after = set(), [], hashlib.sha256(), hashlib.sha256()
    changes = Counter()
    for row in rows:
        sid = row.get("sample_id")
        if sid in seen:
            raise ValueError("duplicate evaluated sample identity")
        seen.add(sid)
        fixed = correct_record(row, source)
        before.update(canonical(protected_projection(row)) + b"\n")
        after.update(canonical(protected_projection(fixed)) + b"\n")
        if row["negative_edit_level"] != fixed["negative_edit_level"]:
            changes[(str(row["negative_edit_level"]), str(fixed["negative_edit_level"]))] += 1
        output.append(fixed)
    if seen != set(source):
        raise ValueError("evaluated records do not cover the exact sealed annotation population")
    if before.hexdigest() != after.hexdigest():
        raise ValueError("protected projection differs")
    return output, {"protected_projection_sha256": before.hexdigest(),
                    "all_other_fields_canonical_bitwise_identical": True,
                    "edit_level_changes": [{"from": a, "to": b, "records": n}
                                           for (a,b),n in sorted(changes.items())],
                    "changed_edit_level_records": sum(changes.values())}


def write_new_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError("append-only erratum output exists")
    with temporary.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n");handle.flush();os.fsync(handle.fileno())
    temporary.rename(path)


def write_new_records(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError("append-only erratum records exist")
    with temporary.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush();os.fsync(handle.fileno())
    temporary.rename(path)


def run(args):
    original = bind(args.evaluation_postflight)
    post = read_binding(original)
    if (post.get("schema") != "arrow.confidence_readout.cache_evaluation_postflight/v1"
            or post.get("status") != "complete" or post.get("mode") != "evaluation"
            or set(post.get("records", {})) != set(SEEDS)):
        raise ValueError("complete original three-seed evaluation postflight required")
    design = read_binding(post["design"])
    manifest = read_binding(design["cache"])
    if manifest.get("status") != "complete" or manifest.get("formal") is not True or manifest.get("split") != "val":
        raise ValueError("sealed FineCops val cache manifest required")
    annotation = read_binding(manifest["annotation"])
    index_path = verify(manifest["index"])
    index = records(index_path)
    source = official_lookup(annotation, index)
    if len(source) != 18455 or dict(Counter(r["kind"] for r in source.values())) != {"positive":9426,"text":9029}:
        raise ValueError("official FineCops validation population drift")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("new metadata export directory must not exist")
    # Read and verify every source before creating any new record export.
    corrected, audits = {}, {}
    for seed in SEEDS:
        path = verify(post["records"][seed])
        corrected[seed], audits[seed] = reconcile_rows(records(path), source)
    output.mkdir(parents=True)
    bindings = {}
    for seed in SEEDS:
        path = output/f"seed{seed}_records_metadata_v2.jsonl"
        write_new_records(path, corrected[seed]);bindings[seed]=bind(path)
    sources = {"original_evaluation_postflight": original, "original_evaluation_design": post["design"],
               "cache_manifest": design["cache"], "official_annotation": manifest["annotation"],
               "cache_index": manifest["index"], "original_records": post["records"]}
    for item in (original,post["design"],design["cache"],manifest["annotation"],manifest["index"],*post["records"].values()):
        verify(item)
    official_counts = Counter((r["raw_annotation_level"],r["negative_edit_level"])
                              for r in source.values() if r["kind"]=="text")
    receipt = {"schema":"arrow.confidence_readout.metadata_erratum/v1","status":"complete",
        "revision":REVISION,"new_record_schema":ROW_SCHEMA,"localizer":post["localizer"],
        "description":"Unused diagnostic field only: negative_edit_level now comes from official negative_level; raw annotation level is stored separately.",
        "allowed_changes":sorted(ALLOWED),"sources":sources,"records":bindings,"audits":audits,
        "official_negative_levels":[{"raw_annotation_level":a,"negative_edit_level":b,"records":n}
                                    for (a,b),n in sorted(official_counts.items(),key=lambda x:(x[0][0],-1 if x[0][1] is None else x[0][1]))],
        "metric_consumed_fields_changed":False,"positive_and_parent_difficulty_changed":False,
        "scores_or_winner_diagnostics_changed":False,"sample_order_changed":False,
        "original_artifacts_retained":True,"model_forward":False,"metric_recompute":False,
        "threshold_fitting":False,"producer":bind(__file__)}
    write_new_json(output/"receipt.json",receipt)
    print(json.dumps({"receipt":bind(output/"receipt.json"),"originals_retained":True,
                      "changed_unused_fields_per_seed":audits[SEEDS[0]]["changed_edit_level_records"]}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-postflight",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
