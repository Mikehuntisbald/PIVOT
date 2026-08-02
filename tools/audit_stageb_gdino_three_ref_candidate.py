#!/usr/bin/env python3
"""Audit annotation/image overlap for the pure-GDINO three-Ref candidate."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple


AnnKey = Tuple[int, int]


@dataclass(frozen=True)
class RefSplit:
    name: str
    dataset_dir: str
    refs_name: str
    split: str


SPLITS = (
    RefSplit("refcoco_val", "refcoco", "refs(unc).p", "val"),
    RefSplit("refcoco_testA", "refcoco", "refs(unc).p", "testA"),
    RefSplit("refcoco_testB", "refcoco", "refs(unc).p", "testB"),
    RefSplit("refcocop_val", "refcoco+", "refs(unc).p", "val"),
    RefSplit("refcocop_testA", "refcoco+", "refs(unc).p", "testA"),
    RefSplit("refcocop_testB", "refcoco+", "refs(unc).p", "testB"),
    RefSplit("refcocog_val", "refcocog", "refs(umd).p", "val"),
    RefSplit("refcocog_test", "refcocog", "refs(umd).p", "test"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_jsonl_identity(path: Path, *, require_ann_id: bool) -> Dict[str, Any]:
    images: Set[int] = set()
    annotations: Set[AnnKey] = set()
    rows = 0
    missing_ann_id = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("image_id") is None:
                raise ValueError(f"Missing image_id at {path}:{line_number}")
            image_id = int(row["image_id"])
            images.add(image_id)
            rows += 1
            if row.get("ann_id") is None:
                missing_ann_id += 1
                continue
            annotations.add((image_id, int(row["ann_id"])))
    if require_ann_id and missing_ann_id:
        raise ValueError(f"{path} has {missing_ann_id} rows without ann_id")
    return {
        "path": str(path),
        "rows": rows,
        "images": images,
        "annotations": annotations,
        "missing_ann_id_rows": missing_ann_id,
    }


def _dataset_images(repo_root: Path, datasets_path: Path) -> Dict[str, Any]:
    meta = json.loads(datasets_path.read_text(encoding="utf-8"))
    images: Set[int] = set()
    rows = 0
    entries = []
    for index, entry in enumerate(meta.get("train", [])):
        anno_path = _resolve(repo_root, str(entry["anno"]))
        identity = _load_jsonl_identity(anno_path, require_ann_id=False)
        images.update(identity["images"])
        rows += int(identity["rows"])
        entries.append(
            {
                "index": index,
                "anno": str(anno_path),
                "rows": int(identity["rows"]),
                "unique_images": len(identity["images"]),
            }
        )
    return {"rows": rows, "images": images, "entries": entries}


def _eval_identity(refs_path: Path, split: str) -> Dict[str, Any]:
    refs = pickle.load(refs_path.open("rb"))
    images: Set[int] = set()
    annotations: Set[AnnKey] = set()
    ref_ids: Set[int] = set()
    sentence_ids: Set[int] = set()
    sentence_rows = 0
    for ref in refs:
        if str(ref.get("split")) != split:
            continue
        image_id = int(ref["image_id"])
        ann_id = int(ref["ann_id"])
        images.add(image_id)
        annotations.add((image_id, ann_id))
        ref_ids.add(int(ref["ref_id"]))
        for sentence in ref.get("sentences", []) or []:
            sentence_rows += 1
            if sentence.get("sent_id") is not None:
                sentence_ids.add(int(sentence["sent_id"]))
    return {
        "images": images,
        "annotations": annotations,
        "ref_ids": ref_ids,
        "sentence_ids": sentence_ids,
        "sentence_rows": sentence_rows,
    }


def _union(records: Iterable[Dict[str, Any]], key: str) -> Set[Any]:
    result: Set[Any] = set()
    for record in records:
        result.update(record[key])
    return result


def _overlap(scope: Set[Any], evaluation: Set[Any]) -> int:
    return len(scope.intersection(evaluation))


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = _repo_root()
    data_root = _resolve(repo_root, args.data_root)
    baseline_path = _resolve(repo_root, args.baseline_datasets)
    candidate_path = _resolve(repo_root, args.candidate_datasets)

    train_sources = {
        "refcoco_train_added": _load_jsonl_identity(
            _resolve(repo_root, args.refcoco_source), require_ann_id=True
        ),
        "refcocoplus_train_baseline": _load_jsonl_identity(
            _resolve(repo_root, args.refcocoplus_source), require_ann_id=True
        ),
        "refcocog_train_baseline": _load_jsonl_identity(
            _resolve(repo_root, args.refcocog_source), require_ann_id=True
        ),
        "tn_source_baseline": _load_jsonl_identity(
            _resolve(repo_root, args.tn_source), require_ann_id=True
        ),
    }

    baseline_images = _dataset_images(repo_root, baseline_path)
    candidate_images = _dataset_images(repo_root, candidate_path)
    candidate_reused = candidate_images["entries"][:2] + candidate_images["entries"][3:]
    candidate_reused = [{k: v for k, v in row.items() if k != "index"} for row in candidate_reused]
    baseline_reused = [
        {k: v for k, v in row.items() if k != "index"} for row in baseline_images["entries"]
    ]
    if candidate_reused != baseline_reused:
        raise RuntimeError("Candidate entries do not reduce to the fixed baseline entries")

    positive_sources = [
        train_sources["refcoco_train_added"],
        train_sources["refcocoplus_train_baseline"],
        train_sources["refcocog_train_baseline"],
    ]
    all_ref_positive_annotations = _union(positive_sources, "annotations")
    all_ref_positive_images = _union(positive_sources, "images")
    all_ref_and_tn_annotations = _union(train_sources.values(), "annotations")
    all_ref_and_tn_images = _union(train_sources.values(), "images")
    image_delta = candidate_images["images"] - baseline_images["images"]

    split_results: Dict[str, Any] = {}
    for spec in SPLITS:
        refs_path = data_root / "COCO" / spec.dataset_dir / spec.refs_name
        evaluation = _eval_identity(refs_path, spec.split)
        split_results[spec.name] = {
            "refs_path": str(refs_path),
            "split": spec.split,
            "eval_unique_annotations": len(evaluation["annotations"]),
            "eval_unique_images": len(evaluation["images"]),
            "eval_unique_ref_ids": len(evaluation["ref_ids"]),
            "eval_sentence_rows": int(evaluation["sentence_rows"]),
            "added_refcoco_annotation_overlap": _overlap(
                train_sources["refcoco_train_added"]["annotations"], evaluation["annotations"]
            ),
            "added_refcoco_image_overlap": _overlap(
                train_sources["refcoco_train_added"]["images"], evaluation["images"]
            ),
            "all_ref_positive_annotation_overlap": _overlap(
                all_ref_positive_annotations, evaluation["annotations"]
            ),
            "all_ref_positive_image_overlap": _overlap(
                all_ref_positive_images, evaluation["images"]
            ),
            "all_ref_and_tn_annotation_overlap": _overlap(
                all_ref_and_tn_annotations, evaluation["annotations"]
            ),
            "all_ref_and_tn_image_overlap": _overlap(
                all_ref_and_tn_images, evaluation["images"]
            ),
            "fixed_baseline_all_entry_image_overlap": _overlap(
                baseline_images["images"], evaluation["images"]
            ),
            "candidate_all_entry_image_overlap": _overlap(
                candidate_images["images"], evaluation["images"]
            ),
            "candidate_new_image_id_overlap": _overlap(image_delta, evaluation["images"]),
        }

    result = {
        "baseline_datasets": str(baseline_path),
        "candidate_datasets": str(candidate_path),
        "annotation_key": ["image_id", "ann_id"],
        "annotation_scope_note": (
            "Annotation overlap is exact for Ref-expression/TN prebuilt sources. "
            "The LVIS/COCO ODVG files do not retain source ann_id, so all-entry "
            "candidate overlap is reported at image granularity only."
        ),
        "candidate_is_strict_one_entry_insertion": True,
        "train_sources": {
            name: {
                "path": record["path"],
                "rows": record["rows"],
                "unique_annotations": len(record["annotations"]),
                "unique_images": len(record["images"]),
                "missing_ann_id_rows": record["missing_ann_id_rows"],
            }
            for name, record in train_sources.items()
        },
        "fixed_baseline": {
            "rows": baseline_images["rows"],
            "unique_images": len(baseline_images["images"]),
            "entries": baseline_images["entries"],
        },
        "candidate": {
            "rows": candidate_images["rows"],
            "unique_images": len(candidate_images["images"]),
            "new_unique_images_vs_baseline": len(image_delta),
            "entries": candidate_images["entries"],
        },
        "splits": split_results,
    }
    output_path = _resolve(repo_root, args.output)
    _write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/user/datasets/pivot_data")
    parser.add_argument(
        "--baseline-datasets",
        default="config/ablations/gdino_ft_stage_b_rebuild_20260711/datasets_gdino_ft_stageb_with_tn_local.json",
    )
    parser.add_argument(
        "--candidate-datasets",
        default="config/ablations/gdino_ft_stage_b_rebuild_20260711/datasets_gdino_ft_stageb_three_ref_with_tn_local.json",
    )
    parser.add_argument(
        "--refcoco-source",
        default="data/ablations/stageb_refexp_three_train_20260711/refcoco_stageb_phrase_v1.jsonl",
    )
    parser.add_argument(
        "--refcocoplus-source",
        default="/home/user/datasets/pivot_data/patch_episode_prebuilt/refcocoplus_stageb_phrase_v1.jsonl",
    )
    parser.add_argument(
        "--refcocog-source",
        default="/home/user/datasets/pivot_data/patch_episode_prebuilt/refcocog_stageb_phrase_v1.jsonl",
    )
    parser.add_argument(
        "--tn-source",
        default="/home/user/datasets/pivot_data/patch_episode_prebuilt/refexp_tn_stageb_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/ablations/gdino_ft_stage_b_rebuild_three_ref_20260711/ref_split_overlap_audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    report = audit(parse_args())
    print(json.dumps(report, indent=2))
