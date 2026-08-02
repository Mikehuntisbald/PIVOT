#!/usr/bin/env python3
"""Build paired, fail-closed category interventions for paper Table A.

Each pair keeps the image and frozen checkpoint fixed while changing both the
canonical category prompt and the support patch to another category that is
actually annotated in the same image.  The generated support TSV contains
exactly one clean asset per used category, making the support intervention
deterministic at runtime without changing the shared dataset implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.path_compat import remap_legacy_path  # noqa: E402


SCHEMA = "stageb-table-a-category-intervention-pair-v1"
AUDIT_SCHEMA = "stageb-table-a-category-intervention-audit-v1"
CONTRACT = {
    "same_image": True,
    "same_frozen_checkpoint": True,
    "changed_inputs": ["canonical_category_prompt", "support_patch"],
    "both_categories_have_ground_truth": True,
    "not_exhaustive_categories_forbidden": True,
    "one_support_asset_per_category": True,
    "human_judgment_required": False,
}
DEFAULT_SOURCE = (
    REPO_ROOT
    / "data/ablations/ogc_original_finetune_stage_a_20260711/"
    "stagea_odvg_val_0_lvis.jsonl"
)
DEFAULT_CANONICAL_MAP = (
    REPO_ROOT
    / "data/ablations/ogc_original_finetune_stage_a_20260711/"
    "stagea_odvg_canonical_label_map.json"
)
DEFAULT_SUPPORT_TSV = Path(
    "/media/haoyi/T9/data/patches_quality_emb/emb_index_from_quality.tsv"
)
DEFAULT_SUPPORT_IMAGE_ROOT = Path("/media/haoyi/T9/data/patches_quality")
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_table_a_category_intervention_20260717"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "category_intervention_pairs.jsonl"
DEFAULT_OUTPUT_SUPPORT_TSV = DEFAULT_OUTPUT_ROOT / "category_intervention_support.tsv"
DEFAULT_AUDIT = DEFAULT_OUTPUT_ROOT / "audit.json"
DEFAULT_SEED = 170717
DEFAULT_MAX_PAIRS = 512
DEFAULT_MAX_CROSS_IOU = 0.10
_WS_RE = re.compile(r"\s+")


def _norm(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "").replace("_", " ").strip().lower())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hex(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield line_number, value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")


def _load_canonical_map(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("canonical map must be a JSON object")
    by_id: Dict[int, str] = {}
    by_name: Dict[str, int] = {}
    for raw_id, raw_name in value.items():
        class_id = int(raw_id)
        name = str(raw_name).strip()
        key = _norm(name)
        if not name or not key:
            raise ValueError("canonical map contains an empty name")
        by_id[class_id] = name
        # PatchEpisodeDataset's canonical name map uses last-entry-wins for
        # duplicate normalized aliases.  Match that runtime policy so an
        # ambiguous TSV class can never bind a different canonical ID here.
        by_name[key] = class_id
    return by_id, by_name


def _effective_support_path(
    row: Mapping[str, str], *, support_image_root: Path
) -> Path:
    rel = str(row.get("emb_rel_path", "")).strip()
    if rel:
        mirror = support_image_root / rel
        if mirror.suffix == ".npy":
            mirror = mirror.with_suffix(".jpg")
        if mirror.is_file():
            return mirror.resolve()
    raw = str(row.get("path", "")).strip()
    if not raw:
        raise ValueError("support TSV row has no image path")
    return Path(raw).expanduser().resolve()


def _load_support_choices(
    path: Path,
    *,
    by_name: Mapping[str, int],
    support_image_root: Path,
    seed: int,
) -> Tuple[List[str], Dict[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "class" not in reader.fieldnames:
            raise ValueError("support TSV must contain a class column")
        fields = list(reader.fieldnames)
        choices: Dict[int, List[Dict[str, Any]]] = {}
        for row in reader:
            if str(row.get("bucket", "")) != "clean":
                continue
            class_id = by_name.get(_norm(row.get("class")))
            if class_id is None:
                continue
            effective = _effective_support_path(
                row, support_image_root=support_image_root
            )
            if not effective.is_file():
                continue
            choices.setdefault(class_id, []).append(
                {
                    "tsv_row": dict(row),
                    "effective_path": str(effective),
                    "priority": _stable_hex(seed, class_id, effective),
                }
            )
    selected: Dict[int, Dict[str, Any]] = {}
    for class_id, rows in choices.items():
        chosen = min(rows, key=lambda row: row["priority"])
        effective = Path(chosen["effective_path"])
        selected[class_id] = {
            **chosen,
            "sha256": _sha256_file(effective),
            "size_bytes": effective.stat().st_size,
        }
    return fields, selected


def _valid_xyxy(value: Any) -> List[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    return inter / max(area_a + area_b - inter, 1e-12)


def _max_cross_iou(
    boxes_a: Sequence[Sequence[float]], boxes_b: Sequence[Sequence[float]]
) -> float:
    return max((_iou(a, b) for a in boxes_a for b in boxes_b), default=0.0)


def _pair_candidates(
    source: Path,
    *,
    canonical_by_id: Mapping[int, str],
    support: Mapping[int, Mapping[str, Any]],
    seed: int,
    max_cross_iou: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for line_number, row in _iter_jsonl(source):
        image_id = int(row.get("image_id", -1))
        filename = str(row.get("filename", ""))
        width = int(row.get("width", 0))
        height = int(row.get("height", 0))
        if image_id < 0 or not filename or width <= 0 or height <= 0:
            continue
        resolved_image = remap_legacy_path(filename).expanduser().resolve()
        if not resolved_image.is_file():
            continue
        forbidden = {
            int(value) for value in (row.get("not_exhaustive_labels", []) or [])
        }
        grouped: Dict[int, List[List[float]]] = {}
        instances = (row.get("detection") or {}).get("instances", [])
        for instance in instances:
            if not isinstance(instance, Mapping):
                continue
            class_id = int(instance.get("label", -1))
            box = _valid_xyxy(instance.get("bbox"))
            if (
                class_id in forbidden
                or class_id not in canonical_by_id
                or class_id not in support
                or box is None
            ):
                continue
            grouped.setdefault(class_id, []).append(box)
        local: List[Dict[str, Any]] = []
        for class_a, class_b in itertools.combinations(sorted(grouped), 2):
            cross_iou = _max_cross_iou(grouped[class_a], grouped[class_b])
            if cross_iou > float(max_cross_iou):
                continue
            local.append(
                {
                    "source_line": line_number,
                    "image_id": image_id,
                    "filename": filename,
                    "resolved_image": str(resolved_image),
                    "width": width,
                    "height": height,
                    "class_a": class_a,
                    "class_b": class_b,
                    "boxes_a": grouped[class_a],
                    "boxes_b": grouped[class_b],
                    "max_cross_iou": cross_iou,
                    "priority": _stable_hex(seed, image_id, class_a, class_b),
                }
            )
        if local:
            candidates.append(min(local, key=lambda value: value["priority"]))
    return sorted(candidates, key=lambda value: value["priority"])


def _arm_row(
    pair: Mapping[str, Any],
    *,
    arm: str,
    canonical_by_id: Mapping[int, str],
    support: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    if arm not in {"A", "B"}:
        raise ValueError("category intervention arm must be A or B")
    own_suffix, other_suffix = ("a", "b") if arm == "A" else ("b", "a")
    own_id = int(pair[f"class_{own_suffix}"])
    other_id = int(pair[f"class_{other_suffix}"])
    own_name = canonical_by_id[own_id]
    other_name = canonical_by_id[other_id]
    pair_id = "cat-int:" + _stable_hex(
        pair["image_id"], pair["class_a"], pair["class_b"]
    )[:24]
    identity = int(_stable_hex(pair_id, arm)[:15], 16)
    own_support = support[own_id]
    other_support = support[other_id]
    intervention = {
        "schema": SCHEMA,
        "pair_id": pair_id,
        "arm": arm,
        "image_width": int(pair["width"]),
        "image_height": int(pair["height"]),
        "image_path": str(pair["resolved_image"]),
        "image_sha256": _sha256_file(Path(pair["resolved_image"])),
        "class_a": {
            "id": int(pair["class_a"]),
            "name": canonical_by_id[int(pair["class_a"])],
            "boxes_xyxy": pair["boxes_a"],
            "support_path": support[int(pair["class_a"])]["effective_path"],
            "support_sha256": support[int(pair["class_a"])]["sha256"],
        },
        "class_b": {
            "id": int(pair["class_b"]),
            "name": canonical_by_id[int(pair["class_b"])],
            "boxes_xyxy": pair["boxes_b"],
            "support_path": support[int(pair["class_b"])]["effective_path"],
            "support_sha256": support[int(pair["class_b"])]["sha256"],
        },
        "active_class_id": own_id,
        "active_class_name": own_name,
        "counterfactual_class_id": other_id,
        "counterfactual_class_name": other_name,
        "canonical_prompt": f"{own_name} .",
        "active_support_path": own_support["effective_path"],
        "active_support_sha256": own_support["sha256"],
        "counterfactual_support_path": other_support["effective_path"],
        "counterfactual_support_sha256": other_support["sha256"],
        "max_cross_category_gt_iou": float(pair["max_cross_iou"]),
        "prompt_and_support_changed_together": True,
    }
    return {
        "schema": SCHEMA,
        "filename": str(pair["filename"]),
        "source": "table_a_category_intervention_lvis_val",
        "split": "val",
        "image_id": int(pair["image_id"]),
        "ann_id": identity,
        "ref_id": identity,
        "sent_id": 0 if arm == "A" else 1,
        "sample_id": f"{pair_id}:{arm}",
        "instances": [
            {
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1]),
                ],
                "class_id": own_id,
                "raw_phrase": own_name,
                "head_phrase": own_name,
                "head": own_name,
                "canonical_name": own_name,
                "positive_phrase": own_name,
                "text_is_negative": False,
            }
            for box in pair[f"boxes_{own_suffix}"]
        ],
        "category_intervention": intervention,
    }


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != SCHEMA:
            raise ValueError("category intervention row schema mismatch")
        intervention = row.get("category_intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError("category intervention metadata is missing")
        groups.setdefault(str(intervention.get("pair_id")), []).append(row)
    used_classes = set()
    for pair_id, pair_rows in groups.items():
        if len(pair_rows) != 2:
            raise ValueError(f"{pair_id} does not contain exactly two arms")
        by_arm = {
            str(row["category_intervention"]["arm"]): row for row in pair_rows
        }
        if set(by_arm) != {"A", "B"}:
            raise ValueError(f"{pair_id} arms are not exactly A/B")
        a = by_arm["A"]["category_intervention"]
        b = by_arm["B"]["category_intervention"]
        for key in ("image_path", "image_sha256", "image_width", "image_height"):
            if a[key] != b[key]:
                raise ValueError(f"{pair_id} image binding differs at {key}")
        if a["active_class_id"] != b["counterfactual_class_id"]:
            raise ValueError(f"{pair_id} A class does not match B counterfactual")
        if b["active_class_id"] != a["counterfactual_class_id"]:
            raise ValueError(f"{pair_id} B class does not match A counterfactual")
        if a["active_support_sha256"] == b["active_support_sha256"]:
            raise ValueError(f"{pair_id} category arms reuse the same support asset")
        used_classes.update([int(a["active_class_id"]), int(b["active_class_id"])])
    return {
        "pairs": len(groups),
        "rows": len(rows),
        "unique_images": len(
            {int(row["image_id"]) for row in rows}
        ),
        "used_classes": len(used_classes),
        "used_class_ids": sorted(used_classes),
    }


def _write_support_tsv(
    path: Path,
    *,
    fields: Sequence[str],
    support: Mapping[int, Mapping[str, Any]],
    used_class_ids: Sequence[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t")
        writer.writeheader()
        for class_id in sorted(int(value) for value in used_class_ids):
            row = dict(support[class_id]["tsv_row"])
            # Bind the exact effective image path.  Clearing emb_rel_path keeps
            # PatchEpisodeDataset from silently preferring a different mirror
            # when DATA_ROOT changes at evaluation time.
            row["path"] = support[class_id]["effective_path"]
            if "emb_rel_path" in row:
                row["emb_rel_path"] = ""
            writer.writerow(row)


def verify(
    *,
    output: Path,
    output_support_tsv: Path,
    audit_path: Path,
    require_canonical: bool = False,
) -> Dict[str, Any]:
    output = Path(output).expanduser().resolve(strict=True)
    output_support_tsv = Path(output_support_tsv).expanduser().resolve(strict=True)
    audit_path = Path(audit_path).expanduser().resolve(strict=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping) or audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("category intervention audit schema mismatch")
    if audit.get("contract") != CONTRACT:
        raise ValueError("category intervention audit contract mismatch")
    if audit.get("evidence_status") != "runtime_inputs_built_no_model_results":
        raise ValueError("category intervention evidence status mismatch")
    seed = int(audit.get("seed", -1))
    max_pairs = int(audit.get("max_pairs", -1))
    max_cross_iou = float(audit.get("max_cross_iou", math.nan))
    if seed < 0 or max_pairs <= 0 or not math.isfinite(max_cross_iou):
        raise ValueError("category intervention build scalars are invalid")

    inputs = audit.get("inputs")
    outputs = audit.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("category intervention audit bindings are incomplete")
    input_paths: Dict[str, Path] = {}
    for key in ("source", "canonical_map", "support_tsv"):
        record = inputs.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"category intervention input {key} is missing")
        path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        if _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"category intervention input {key} SHA-256 mismatch")
        input_paths[key] = path
    declared_support_root = inputs.get("support_image_root")
    support_image_root = (
        Path(str(declared_support_root.get("path", ""))).expanduser().resolve(
            strict=False
        )
        if isinstance(declared_support_root, Mapping)
        else DEFAULT_SUPPORT_IMAGE_ROOT.resolve(strict=True)
    )

    for key, path in (("episodes", output), ("support_tsv", output_support_tsv)):
        record = outputs.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"category intervention output {key} is missing")
        if Path(str(record.get("path", ""))).expanduser().resolve(strict=True) != path:
            raise ValueError(f"category intervention output {key} path mismatch")
        if _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"{key} SHA-256 mismatch")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"{key} size mismatch")

    if require_canonical:
        expected_paths = {
            "source": DEFAULT_SOURCE.resolve(strict=True),
            "canonical_map": DEFAULT_CANONICAL_MAP.resolve(strict=True),
            "support_tsv": DEFAULT_SUPPORT_TSV.resolve(strict=True),
        }
        if input_paths != expected_paths:
            raise ValueError("formal category intervention input paths drifted")
        if support_image_root != DEFAULT_SUPPORT_IMAGE_ROOT.resolve(strict=True):
            raise ValueError("formal category support image root drifted")
        if (
            output != DEFAULT_OUTPUT.resolve(strict=True)
            or output_support_tsv != DEFAULT_OUTPUT_SUPPORT_TSV.resolve(strict=True)
            or audit_path != DEFAULT_AUDIT.resolve(strict=True)
            or seed != DEFAULT_SEED
            or max_pairs != DEFAULT_MAX_PAIRS
            or max_cross_iou != DEFAULT_MAX_CROSS_IOU
        ):
            raise ValueError("formal category intervention contract is not canonical")

    canonical_by_id, by_name = _load_canonical_map(input_paths["canonical_map"])
    fields, support = _load_support_choices(
        input_paths["support_tsv"],
        by_name=by_name,
        support_image_root=support_image_root,
        seed=seed,
    )
    candidates = _pair_candidates(
        input_paths["source"],
        canonical_by_id=canonical_by_id,
        support=support,
        seed=seed,
        max_cross_iou=max_cross_iou,
    )[:max_pairs]
    expected_rows = [
        _arm_row(
            pair,
            arm=arm,
            canonical_by_id=canonical_by_id,
            support=support,
        )
        for pair in candidates
        for arm in ("A", "B")
    ]
    rows = [row for _line, row in _iter_jsonl(output)]
    if rows != expected_rows:
        raise ValueError("category intervention rows differ from full regeneration")
    summary = _validate_rows(rows)
    if summary != audit.get("summary"):
        raise ValueError("category intervention audit summary drifted")

    with output_support_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if list(reader.fieldnames or []) != fields:
            raise ValueError("category support TSV columns drifted")
        observed_support_rows = list(reader)
    expected_support_rows = []
    for class_id in summary["used_class_ids"]:
        row = dict(support[int(class_id)]["tsv_row"])
        row["path"] = support[int(class_id)]["effective_path"]
        if "emb_rel_path" in row:
            row["emb_rel_path"] = ""
        expected_support_rows.append(row)
    if observed_support_rows != expected_support_rows:
        raise ValueError(
            "category support TSV differs from the deterministic class selection"
        )
    return summary


def build(args: argparse.Namespace) -> Dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    canonical_map = Path(args.canonical_map).expanduser().resolve()
    support_tsv = Path(args.support_tsv).expanduser().resolve()
    support_image_root = Path(args.support_image_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output_support_tsv = Path(args.output_support_tsv).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    canonical_by_id, by_name = _load_canonical_map(canonical_map)
    fields, support = _load_support_choices(
        support_tsv,
        by_name=by_name,
        support_image_root=support_image_root,
        seed=int(args.seed),
    )
    candidates = _pair_candidates(
        source,
        canonical_by_id=canonical_by_id,
        support=support,
        seed=int(args.seed),
        max_cross_iou=float(args.max_cross_iou),
    )
    if int(args.max_pairs) > 0:
        candidates = candidates[: int(args.max_pairs)]
    if not candidates:
        raise ValueError("no eligible category intervention pairs were found")
    rows = [
        _arm_row(pair, arm=arm, canonical_by_id=canonical_by_id, support=support)
        for pair in candidates
        for arm in ("A", "B")
    ]
    summary = _validate_rows(rows)
    _write_jsonl(output, rows)
    _write_support_tsv(
        output_support_tsv,
        fields=fields,
        support=support,
        used_class_ids=summary["used_class_ids"],
    )
    audit = {
        "schema": AUDIT_SCHEMA,
        "contract": dict(CONTRACT),
        "seed": int(args.seed),
        "max_pairs": int(args.max_pairs),
        "max_cross_iou": float(args.max_cross_iou),
        "inputs": {
            "source": {"path": str(source), "sha256": _sha256_file(source)},
            "canonical_map": {
                "path": str(canonical_map),
                "sha256": _sha256_file(canonical_map),
            },
            "support_tsv": {
                "path": str(support_tsv),
                "sha256": _sha256_file(support_tsv),
            },
            "support_image_root": {"path": str(support_image_root)},
        },
        "outputs": {
            "episodes": {
                "path": str(output),
                "sha256": _sha256_file(output),
                "size_bytes": output.stat().st_size,
            },
            "support_tsv": {
                "path": str(output_support_tsv),
                "sha256": _sha256_file(output_support_tsv),
                "size_bytes": output_support_tsv.stat().st_size,
            },
        },
        "summary": summary,
        "evidence_status": "runtime_inputs_built_no_model_results",
    }
    _write_json(audit_path, audit)
    verify(
        output=output,
        output_support_tsv=output_support_tsv,
        audit_path=audit_path,
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
    )
    parser.add_argument(
        "--canonical-map",
        default=str(DEFAULT_CANONICAL_MAP),
    )
    parser.add_argument(
        "--support-tsv",
        default=str(DEFAULT_SUPPORT_TSV),
    )
    parser.add_argument(
        "--support-image-root", default=str(DEFAULT_SUPPORT_IMAGE_ROOT)
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--output-support-tsv", default=str(DEFAULT_OUTPUT_SUPPORT_TSV)
    )
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument("--max-cross-iou", type=float, default=DEFAULT_MAX_CROSS_IOU)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        summary = verify(
            output=Path(args.output).expanduser().resolve(),
            output_support_tsv=Path(args.output_support_tsv).expanduser().resolve(),
            audit_path=Path(args.audit).expanduser().resolve(),
        )
        print(json.dumps({"verified": True, **summary}, sort_keys=True))
        return
    audit = build(args)
    print(json.dumps(audit["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
