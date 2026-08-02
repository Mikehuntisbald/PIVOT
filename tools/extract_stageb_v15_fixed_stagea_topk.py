#!/usr/bin/env python3
"""Extract the exact fixed-Stage-A patch-logit Top-K candidate set.

The command is intentionally narrower than an image-global semantic verifier.
It freezes one checkpoint, canonical caption, query transform, and support patch
per source row.  Completed rows are written as resume-safe shards; only a full
ordered collection is promoted to the extraction JSONL and completed audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
from util.path_compat import default_data_root, remap_legacy_path  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402
from util.stageb_exact_topk_contract import (  # noqa: E402
    EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA,
    EXACT_TOPK_EXTRACTION_SCHEMA,
    EXACT_TOPK_PROTOCOL,
    ExactTopKContractError,
    canonical_sha256,
    file_record,
    normalize_exact_contract,
    sha256_file,
    validate_extraction_candidates,
)


CANDIDATE_SELECTION_SCHEMA = (
    "stage-b-v15-fixed-stagea-topk-candidate-selection-contract-v1"
)
QUERY_TRANSFORM_SCHEMA = "stage-b-v15-fixed-stagea-query-transform-contract-v1"
SUPPORT_TRANSFORM_SCHEMA = "stage-b-v15-fixed-stagea-support-transform-contract-v1"
PROGRESS_SCHEMA = "stage-b-v15-fixed-stagea-topk-extraction-progress-v1"
PLAN_SCHEMA = "stage-b-v15-fixed-stagea-topk-extraction-plan-v1"
EXPECTED_DECODER_QUERIES = 900
DEFAULT_TOPK = 50
DEFAULT_BOX_ATOL = 1.0e-5


class ExtractionError(RuntimeError):
    pass


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


def _read_json(path: Path, *, label: str) -> Any:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ExtractionError(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExtractionError(f"invalid {label} {path}: {error}") from error


def _iter_jsonl(path: Path, *, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ExtractionError(f"missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ExtractionError(f"blank row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExtractionError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ExtractionError(f"non-object row at {path}:{line_number}")
            yield line_number, row


def _hashed_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def _expand_path(value: Any, *, data_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionError(f"invalid path value: {value!r}")
    expanded = value
    defaults = {
        "DATA_ROOT": str(data_root),
        "T9_ROOT": str(data_root.parent),
        "GDINO_ROOT": str(data_root.parent / "gdino"),
        "MEDIA_USER": os.environ.get("MEDIA_USER", "haoyi"),
    }
    for key, replacement in defaults.items():
        expanded = expanded.replace(f"${{{key}}}", replacement)
        expanded = expanded.replace(f"${key}", replacement)
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    return remap_legacy_path(expanded, data_root=data_root).expanduser().resolve()


def _clean_name(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").replace(".", " ").split())


def _canonical_maps(path: Path) -> tuple[dict[int, str], dict[str, int]]:
    raw = _read_json(path, label="canonical classes")
    if not isinstance(raw, list):
        raise ExtractionError("canonical classes must be a JSON list")
    preferred: dict[int, str] = {}
    by_name: dict[str, int] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or item.get("id") is None:
            raise ExtractionError(f"invalid canonical class entry {index}")
        try:
            class_id = int(item["id"])
        except (TypeError, ValueError) as error:
            raise ExtractionError(f"invalid canonical id at entry {index}") from error
        names: list[str] = []
        for key in ("base_name", "raw_name", "norm_name", "synset"):
            cleaned = _clean_name(item.get(key))
            if cleaned:
                names.append(cleaned)
        for value in item.get("synonyms", []) or []:
            cleaned = _clean_name(value)
            if cleaned:
                names.append(cleaned)
        for value in item.get("aliases", []) or []:
            if isinstance(value, Mapping):
                for key in ("name", "norm_name"):
                    cleaned = _clean_name(value.get(key))
                    if cleaned:
                        names.append(cleaned)
            else:
                cleaned = _clean_name(value)
                if cleaned:
                    names.append(cleaned)
        if not names:
            raise ExtractionError(f"canonical class {class_id} has no usable name")
        if class_id in preferred:
            raise ExtractionError(f"duplicate canonical class id: {class_id}")
        preferred[class_id] = names[0]
        for name in names:
            by_name.setdefault(name.casefold(), class_id)
    return preferred, by_name


def _select_dataset_entry(
    data_config: Path,
    source_pairs: Path,
    *,
    data_root: Path,
    requested_index: int | None,
) -> tuple[int, dict[str, Any]]:
    raw = _read_json(data_config, label="data config")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("train"), list):
        raise ExtractionError("data config must contain a train list")
    entries = raw["train"]
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(entries):
            raise ExtractionError("dataset entry index is outside the train list")
        candidates = [(requested_index, entries[requested_index])]
    else:
        candidates = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or not entry.get("anno"):
                continue
            if _expand_path(entry["anno"], data_root=data_root) == source_pairs:
                candidates.append((index, entry))
        if len(candidates) != 1:
            raise ExtractionError(
                "source pairs must match exactly one data-config train entry; "
                "use --dataset-entry-index only to disambiguate identical entries"
            )
    index, value = candidates[0]
    if not isinstance(value, Mapping):
        raise ExtractionError(f"data-config train entry {index} is not an object")
    entry = dict(value)
    if _expand_path(entry.get("anno"), data_root=data_root) != source_pairs:
        raise ExtractionError("selected data-config entry does not bind source pairs")
    if bool(entry.get("support_patch_use_embedding", False)):
        raise ExtractionError("exact visual support extraction requires image patches")
    return index, entry


def _resolve_support_path(
    parts: Sequence[str],
    columns: Mapping[str, int],
    *,
    tsv_parent: Path,
    image_root: Path | None,
) -> Path | None:
    emb_index = columns.get("emb_rel_path")
    if image_root is not None and emb_index is not None and emb_index < len(parts):
        relative = parts[emb_index].strip()
        if relative:
            candidate = image_root / relative
            if candidate.suffix == ".npy":
                candidate = candidate.with_suffix(".jpg")
            if candidate.is_file():
                return candidate.resolve()
    path_index = columns.get("path")
    if path_index is None or path_index >= len(parts) or not parts[path_index].strip():
        return None
    candidate = Path(parts[path_index].strip()).expanduser()
    if not candidate.is_absolute():
        candidate = tsv_parent / candidate
    return candidate.resolve() if candidate.is_file() else None


def load_support_bank(
    entry: Mapping[str, Any],
    *,
    data_root: Path,
    canonical_name_to_id: Mapping[str, int],
) -> tuple[dict[int, list[Path]], dict[str, Any]]:
    tsv = _expand_path(entry.get("support_patch_tsv"), data_root=data_root)
    if not tsv.is_file():
        raise ExtractionError(f"support patch TSV is missing: {tsv}")
    image_root = None
    if entry.get("support_patch_image_root"):
        image_root = _expand_path(entry["support_patch_image_root"], data_root=data_root)
    wanted_bucket = entry.get("support_patch_bucket")
    bank: dict[int, set[Path]] = defaultdict(set)
    with tsv.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        columns = {name: index for index, name in enumerate(header)}
        class_index = next(
            (
                columns[name]
                for name in ("class_id", "canonical_class_id", "support_class", "class")
                if name in columns
            ),
            None,
        )
        if class_index is None:
            raise ExtractionError("support TSV has no canonical class column")
        bucket_index = columns.get("bucket")
        for line_number, line in enumerate(handle, 2):
            parts = line.rstrip("\n").split("\t")
            if class_index >= len(parts):
                raise ExtractionError(f"short support TSV row at {tsv}:{line_number}")
            if wanted_bucket is not None and bucket_index is not None:
                if bucket_index >= len(parts) or parts[bucket_index] != wanted_bucket:
                    continue
            class_value = parts[class_index].strip()
            try:
                class_id = int(class_value)
            except ValueError:
                class_id = canonical_name_to_id.get(_clean_name(class_value).casefold(), -1)
            if class_id < 0:
                continue
            path = _resolve_support_path(
                parts,
                columns,
                tsv_parent=tsv.parent,
                image_root=image_root,
            )
            if path is not None:
                bank[class_id].add(path)
    result = {class_id: sorted(paths, key=lambda path: str(path)) for class_id, paths in bank.items()}
    if not result:
        raise ExtractionError("support patch TSV resolved to an empty image bank")
    return result, file_record(tsv)


def choose_fixed_support(
    sample_id: str,
    class_id: int,
    candidates: Sequence[Path],
) -> Path:
    if not candidates:
        raise ExtractionError(f"class {class_id} has no usable support patches")
    ordered = sorted({path.expanduser().resolve() for path in candidates}, key=str)
    key = canonical_sha256(
        {
            "schema": "stage-b-v15-fixed-support-selection-key-v1",
            "sample_id": sample_id,
            "class_id": int(class_id),
        }
    )
    return ordered[int(key, 16) % len(ordered)]


def query_transform_contract(
    query_cfg_path: Path,
    *,
    query_cfg: Any,
) -> dict[str, Any]:
    if bool(getattr(query_cfg, "fix_size", False)) is not True:
        raise ExtractionError("exact query transform requires fix_size=true")
    if not hasattr(query_cfg, "data_aug_hflip_prob"):
        raise ExtractionError(
            "query-transform config must explicitly set data_aug_hflip_prob=0.0"
        )
    if float(query_cfg.data_aug_hflip_prob) != 0.0:
        raise ExtractionError("exact query transform requires data_aug_hflip_prob=0.0")
    scales = [int(value) for value in getattr(query_cfg, "data_aug_scales", [])]
    max_size = int(getattr(query_cfg, "data_aug_max_size", 0))
    if max(scales or [0]) != 800 or max_size != 1333:
        raise ExtractionError("exact fixed query transform requires size 1333x800")
    return _hashed_contract(
        {
            "schema": QUERY_TRANSFORM_SCHEMA,
            "deterministic": True,
            "image_set": "train",
            "fix_size": True,
            "hflip_probability": 0.0,
            "resize_wh": [1333, 800],
            "resize_implementation": "datasets.transforms.resize(tuple_wh)",
            "interpolation": "torchvision_pil_bilinear",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "canonical_caption_policy": "preferred_base_raw_norm_synset_then_space_dot",
            "query_transform_config": file_record(query_cfg_path),
        }
    )


def support_transform_contract(
    *,
    support_tsv: Mapping[str, Any],
    entry_index: int,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    return _hashed_contract(
        {
            "schema": SUPPORT_TRANSFORM_SCHEMA,
            "resize_short_side": 256,
            "center_crop_hw": [224, 224],
            "interpolation": "torchvision_resize_default_bilinear",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "fixed_support_patch_per_row": True,
            "support_selection": (
                "sha256(sample_id,class_id)-mod-lexicographically-sorted-"
                "existing-unique-class-paths"
            ),
            "dataset_entry_index": int(entry_index),
            "support_bucket": entry.get("support_patch_bucket"),
            "support_patch_tsv": dict(support_tsv),
        }
    )


def candidate_selection_contract(
    *, topk: int, box_atol: float, amp: bool
) -> dict[str, Any]:
    return _hashed_contract(
        {
            "schema": CANDIDATE_SELECTION_SCHEMA,
            "candidate_topk": int(topk),
            "score_source": "score_patch_logits",
            "selection": "torch.topk(largest=true,sorted=true)",
            "candidate_order": "descending_patch_logit",
            "candidate_box_space": "normalized_cxcywh",
            "fixed_support_patch_per_row": True,
            "deterministic_query_transform": True,
            "dynamic_candidate_replay_must_match": True,
            "candidate_box_atol": float(box_atol),
            "expected_decoder_queries": EXPECTED_DECODER_QUERIES,
            "single_patch_slot": True,
            "forward_amp": bool(amp),
            "patch_logits_are_raw_not_sigmoid": True,
        }
    )


def _image_path(row: Mapping[str, Any], *, image_root: Path) -> Path:
    value = row.get("image_path")
    if isinstance(value, str) and value.strip():
        path = remap_legacy_path(value).expanduser().resolve()
    else:
        file_name = row.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ExtractionError("source row has neither image_path nor file_name")
        path = (image_root / file_name).resolve()
    if not path.is_file():
        raise ExtractionError(f"source image is missing: {path}")
    return path


def _validate_source_pair(row: Mapping[str, Any], *, context: str) -> tuple[str, int]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ExtractionError(f"{context} has no sample_id")
    if str(row.get("split", "")).strip().lower() != "train":
        raise ExtractionError(f"{context} is not a train row")
    positive = str(row.get("sent", "")).strip()
    negative = str(row.get("try_tn", "")).strip()
    if not positive or not negative or positive.casefold() == negative.casefold():
        raise ExtractionError(f"{context} has invalid positive/TN expressions")
    try:
        class_id = int(row["class_id"])
        bbox = [float(value) for value in row["target_bbox_used"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ExtractionError(f"{context} has invalid class/bbox") from error
    if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise ExtractionError(f"{context} has invalid target_bbox_used")
    return sample_id.strip(), class_id


def _config_dependency_records(path: Path) -> list[dict[str, Any]]:
    """Record the small Python `_base_` closure without changing the contract key."""
    seen: set[Path] = set()
    records: list[dict[str, Any]] = []

    def visit(current: Path) -> None:
        current = current.expanduser().resolve()
        if current in seen:
            return
        seen.add(current)
        records.append(file_record(current))
        text = current.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("_base_") or "=" not in stripped:
                continue
            raw = stripped.split("=", 1)[1].strip().strip("'\"")
            if raw and not raw.startswith(("[", "(")):
                visit((current.parent / raw).resolve())

    visit(path)
    return records


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model_config = Path(args.model_config).expanduser().resolve()
    query_config = Path(args.query_transform_config).expanduser().resolve()
    data_config = Path(args.data_config).expanduser().resolve()
    source_pairs = Path(args.source_pairs).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    for label, path in (
        ("checkpoint", checkpoint),
        ("model config", model_config),
        ("query transform config", query_config),
        ("data config", data_config),
        ("source pairs", source_pairs),
    ):
        if not path.is_file():
            raise ExtractionError(f"missing {label}: {path}")
    topk = int(args.candidate_topk)
    if topk != DEFAULT_TOPK and not bool(getattr(args, "allow_nonstandard_topk", False)):
        raise ExtractionError("formal D4 extraction requires candidate_topk=50")
    box_atol = float(args.candidate_box_atol)
    if not math.isfinite(box_atol) or box_atol != DEFAULT_BOX_ATOL:
        raise ExtractionError("formal D4 extraction requires candidate_box_atol=1e-5")

    entry_index, entry = _select_dataset_entry(
        data_config,
        source_pairs,
        data_root=data_root,
        requested_index=args.dataset_entry_index,
    )
    canonical_path = _expand_path(entry.get("canonical_classes_json"), data_root=data_root)
    if args.canonical_classes is not None:
        requested = Path(args.canonical_classes).expanduser().resolve()
        if requested != canonical_path:
            raise ExtractionError("canonical classes do not match the selected data-config entry")
    canonical_names, canonical_name_to_id = _canonical_maps(canonical_path)
    support_bank, support_tsv_record = load_support_bank(
        entry,
        data_root=data_root,
        canonical_name_to_id=canonical_name_to_id,
    )
    image_root_value = entry.get("sam3_tn_image_root", entry.get("root"))
    image_root = _expand_path(image_root_value, data_root=data_root)
    query_cfg = SLConfig.fromfile(str(query_config))
    query_contract = query_transform_contract(query_config, query_cfg=query_cfg)
    support_contract = support_transform_contract(
        support_tsv=support_tsv_record,
        entry_index=entry_index,
        entry=entry,
    )
    selection_contract = candidate_selection_contract(
        topk=topk,
        box_atol=box_atol,
        amp=bool(args.amp),
    )
    exact_contract = normalize_exact_contract(
        {
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_config_sha256": sha256_file(model_config),
            "data_config_sha256": sha256_file(data_config),
            "canonical_classes_sha256": sha256_file(canonical_path),
            "query_transform_contract_sha256": query_contract["sha256"],
            "support_transform_contract_sha256": support_contract["sha256"],
            "candidate_selection_contract_sha256": selection_contract["sha256"],
            "candidate_topk": topk,
            "candidate_box_atol": box_atol,
        }
    )

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    image_records: dict[Path, dict[str, Any]] = {}
    support_records: dict[Path, dict[str, Any]] = {}
    for line_number, raw in _iter_jsonl(source_pairs, label="source pairs"):
        sample_id, class_id = _validate_source_pair(
            raw, context=f"source row {line_number}"
        )
        if sample_id in seen:
            raise ExtractionError(f"duplicate source sample_id: {sample_id}")
        seen.add(sample_id)
        canonical_name = canonical_names.get(class_id)
        if canonical_name is None:
            raise ExtractionError(f"source class {class_id} is absent from canonical classes")
        image_path = _image_path(raw, image_root=image_root)
        support_path = choose_fixed_support(
            sample_id, class_id, support_bank.get(class_id, [])
        )
        image_record_value = image_records.setdefault(image_path, file_record(image_path))
        support_record_value = support_records.setdefault(
            support_path, file_record(support_path)
        )
        source_pair = dict(raw)
        source_pair["sample_id"] = sample_id
        source_pair["image_path"] = str(image_path)
        fixed_support = {
            "path": str(support_path),
            "sha256": support_record_value["sha256"],
            "class_id": class_id,
            "transform_contract_sha256": support_contract["sha256"],
        }
        rows.append(
            {
                "ordinal": len(rows),
                "source_line": line_number,
                "sample_id": sample_id,
                "class_id": class_id,
                "canonical_caption": f"{canonical_name} .",
                "source_pair": source_pair,
                "source_pair_sha256": canonical_sha256(source_pair),
                "image": image_record_value,
                "fixed_support_patch": fixed_support,
            }
        )
    if not rows:
        raise ExtractionError("source pairs are empty")
    provenance = {
        "checkpoint": file_record(checkpoint),
        "model_config": file_record(model_config),
        "data_config": file_record(data_config),
        "canonical_classes": file_record(canonical_path),
        "source_pairs": file_record(source_pairs, rows=len(rows)),
        "support_patch_tsv": support_tsv_record,
        "model_config_dependencies": _config_dependency_records(model_config),
        "query_transform_config_dependencies": _config_dependency_records(query_config),
    }
    plan_binding = {
        "schema": PLAN_SCHEMA,
        "exact_contract": exact_contract,
        "provenance": provenance,
        "row_bindings": [
            {
                "ordinal": row["ordinal"],
                "sample_id": row["sample_id"],
                "source_pair_sha256": row["source_pair_sha256"],
                "image_sha256": row["image"]["sha256"],
                "support_sha256": row["fixed_support_patch"]["sha256"],
                "canonical_caption": row["canonical_caption"],
            }
            for row in rows
        ],
    }
    return {
        **plan_binding,
        "plan_sha256": canonical_sha256(plan_binding),
        "rows": rows,
        "query_transform_contract": query_contract,
        "support_transform_contract": support_contract,
        "candidate_selection_contract": selection_contract,
        "paths": {
            "checkpoint": checkpoint,
            "model_config": model_config,
            "query_config": query_config,
            "data_config": data_config,
            "source_pairs": source_pairs,
        },
    }


def apply_query_transform(image: Image.Image) -> tuple[torch.Tensor, dict[str, Any]]:
    import datasets.transforms as transforms

    image = image.convert("RGB")
    original_w, original_h = image.size
    transformed, _ = transforms.resize(image, {}, (1333, 800), None)
    resized_w, resized_h = transformed.size
    tensor, _ = transforms.ToTensor()(transformed, {})
    tensor, _ = transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )(tensor, {})
    trace = {
        "schema": "stage-b-v15-fixed-stagea-query-transform-trace-v1",
        "original_hw": [int(original_h), int(original_w)],
        "output_hw": [int(resized_h), int(resized_w)],
        "scale_xy": [
            float(resized_w) / float(original_w),
            float(resized_h) / float(original_h),
        ],
        "offset_xy": [0.0, 0.0],
        "operations": [
            {
                "op": "fixed_resize",
                "requested_wh": [1333, 800],
                "before_hw": [int(original_h), int(original_w)],
                "after_hw": [int(resized_h), int(resized_w)],
            }
        ],
    }
    return tensor, trace


def apply_support_transform(image: Image.Image) -> torch.Tensor:
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transform(image.convert("RGB"))


def make_candidates(
    patch_logits: torch.Tensor,
    boxes: torch.Tensor,
    *,
    topk: int,
) -> tuple[list[dict[str, Any]], str]:
    scores = torch.as_tensor(patch_logits).detach().to(dtype=torch.float32, device="cpu")
    candidate_boxes = torch.as_tensor(boxes).detach().to(dtype=torch.float32, device="cpu")
    if scores.dim() == 2 and scores.shape[-1] == 1:
        scores = scores[:, 0]
    if scores.dim() != 1 or candidate_boxes.shape != (scores.numel(), 4):
        raise ExtractionError(
            f"patch score/box shape mismatch: {tuple(scores.shape)} / {tuple(candidate_boxes.shape)}"
        )
    if int(scores.numel()) != EXPECTED_DECODER_QUERIES:
        raise ExtractionError(
            f"expected {EXPECTED_DECODER_QUERIES} decoder queries, got {scores.numel()}"
        )
    if topk <= 0 or topk > scores.numel():
        raise ExtractionError("candidate topk is outside the query count")
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(candidate_boxes).all()):
        raise ExtractionError("model output contains non-finite values")
    if bool((candidate_boxes < 0.0).any()) or bool((candidate_boxes > 1.0).any()):
        raise ExtractionError("candidate boxes are outside normalized [0,1]")
    if bool((candidate_boxes[:, 2:] <= 0.0).any()):
        raise ExtractionError("candidate boxes have non-positive width/height")
    values, indices = torch.topk(scores, k=int(topk), largest=True, sorted=True)
    payloads: list[dict[str, Any]] = []
    for rank, (query_index, patch_logit) in enumerate(zip(indices.tolist(), values.tolist())):
        payload = {
            "rank": rank,
            "query_index": int(query_index),
            "bbox_cxcywh_normalized": [
                float(value) for value in candidate_boxes[int(query_index)].tolist()
            ],
            "patch_logit": float(patch_logit),
        }
        payloads.append({**payload, "candidate_sha256": canonical_sha256(payload)})
    candidate_set_sha = canonical_sha256(
        [{key: value for key, value in row.items() if key != "candidate_sha256"} for row in payloads]
    )
    validate_extraction_candidates(
        payloads,
        candidate_topk=topk,
        candidate_set_sha256=candidate_set_sha,
    )
    return payloads, candidate_set_sha


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_error:
        if "Weights only load failed" not in str(first_error):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def load_stagea_model(
    plan: Mapping[str, Any], device: torch.device, *, allow_state_key_drift: bool
) -> torch.nn.Module:
    cfg = SLConfig.fromfile(str(plan["paths"]["model_config"]))
    cfg.device = str(device)
    cfg.patch_only = True
    cfg.use_coco_eval = False
    build = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build is None:
        raise ExtractionError(f"unknown modelname={cfg.modelname!r}")
    model, _criterion, _postprocessors = build(cfg)
    checkpoint = _torch_load(plan["paths"]["checkpoint"])
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping):
        raise ExtractionError("checkpoint does not contain a state dictionary")
    missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=False)
    if (missing or unexpected) and not allow_state_key_drift:
        raise ExtractionError(
            f"checkpoint/model key drift: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.to(device).eval()
    return model


def _shard_path(shards: Path, row: Mapping[str, Any]) -> Path:
    suffix = canonical_sha256(str(row["sample_id"]))[:16]
    return shards / f"{int(row['ordinal']):08d}-{suffix}.json"


def _validate_shard(row: Mapping[str, Any], planned: Mapping[str, Any], exact_contract: Mapping[str, Any]) -> None:
    if row.get("schema") != EXACT_TOPK_EXTRACTION_SCHEMA or row.get("protocol") != EXACT_TOPK_PROTOCOL:
        raise ExtractionError("resume shard schema/protocol drifted")
    if row.get("exact_contract") != exact_contract:
        raise ExtractionError("resume shard exact contract drifted")
    for key in ("sample_id", "source_pair", "source_pair_sha256", "image", "fixed_support_patch"):
        if row.get(key) != planned.get(key):
            raise ExtractionError(f"resume shard {key} drifted")
    if row.get("query_transform_trace_sha256") != canonical_sha256(row.get("query_transform_trace")):
        raise ExtractionError("resume shard transform trace hash drifted")
    try:
        validate_extraction_candidates(
            row.get("candidates"),
            candidate_topk=int(exact_contract["candidate_topk"]),
            candidate_set_sha256=row.get("candidate_set_sha256"),
        )
    except ExactTopKContractError as error:
        raise ExtractionError(str(error)) from error


@torch.inference_mode()
def _forward_batch(
    model: torch.nn.Module,
    batch: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    amp: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    images: list[torch.Tensor] = []
    patches: list[torch.Tensor] = []
    traces: list[dict[str, Any]] = []
    captions: list[str] = []
    for row in batch:
        with Image.open(row["image"]["path"]) as image:
            image_tensor, trace = apply_query_transform(image)
        with Image.open(row["fixed_support_patch"]["path"]) as support:
            support_tensor = apply_support_transform(support)
        images.append(image_tensor)
        patches.append(support_tensor)
        traces.append(trace)
        captions.append(str(row["canonical_caption"]))
    samples = nested_tensor_from_tensor_list(images).to(device)
    patch_batch = torch.stack(patches, dim=0).to(device, non_blocking=True)
    enabled = bool(amp) and device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=enabled, dtype=torch.float16 if enabled else None):
        outputs = model(
            samples,
            captions=captions,
            patches=patch_batch,
            patch_only=True,
            disable_patch_dn=True,
            patch_only_compute_text_logits=False,
        )
    logits = outputs.get("pred_logits_patch")
    boxes = outputs.get("pred_boxes")
    if not torch.is_tensor(logits) or not torch.is_tensor(boxes):
        raise ExtractionError("Stage-A output lacks pred_logits_patch/pred_boxes")
    if logits.dim() == 3:
        if logits.shape[-1] != 1:
            raise ExtractionError("exact extraction requires one patch slot")
        logits = logits[..., 0]
    if logits.shape[:2] != boxes.shape[:2] or logits.shape[0] != len(batch):
        raise ExtractionError("Stage-A output batch/query dimensions drifted")
    return [(logits[index], boxes[index], traces[index]) for index in range(len(batch))]


def _make_extraction_row(
    planned: Mapping[str, Any],
    *,
    exact_contract: Mapping[str, Any],
    logits: torch.Tensor,
    boxes: torch.Tensor,
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    candidates, candidate_set_sha = make_candidates(
        logits,
        boxes,
        topk=int(exact_contract["candidate_topk"]),
    )
    return {
        "schema": EXACT_TOPK_EXTRACTION_SCHEMA,
        "protocol": EXACT_TOPK_PROTOCOL,
        "sample_id": planned["sample_id"],
        "exact_contract": dict(exact_contract),
        "source_pair": dict(planned["source_pair"]),
        "source_pair_sha256": planned["source_pair_sha256"],
        "image": dict(planned["image"]),
        "fixed_support_patch": dict(planned["fixed_support_patch"]),
        "canonical_stage_a_caption": planned["canonical_caption"],
        "query_transform_trace": dict(trace),
        "query_transform_trace_sha256": canonical_sha256(trace),
        "candidates": candidates,
        "candidate_set_sha256": candidate_set_sha,
    }


def _audit_payload(plan: Mapping[str, Any], output: Path) -> dict[str, Any]:
    rows = len(plan["rows"])
    return {
        "schema": EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA,
        "protocol": EXACT_TOPK_PROTOCOL,
        "complete": True,
        "rows": rows,
        "exact_contract": dict(plan["exact_contract"]),
        "extractions": file_record(output, rows=rows),
        "provenance": dict(plan["provenance"]),
        "query_transform_contract": dict(plan["query_transform_contract"]),
        "support_transform_contract": dict(plan["support_transform_contract"]),
        "candidate_selection_contract": dict(plan["candidate_selection_contract"]),
        "plan_sha256": plan["plan_sha256"],
        "claims": {
            "fixed_stagea_topk_candidates_extracted": True,
            "all_stagea_topk_candidates_reviewed": False,
            "all_stagea_queries_verified": False,
            "image_global_semantic_absence_proven": False,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "amp": bool(plan["candidate_selection_contract"]["forward_amp"]),
        },
    }


def extraction_summary(plan: Mapping[str, Any], *, kind: str, list_rows: bool) -> dict[str, Any]:
    result = {
        "schema": PLAN_SCHEMA,
        "kind": kind,
        "rows": len(plan["rows"]),
        "plan_sha256": plan["plan_sha256"],
        "exact_contract": plan["exact_contract"],
        "provenance": plan["provenance"],
        "query_transform_contract": plan["query_transform_contract"],
        "support_transform_contract": plan["support_transform_contract"],
        "candidate_selection_contract": plan["candidate_selection_contract"],
        "model_or_gpu_loaded": False,
    }
    if list_rows:
        result["planned_rows"] = [
            {
                "ordinal": row["ordinal"],
                "sample_id": row["sample_id"],
                "image": row["image"],
                "fixed_support_patch": row["fixed_support_patch"],
                "canonical_stage_a_caption": row["canonical_caption"],
            }
            for row in plan["rows"]
        ]
    return result


def extract(args: argparse.Namespace) -> dict[str, Any]:
    plan = prepare_plan(args)
    output = Path(args.output).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    shards = work_dir / "rows"
    progress_path = work_dir / "progress.json"
    if output.exists() or audit_path.exists():
        if output.is_file() and audit_path.is_file():
            return verify_completed(args, prepared=plan)
        raise ExtractionError("partial finalized extraction/audit exists; refuse overwrite")
    shards.mkdir(parents=True, exist_ok=True)
    expected_shards = {_shard_path(shards, row) for row in plan["rows"]}
    orphan = set(shards.glob("*.json")).difference(expected_shards)
    if orphan:
        raise ExtractionError(f"found {len(orphan)} orphan resume shards")

    completed: dict[int, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for planned in plan["rows"]:
        shard = _shard_path(shards, planned)
        if shard.is_file():
            value = _read_json(shard, label="resume shard")
            if not isinstance(value, Mapping):
                raise ExtractionError(f"resume shard is not an object: {shard}")
            _validate_shard(value, planned, plan["exact_contract"])
            completed[int(planned["ordinal"])] = dict(value)
        else:
            pending.append(planned)
    _atomic_json(
        progress_path,
        {
            "schema": PROGRESS_SCHEMA,
            "complete": False,
            "plan_sha256": plan["plan_sha256"],
            "expected_rows": len(plan["rows"]),
            "completed_rows": len(completed),
        },
    )
    if pending:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ExtractionError("CUDA was requested but is unavailable")
        model = load_stagea_model(
            plan,
            device,
            allow_state_key_drift=bool(args.allow_state_key_drift),
        )
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise ExtractionError("batch size must be positive")
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            results = _forward_batch(
                model,
                batch,
                device=device,
                amp=bool(args.amp),
            )
            for planned, (logits, boxes, trace) in zip(batch, results):
                row = _make_extraction_row(
                    planned,
                    exact_contract=plan["exact_contract"],
                    logits=logits,
                    boxes=boxes,
                    trace=trace,
                )
                shard = _shard_path(shards, planned)
                _atomic_json(shard, row)
                completed[int(planned["ordinal"])] = row
            _atomic_json(
                progress_path,
                {
                    "schema": PROGRESS_SCHEMA,
                    "complete": False,
                    "plan_sha256": plan["plan_sha256"],
                    "expected_rows": len(plan["rows"]),
                    "completed_rows": len(completed),
                },
            )
            if int(args.log_every) > 0 and len(completed) % int(args.log_every) < len(batch):
                print(f"[INFO] extracted {len(completed)}/{len(plan['rows'])} rows", flush=True)
    if len(completed) != len(plan["rows"]):
        raise ExtractionError("resume shard collection is incomplete")
    if file_record(plan["paths"]["checkpoint"])["sha256"] != plan["exact_contract"]["checkpoint_sha256"]:
        raise ExtractionError("checkpoint changed during extraction")
    ordered = [completed[index] for index in range(len(plan["rows"]))]
    _atomic_jsonl(output, ordered)
    audit = _audit_payload(plan, output)
    _atomic_json(audit_path, audit)
    _atomic_json(
        progress_path,
        {
            "schema": PROGRESS_SCHEMA,
            "complete": True,
            "plan_sha256": plan["plan_sha256"],
            "expected_rows": len(ordered),
            "completed_rows": len(ordered),
            "extractions": audit["extractions"],
            "audit": file_record(audit_path),
        },
    )
    return audit


def verify_completed(
    args: argparse.Namespace, *, prepared: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    from tools.build_stageb_v15_fixed_stagea_topk_exact_pairs import (
        _validate_extraction_audit,
        _validate_extraction_row,
    )

    plan = dict(prepared) if prepared is not None else prepare_plan(args)
    output = Path(args.output).expanduser().resolve()
    audit_path = Path(args.audit).expanduser().resolve()
    audit, contract = _validate_extraction_audit(audit_path, output)
    if contract != plan["exact_contract"] or audit.get("plan_sha256") != plan["plan_sha256"]:
        raise ExtractionError("completed extraction no longer matches the locked plan")
    if audit.get("provenance") != plan["provenance"]:
        raise ExtractionError("completed extraction provenance drifted")
    if audit.get("query_transform_contract") != plan["query_transform_contract"]:
        raise ExtractionError("query transform contract drifted")
    if audit.get("support_transform_contract") != plan["support_transform_contract"]:
        raise ExtractionError("support transform contract drifted")
    if audit.get("candidate_selection_contract") != plan["candidate_selection_contract"]:
        raise ExtractionError("candidate selection contract drifted")
    planned_by_sample = {row["sample_id"]: row for row in plan["rows"]}
    support_cache: dict[Path, str] = {}
    seen: set[str] = set()
    for line_number, row in _iter_jsonl(output, label="extractions"):
        value = _validate_extraction_row(
            row,
            line_number=line_number,
            exact_contract=contract,
            support_hash_cache=support_cache,
        )
        planned = planned_by_sample.get(value["sample_id"])
        if planned is None:
            raise ExtractionError(f"orphan extraction row: {value['sample_id']}")
        _validate_shard(row, planned, contract)
        if value["sample_id"] in seen:
            raise ExtractionError(f"duplicate extraction row: {value['sample_id']}")
        seen.add(value["sample_id"])
    if seen != set(planned_by_sample):
        raise ExtractionError("completed extraction coverage is incomplete")
    return {
        "schema": EXACT_TOPK_EXTRACTION_AUDIT_SCHEMA,
        "verified": True,
        "rows": len(seen),
        "plan_sha256": plan["plan_sha256"],
        "extractions": file_record(output, rows=len(seen)),
        "audit": file_record(audit_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--query-transform-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--canonical-classes", type=Path)
    parser.add_argument("--dataset-entry-index", type=int)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--candidate-topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--candidate-box-atol", type=float, default=DEFAULT_BOX_ATOL)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-state-key-drift", action="store_true")
    parser.add_argument("--allow-nonstandard-topk", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--log-every", type=int, default=100)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.verify_only:
            result = verify_completed(args)
        elif args.dry_run or args.list:
            plan = prepare_plan(args)
            result = extraction_summary(
                plan,
                kind="list_no_model_or_gpu" if args.list else "dry_run_no_model_or_gpu",
                list_rows=bool(args.list),
            )
        else:
            result = extract(args)
    except (ExtractionError, ExactTopKContractError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
