import json
import os
import random
import time
from array import array
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import csv
import hashlib
import math
import pickle
import re
from collections import OrderedDict
from difflib import SequenceMatcher

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as TV
from transformers import AutoTokenizer
import fcntl

import datasets.transforms as T
from datasets.coco import make_coco_transforms as make_query_transforms
from util.path_compat import remap_legacy_path
from util.stageb_exact_topk_contract import (
    EXACT_TOPK_TN_SCOPE,
    ExactTopKContractError,
    normalize_exact_contract,
    sha256_file,
    validate_exact_pair_collection,
)
from util.stage_b_table_b_contract import (
    TABLE_B_PAIR_SCHEMA,
    TableBConfidenceContract,
    TableBContractError,
    validate_table_b_dataset_binding,
)


def _validate_u2v5_matched_table_b_binding(args, datasetinfo):
    if not bool(getattr(args, "stage_b_u2v5_matched_data", False)):
        return None
    table_id = str(getattr(args, "stage_b_v19_table_b_id", ""))
    scopes = {"D2m": "traceable_counterfactual_edit", "D3m": "proposal_covered_verified"}
    if table_id not in scopes or datasetinfo.get("paper_table_b_id") != table_id:
        raise TableBContractError("U2-v5 matched row ID drifted")
    scope = scopes[table_id]
    if datasetinfo.get("paper_tn_scope") != scope or tuple(
        getattr(args, "stage_b_v19_table_b_scope_allowlist", ())
    ) != (scope,):
        raise TableBContractError("U2-v5 matched scope drifted")
    audit_path = remap_legacy_path(datasetinfo.get("paper_contract_audit")).resolve(strict=True)
    expected_audit = Path(
        "data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2/audit.json"
    ).resolve(strict=True)
    if audit_path != expected_audit or sha256_file(audit_path) != (
        "5ff62a838a5123d580a72e353147b97bb69e9d7967348b55cba4ccb9ca36cb96"
    ):
        raise TableBContractError("U2-v5 matched audit drifted")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "stage-b-paper-c2-parent-matched-tn-v2" or audit.get(
        "invariants", {}
    ).get("strict_union_image_overlap") != 0:
        raise TableBContractError("U2-v5 matched audit invariants failed")
    output = audit.get("outputs", {}).get(f"{table_id.lower()}_train")
    annotation = remap_legacy_path(datasetinfo.get("anno")).resolve(strict=True)
    if not isinstance(output, Mapping) or annotation != Path(str(output.get("path"))).resolve(strict=True) or sha256_file(annotation) != output.get("sha256"):
        raise TableBContractError("U2-v5 matched annotation drifted")
    if datasetinfo.get("paper_runtime_contract") != "v24_parent_matched_class_aligned_v2_fail_closed":
        raise TableBContractError("U2-v5 matched runtime contract drifted")
    return TableBConfidenceContract(
        table_b_id=table_id,
        scope=scope,
        scope_allowlist=(scope,),
        audit_path=audit_path,
        audit_sha256=sha256_file(audit_path),
        train_path=annotation,
        train_sha256=sha256_file(annotation),
        allow_single_edit_token_provenance=False,
    )

_WS_RE = re.compile(r"\s+")
_PUNC_RE = re.compile(r"[^a-z0-9 _-]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TN_CATEGORY_SEP_RE = re.compile(r"[_/,\-]+")
_DATA_DRIVEN_SAME_CATEGORY_RANK_SUPERVISIONS = {
    "primary_vs_same_category_aux_v1",
    "primary_vs_same_category_aux_plus_gap3_coverage_v1",
    "official_same_image_same_category_assignment_v1",
    "role_routed_official_assignment_top1_v1",
    "role_routed_official_assignment_all_exclusive_nonowned_v2",
}
_DATA_DRIVEN_ASSIGNMENT_RANK_SUPERVISIONS = {
    "official_same_image_same_category_assignment_v1",
    "role_routed_official_assignment_top1_v1",
    "role_routed_official_assignment_all_exclusive_nonowned_v2",
}
_DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA = (
    "pivot.stageb.data_driven.official_assignment_pair/v1"
)
_DATA_DRIVEN_ASSIGNMENT_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.official_assignment_pair_receipt/v1"
)
_DATA_DRIVEN_ASSIGNMENT_OVERFIT_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.assignment_overfit64_receipt/v1"
)
_DATA_DRIVEN_ROLE_ROUTED_CLEAN_ASSIGNMENT_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.role_routed_clean_assignment_receipt/v1"
)
_DATA_DRIVEN_ROLE_ROUTED_CLEAN_ASSIGNMENT_SCOPE = (
    "official_assignment_clean_train_263661_v1"
)
_DATA_DRIVEN_NEW_HEAD_PARTITION_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.new_head_partition_receipt/v1"
)
_DATA_DRIVEN_SUPPORT_PARTITION_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.support_partition_receipt/v1"
)
_NATIVE_PATCH_CATEGORY_D1_RECEIPT_SCHEMA = (
    "pivot.stageb.native_patch_category_d1_receipt/v1"
)
_NATIVE_PATCH_CATEGORY_D1_ROW_SCHEMA = (
    "pivot.stageb.native_patch_category_d1_row/v1"
)
_NATIVE_PATCH_CATEGORY_D2_RECEIPT_SCHEMA = (
    "pivot.stageb.native_patch_category_d2_receipt/v1"
)
_NATIVE_PATCH_CATEGORY_D2_ROW_SCHEMA = (
    "pivot.stageb.native_patch_category_d2_row/v1"
)
_NATIVE_PATCH_CATEGORY_D2_SAMPLING_CONTRACT = (
    "source_mix_2_2_1_group_dedup_capped_sqrt_class_v1"
)
_NATIVE_PATCH_CATEGORY_D2_WEIGHT_FIELD = (
    "native_patch_category_sampling_weight"
)
_DATA_DRIVEN_SUPPORT_REQUIRED_SETTINGS = {
    "patch_bank_cache": False,
    "patch_bank_cache_write": False,
    "support_patch_use_embedding": False,
    "support_patch_max_per_class": 200,
}
_DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS = (
    "refcoco_stageb_phrase_v1.jsonl",
    "refcocoplus_stageb_phrase_v1.jsonl",
    "refcocog_stageb_phrase_v1.jsonl",
)
_DATA_DRIVEN_NEW_HEAD_VARIANTS = (
    "d0_ordinary_primary",
    "d1_category_complete",
)
_DATA_DRIVEN_NEW_HEAD_VARIANT_BY_DATASET_VARIANT = {
    "dd0_ordinary_primary": "d0_ordinary_primary",
    "dd1_category_complete": "d1_category_complete",
}
_DATA_DRIVEN_NEW_HEAD_PARTITIONS = (
    "train",
    "dev_full",
    "dev_screen",
    "quarantine",
)
_LOWER_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_NATIVE_PATCH_SUPPORT_WITNESS_KEYS = {
    "candidate_id",
    "class_assignment",
    "class_id",
    "coco_id",
    "content_sha256",
    "path",
    "selection_priority_sha256",
    "size_bytes",
    "source",
    "source_cache_class_id",
    "source_class",
    "source_image_id",
    "source_image_identity",
    "source_row_number",
    "source_row_sha256",
    "support_partition_receipt_sha256",
    "train_filtered",
}
_NATIVE_PATCH_CATEGORY_D2_SUPPORT_ROTATION_CONTRACT = (
    "pivot.stageb.native_patch_category_d2.support_rotation/v1"
)
_NATIVE_PATCH_CATEGORY_D2_SUPPORT_WITNESS_KEYS = (
    _NATIVE_PATCH_SUPPORT_WITNESS_KEYS - {"selection_priority_sha256"}
) | {
    "rotation_key_sha256",
    "rotation_offset",
    "rotation_pool_size",
    "rotation_selected_index",
    "rotation_start_index",
    "selection_contract",
}


def _validate_native_patch_category_meta(
    meta: Dict[str, Any],
    row_index: int,
    *,
    variant: str = "d1",
    expected_source_dataset: Optional[str] = None,
    alias_bridges: Optional[Dict[int, int]] = None,
) -> None:
    context = f"native patch-category row {row_index}"
    instances = meta.get("instances")
    witness = meta.get("support_patch_witness")
    query_witness = meta.get("query_image_witness")
    d1_contract = (
        variant == "d1"
        and meta.get("stage_b_native_patch_category_d1") is True
        and meta.get("stage_b_native_patch_category_d1_schema")
        == _NATIVE_PATCH_CATEGORY_D1_ROW_SCHEMA
        and meta.get("stage_b_u2_category_complete") is True
        and meta.get("stage_b_u2_category_complete_schema")
        == "pivot.stageb.u2_category_complete_ref/v1"
    )
    d2_contract = (
        variant == "d2"
        and meta.get("stage_b_native_patch_category_d2") is True
        and meta.get("stage_b_native_patch_category_d2_schema")
        == _NATIVE_PATCH_CATEGORY_D2_ROW_SCHEMA
        and all(
            key not in meta
            for key in (
                "stage_b_native_patch_category_d1",
                "stage_b_native_patch_category_d1_schema",
                "stage_b_u2_category_complete",
                "stage_b_u2_category_complete_schema",
            )
        )
    )
    if not (
        (d1_contract or d2_contract)
        and meta.get("primary_support_instance_index") == 0
        and isinstance(instances, list)
        and bool(instances)
        and all(isinstance(instance, dict) for instance in instances)
    ):
        raise ValueError(f"{context} lost its sealed category-complete contract")
    class_ids = [instance.get("class_id") for instance in instances]
    if (
        any(isinstance(class_id, bool) or not isinstance(class_id, int) for class_id in class_ids)
        or len(set(class_ids)) != 1
        or instances[0].get("category_complete_primary") is not True
        or not isinstance(instances[0].get("raw_phrase"), str)
        or not instances[0]["raw_phrase"].strip()
    ):
        raise ValueError(f"{context} has invalid same-category full-text instances")
    expected_witness_keys = (
        _NATIVE_PATCH_SUPPORT_WITNESS_KEYS
        if variant == "d1"
        else _NATIVE_PATCH_CATEGORY_D2_SUPPORT_WITNESS_KEYS
    )
    if not (
        isinstance(witness, dict)
        and set(witness) == expected_witness_keys
        and witness.get("train_filtered") is True
        and witness.get("class_id") == class_ids[0]
        and isinstance(witness.get("path"), str)
        and bool(witness["path"].strip())
        and type(witness.get("size_bytes")) is int
        and witness["size_bytes"] > 0
        and isinstance(witness.get("content_sha256"), str)
        and _LOWER_SHA256_RE.fullmatch(witness["content_sha256"]) is not None
    ):
        raise ValueError(f"{context} has an invalid row-locked support witness")
    assignment = witness.get("class_assignment")
    source_class_id = witness.get("source_cache_class_id")
    if assignment == "sealed_cache_identity_v1":
        valid_assignment = source_class_id == class_ids[0]
    elif assignment == "canonical_compact_alias_bridge_v1":
        valid_assignment = bool(
            isinstance(alias_bridges, dict)
            and alias_bridges.get(int(class_ids[0])) == source_class_id
            and source_class_id != class_ids[0]
        )
    else:
        valid_assignment = False
    if not valid_assignment:
        raise ValueError(f"{context} support class assignment is not sealed")
    if variant == "d2":
        rotation_pool_size = witness.get("rotation_pool_size")
        rotation_start_index = witness.get("rotation_start_index")
        rotation_offset = witness.get("rotation_offset")
        rotation_selected_index = witness.get("rotation_selected_index")
        rotation_payload = {
            "namespace": _NATIVE_PATCH_CATEGORY_D2_SUPPORT_ROTATION_CONTRACT,
            "group_id": meta.get("native_patch_category_group_id"),
            "source_identity_sha256": meta.get(
                "native_patch_category_source_identity_sha256"
            ),
        }
        expected_rotation_key = hashlib.sha256(
            json.dumps(
                rotation_payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        valid_rotation = (
            witness.get("selection_contract")
            == _NATIVE_PATCH_CATEGORY_D2_SUPPORT_ROTATION_CONTRACT
            and witness.get("rotation_key_sha256") == expected_rotation_key
            and type(rotation_pool_size) is int
            and rotation_pool_size > 0
            and type(rotation_start_index) is int
            and 0 <= rotation_start_index < rotation_pool_size
            and type(rotation_offset) is int
            and 0 <= rotation_offset < rotation_pool_size
            and type(rotation_selected_index) is int
            and 0 <= rotation_selected_index < rotation_pool_size
            and rotation_selected_index
            == (rotation_start_index + rotation_offset) % rotation_pool_size
        )
        if not valid_rotation:
            raise ValueError(f"{context} support rotation proof drifted")
    if not (
        isinstance(query_witness, dict)
        and set(query_witness)
        == {"content_sha256", "path", "size_bytes", "source_filename"}
        and isinstance(query_witness.get("content_sha256"), str)
        and _LOWER_SHA256_RE.fullmatch(query_witness["content_sha256"])
        is not None
        and query_witness["content_sha256"] != witness["content_sha256"]
    ):
        raise ValueError(f"{context} has an invalid query-image witness")
    query_image_id = meta.get("image_id")
    d1_identity = (
        variant == "d1"
        and type(meta.get("native_patch_category_variant_index")) is int
        and meta["native_patch_category_variant_index"] in {0, 1, 2}
    )
    d2_weight = meta.get(_NATIVE_PATCH_CATEGORY_D2_WEIGHT_FIELD)
    d2_source_dataset = meta.get("native_patch_category_source_dataset")
    d2_source_mix_weight = meta.get(
        "native_patch_category_source_mix_weight"
    )
    expected_d2_mix_weight = {
        "refcoco": 2,
        "refcocoplus": 2,
        "refcocog": 1,
    }.get(d2_source_dataset)
    d2_identity = (
        variant == "d2"
        and meta.get("native_patch_category_class_id") == class_ids[0]
        and d2_source_dataset in {"refcoco", "refcocoplus", "refcocog"}
        and (
            expected_source_dataset is None
            or d2_source_dataset == expected_source_dataset
        )
        and isinstance(
            meta.get("native_patch_category_source_identity_sha256"), str
        )
        and _LOWER_SHA256_RE.fullmatch(
            meta["native_patch_category_source_identity_sha256"]
        )
        is not None
        and type(meta.get("native_patch_category_source_line_number")) is int
        and meta["native_patch_category_source_line_number"] > 0
        and type(
            meta.get("native_patch_category_source_group_expression_count")
        )
        is int
        and meta["native_patch_category_source_group_expression_count"] > 0
        and type(d2_source_mix_weight) is int
        and d2_source_mix_weight == expected_d2_mix_weight
        and meta.get("native_patch_category_sampling_contract")
        == _NATIVE_PATCH_CATEGORY_D2_SAMPLING_CONTRACT
        and isinstance(d2_weight, float)
        and math.isfinite(d2_weight)
        and d2_weight > 0.0
    )
    if (
        type(query_image_id) is not int
        or witness.get("coco_id") == query_image_id
        or not (d1_identity or d2_identity)
        or not isinstance(meta.get("native_patch_category_group_id"), str)
        or not meta["native_patch_category_group_id"]
    ):
        raise ValueError(f"{context} support/query identity contract drifted")


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = _PUNC_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _tokenize_norm(s: str) -> List[str]:
    s = _norm_text(s)
    return s.split() if s else []


def _clean_for_alignment(s: Any) -> str:
    s = str(s or "").replace("_", " ").replace(".", " ").strip()
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _tokenize_with_offsets(s: Any) -> List[Dict[str, Any]]:
    text = _clean_for_alignment(s)
    out: List[Dict[str, Any]] = []
    for m in _TOKEN_RE.finditer(text):
        token = m.group(0)
        norm = _norm_text(token)
        if not norm:
            continue
        out.append({"text": token, "norm": norm, "start": int(m.start()), "end": int(m.end())})
    return out


def _normalize_tn_category(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = _TN_CATEGORY_SEP_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


_TN_GROUP_NAMES = ("color_like", "attr_like", "spatial_like", "relation_action_like", "other")
_TN_GROUP_TO_ID = {name: idx for idx, name in enumerate(_TN_GROUP_NAMES)}
_TN_ID_TO_GROUP = {idx: name for name, idx in _TN_GROUP_TO_ID.items()}

_TN_ATTR_LIKE_CATEGORIES = {
    "attribute",
    "size",
    "height",
    "length",
    "shape",
    "clothing",
    "clothing type",
    "clothing state",
    "sleeve length",
    "age",
    "state",
    "condition",
    "material",
    "accessory",
    "accessory type",
    "pattern",
    "texture",
}
_TN_SPATIAL_LIKE_CATEGORIES = {
    "spatial",
    "position",
    "spatial relation",
    "spatial position",
    "location",
}
_TN_RELATION_ACTION_LIKE_CATEGORIES = {
    "action",
    "posture",
    "pose",
}


def _tn_category_group(value: Any) -> str:
    category = _normalize_tn_category(value)
    if not category:
        return "other"
    if "color" in category:
        return "color_like"
    if category in _TN_ATTR_LIKE_CATEGORIES:
        return "attr_like"
    if category in _TN_SPATIAL_LIKE_CATEGORIES:
        return "spatial_like"
    if category in _TN_RELATION_ACTION_LIKE_CATEGORIES:
        return "relation_action_like"
    return "other"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _validate_single_edit_token_provenance(
    row: Dict[str, Any], *, context: str
) -> None:
    """Validate the immutable single-edit contract before granting supervision."""
    edits = row.get("tn_edits")
    if not (
        isinstance(edits, list)
        and len(edits) == 1
        and isinstance(edits[0], dict)
    ):
        raise ValueError(
            f"single-edit token provenance requires exactly one tn_edits entry at {context}"
        )
    edit = edits[0]
    category = edit.get("category")
    replace_from = edit.get("replace_from")
    replace_to = edit.get("replace_to")
    replace_span = edit.get("replace_span")
    if not all(
        isinstance(value, str) and bool(value.strip())
        for value in (category, replace_from, replace_to)
    ):
        raise ValueError(
            f"single-edit token provenance has invalid text fields at {context}"
        )
    normalized_from = _WS_RE.sub(" ", replace_from.strip().lower())
    normalized_to = _WS_RE.sub(" ", replace_to.strip().lower())
    if normalized_from == normalized_to:
        raise ValueError(
            f"single-edit token provenance does not change text at {context}"
        )
    if not (
        isinstance(replace_span, list)
        and len(replace_span) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in replace_span
        )
        and 0 <= replace_span[0] < replace_span[1]
    ):
        raise ValueError(
            f"single-edit token provenance has an invalid replace_span at {context}"
        )
    expected = {
        "replace_category": [category],
        "replace_from": [replace_from],
        "replace_to": [replace_to],
        "replace_span": [replace_span],
    }
    inconsistent = [
        key for key, value in expected.items() if row.get(key) != value
    ]
    if inconsistent:
        raise ValueError(
            "single-edit token provenance disagrees with top-level "
            f"{', '.join(inconsistent)} at {context}"
        )


_TN_CATEGORY_WEIGHTS = {name: 1.0 for name in _TN_GROUP_NAMES}
_TN_SKIP_CATEGORIES = set()

_CONTENT_EXCLUDED_TOKENS = {
    "a",
    "an",
    "the",
}

_RELATION_ACTION_WORDS = {
    "above",
    "across",
    "behind",
    "below",
    "beside",
    "between",
    "bottom",
    "carrying",
    "center",
    "close",
    "closest",
    "down",
    "eating",
    "far",
    "farthest",
    "front",
    "holding",
    "inside",
    "left",
    "looking",
    "near",
    "nearest",
    "next",
    "outside",
    "over",
    "right",
    "riding",
    "sitting",
    "standing",
    "top",
    "under",
    "up",
    "wearing",
}

_GENERIC_VISUAL_ATTRIBUTE_WORDS = {
    "black",
    "blue",
    "brown",
    "clear",
    "dark",
    "gray",
    "green",
    "grey",
    "light",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "white",
    "yellow",
    "big",
    "large",
    "little",
    "small",
    "short",
    "tall",
    "long",
    "wide",
    "thin",
    "thick",
    "striped",
    "plain",
    "solid",
    "spotted",
    "checked",
    "floral",
    "wooden",
    "metal",
    "plastic",
    "glass",
    "leather",
    "smooth",
    "rough",
    "shiny",
    "matte",
    "broken",
    "whole",
    "open",
    "closed",
    "full",
    "empty",
    "wet",
    "dry",
}


def _path_env_defaults() -> Dict[str, str]:
    media_user = os.environ.get("MEDIA_USER", "haoyi")
    t9_root = os.environ.get("T9_ROOT", f"/media/{media_user}/T9")
    data_root = os.environ.get("DATA_ROOT", os.path.join(t9_root, "data"))
    gdino_root = os.environ.get("GDINO_ROOT", os.path.join(t9_root, "gdino"))
    return {
        "MEDIA_USER": media_user,
        "T9_ROOT": t9_root,
        "DATA_ROOT": data_root,
        "GDINO_ROOT": gdino_root,
    }


def _expand_path_like(value):
    if value is None:
        return None
    if isinstance(value, str):
        out = value
        for key, default_value in _path_env_defaults().items():
            out = out.replace(f"${{{key}}}", default_value)
            out = out.replace(f"${key}", default_value)
        out = os.path.expandvars(out)
        out = os.path.expanduser(out)
        return str(remap_legacy_path(out))
    if isinstance(value, list):
        return [_expand_path_like(v) for v in value]
    return value


def _singularize_token(t: str) -> str:
    if len(t) <= 3:
        return t
    if t.endswith("ss"):
        return t
    if t.endswith("s"):
        return t[:-1]
    return t


def _build_name_to_canonical_id(canonical_classes_json: Optional[str]) -> Dict[str, int]:
    """
    Build name -> canonical_id mapping from your canonical class file
    (`/media/haoyi/T9/data/canonical_classes_with_aliases.json`).
    """
    if not canonical_classes_json:
        return {}
    path = Path(canonical_classes_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("canonical_classes_json must be a list of canonical class entries.")

    out: Dict[str, int] = {}
    for entry in data:
        cid = entry.get("id", None)
        if cid is None:
            continue
        cid = int(cid)
        for k in ("raw_name", "norm_name", "base_name", "synset"):
            v = entry.get(k, None)
            if isinstance(v, str) and v.strip():
                out[_norm_text(v)] = cid
        syns = entry.get("synonyms", None)
        if isinstance(syns, list):
            for v in syns:
                if isinstance(v, str) and v.strip():
                    out[_norm_text(v)] = cid
        aliases = entry.get("aliases", None)
        if isinstance(aliases, list):
            for a in aliases:
                if not isinstance(a, dict):
                    continue
                for k in ("name", "norm_name"):
                    v = a.get(k, None)
                    if isinstance(v, str) and v.strip():
                        out[_norm_text(v)] = cid
    return out


def _build_canonical_text_maps(
    canonical_classes_json: Optional[str],
) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """
    Build canonical_id -> preferred text / alias candidates from canonical class metadata.
    """
    if not canonical_classes_json:
        return {}, {}

    path = Path(canonical_classes_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("canonical_classes_json must be a list of canonical class entries.")

    cid_to_name: Dict[int, str] = {}
    cid_to_aliases: Dict[int, List[str]] = {}

    def _append_alias(dst: List[str], seen: set, value: Optional[str]) -> None:
        if not isinstance(value, str):
            return
        cleaned = " ".join(str(value).replace("_", " ").replace(".", " ").strip().split())
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        dst.append(cleaned)

    for entry in data:
        cid = entry.get("id", None)
        if cid is None:
            continue
        cid = int(cid)

        aliases: List[str] = []
        seen = set()
        preferred = None
        for k in ("base_name", "raw_name", "norm_name", "synset"):
            v = entry.get(k, None)
            if isinstance(v, str) and v.strip():
                cleaned = " ".join(str(v).replace("_", " ").replace(".", " ").strip().split())
                if cleaned:
                    if preferred is None:
                        preferred = cleaned
                    _append_alias(aliases, seen, cleaned)
        for v in entry.get("synonyms", []) or []:
            _append_alias(aliases, seen, v)
        for alias in entry.get("aliases", []) or []:
            if isinstance(alias, dict):
                _append_alias(aliases, seen, alias.get("name", None))
                _append_alias(aliases, seen, alias.get("norm_name", None))
            else:
                _append_alias(aliases, seen, alias)

        if preferred is None and aliases:
            preferred = aliases[0]
        if preferred is not None:
            cid_to_name[cid] = preferred
        cid_to_aliases[cid] = aliases

    return cid_to_name, cid_to_aliases


class _PhraseMatcher:
    """
    Fast token-level matcher for mapping phrases -> canonical_id by longest prefix match.
    """

    def __init__(self, name2cid: Dict[str, int]) -> None:
        by_first: Dict[str, List[Tuple[List[str], int]]] = {}
        for name, cid in name2cid.items():
            toks = _tokenize_norm(name)
            if not toks:
                continue
            by_first.setdefault(toks[0], []).append((toks, int(cid)))
        self.by_first = by_first

    def match_prefix(self, phrase: str) -> Optional[int]:
        toks = _tokenize_norm(phrase)
        if not toks:
            return None
        # Drop common leading determiners/quantifiers.
        while toks and toks[0] in {"a", "an", "the", "this", "that", "these", "those"}:
            toks = toks[1:]
        if not toks:
            return None

        toks_s = [_singularize_token(t) for t in toks]

        first = toks[0]
        cands = self.by_first.get(first, []) + self.by_first.get(toks_s[0], [])
        if not cands:
            return None

        best_len = 0
        best_cid: Optional[int] = None
        ambiguous = False
        for name_toks, cid in cands:
            L = len(name_toks)
            if L > len(toks):
                continue
            if toks[:L] == name_toks or toks_s[:L] == name_toks:
                if L > best_len:
                    best_len = L
                    best_cid = cid
                    ambiguous = False
                elif L == best_len and best_cid is not None and best_cid != cid:
                    ambiguous = True
        if best_len == 0 or best_cid is None or ambiguous:
            return None
        return best_cid


class _PhraseClassifierLabeler:
    """
    Canonical class recognizer for phrases, using the checkpoint trained by
    `models/GroundingDINO/train_classifier_clean.py` (e.g. exp_vg_multiclass_clean/best.pt).
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cpu",
        max_length: int = 24,
        batch_size: int = 64,
        min_conf: float = 0.0,
    ) -> None:
        from transformers import AutoConfig, AutoTokenizer, BertModel

        from models.GroundingDINO.bertwarper import BertModelWarper

        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.min_conf = float(min_conf)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        bert_model_name = ckpt.get("bert_model_name", "bert-base-uncased")
        num_classes = int(ckpt.get("num_classes", 2048))

        config = None
        try:
            config = AutoConfig.from_pretrained(bert_model_name, local_files_only=True)
        except Exception:
            config = AutoConfig.from_pretrained(bert_model_name)

        bert = BertModel(config)

        class _Model(torch.nn.Module):
            def __init__(self, bert_model, num_classes_: int):
                super().__init__()
                self.text_encoder = BertModelWarper(bert_model)
                hidden_size = self.text_encoder.config.hidden_size
                self.dropout = torch.nn.Dropout(0.1)
                self.classifier = torch.nn.Linear(hidden_size, num_classes_)

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                pooled = outputs.pooler_output
                pooled = self.dropout(pooled)
                return self.classifier(pooled)

        self.model = _Model(bert, num_classes)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading phrase classifier: {unexpected[:10]}")
        if missing:
            # allow missing keys only if they are from buffers introduced by HF version changes
            keep_missing = [k for k in missing if "position_ids" in k]
            bad_missing = [k for k in missing if k not in keep_missing]
            if bad_missing:
                raise RuntimeError(f"Missing keys when loading phrase classifier: {bad_missing[:10]}")

        self.model.eval().to(self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)

    @torch.inference_mode()
    def predict_top1(self, phrases: List[str]) -> List[Tuple[Optional[int], float]]:
        out: List[Tuple[Optional[int], float]] = []
        if not phrases:
            return out

        def _canon_phrase(p: str) -> str:
            p = (p or "").strip()
            if p and (p[-1] not in ".?!"):
                p = p + "."
            return p

        phrases = [_canon_phrase(p) for p in phrases]

        for i in range(0, len(phrases), self.batch_size):
            chunk = phrases[i : i + self.batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            token_type_ids = enc.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            prob = torch.softmax(logits, dim=-1)
            conf, pred = prob.max(dim=-1)
            conf = conf.detach().cpu().tolist()
            pred = pred.detach().cpu().tolist()
            for c, p in zip(conf, pred):
                if c >= self.min_conf:
                    out.append((int(p), float(c)))
                else:
                    out.append((None, float(c)))
        return out


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))
    return metas


class _LazyJsonlRows(Sequence[Dict[str, Any]]):
    """Read immutable JSONL rows by byte offset without retaining decoded rows."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        before = self.path.stat()
        offsets = array("Q")
        lengths = array("Q")
        offset = 0
        with self.path.open("rb", buffering=0) as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
                    lengths.append(len(line))
                offset += len(line)
        after = self.path.stat()
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or offset != after.st_size:
            raise RuntimeError(f"JSONL changed while indexing: {self.path}")
        self._identity = after_identity
        self._offsets = offsets
        self._lengths = lengths
        self._fd: Optional[int] = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _open_fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)
        stat = os.fstat(self._fd)
        identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if identity != self._identity:
            self.close()
            raise RuntimeError(f"JSONL changed after indexing: {self.path}")
        return self._fd

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if not isinstance(index, int):
            raise TypeError(f"JSONL row index must be int or slice, got {type(index)}")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        length = int(self._lengths[index])
        payload = os.pread(self._open_fd(), length, int(self._offsets[index]))
        if len(payload) != length:
            raise RuntimeError(f"JSONL row {index} became truncated: {self.path}")
        try:
            row = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"JSONL row {index} became unreadable: {self.path}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL row {index} is not an object: {self.path}")
        return row

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fd"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON root type: {type(data)}; expected list[dict].")


def _xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _clamp_box_xyxy(box: torch.Tensor, w: int, h: int) -> torch.Tensor:
    x0, y0, x1, y1 = box.unbind(-1)
    x0 = x0.clamp(0, w - 1)
    y0 = y0.clamp(0, h - 1)
    x1 = x1.clamp(0, w)
    y1 = y1.clamp(0, h)
    return torch.stack([x0, y0, x1, y1], dim=-1)


def _safe_crop(img: Image.Image, box_xyxy: torch.Tensor) -> Optional[Image.Image]:
    w, h = img.size
    box_xyxy = _clamp_box_xyxy(box_xyxy, w=w, h=h)
    x0, y0, x1, y1 = box_xyxy.tolist()
    x0 = int(max(0, min(w - 1, round(x0))))
    y0 = int(max(0, min(h - 1, round(y0))))
    x1 = int(max(x0 + 1, min(w, round(x1))))
    y1 = int(max(y0 + 1, min(h, round(y1))))
    if x1 <= x0 or y1 <= y0:
        return None
    return img.crop((x0, y0, x1, y1))


@dataclass(frozen=True)
class PatchEpisodeConfig:
    box_format: str = "xyxy"  # "xyxy" or "xywh"
    neg_episode_prob: float = 0.2
    support_min_count: int = 2
    support_patch_size: int = 224
    support_num_patches_min: int = 1
    support_num_patches_max: int = 1
    support_use_all_gt_classes: bool = False
    lvis_neg_category_only: bool = False
    support_patch_max_per_class: int = 0  # 0 = keep all candidates
    negative_max_tries: int = 50
    support_patch_bucket: Optional[str] = None
    support_patch_use_embedding: bool = False
    patch_emb_cache_size: int = 4096
    support_patch_image_root: Optional[str] = None
    keep_only_support_gt: bool = False
    keep_only_patchset_gt: bool = True
    patch_text_augment: bool = False
    patch_text_aug_p_object: float = 0.5
    patch_text_aug_p_alias: float = 0.25
    patch_text_aug_p_vg: float = 0.25
    patch_text_aug_vg_jsonl: Optional[str] = None
    patch_text_aug_vg_pool_size: int = 50000
    patch_text_aug_max_words: int = 6
    vg_phrase_labeler: str = "prefix"  # prefix | classifier | hybrid
    phrase_classifier_ckpt: Optional[str] = None
    phrase_classifier_device: str = "cpu"
    phrase_classifier_max_length: int = 24
    phrase_classifier_batch_size: int = 64
    phrase_classifier_min_conf: float = 0.0
    phrase_cache_size: int = 50000
    patch_bank_cache: bool = True
    patch_bank_cache_path: Optional[str] = None
    patch_bank_cache_write: bool = True
    anno_cache: bool = True
    anno_cache_path: Optional[str] = None
    anno_cache_write: bool = True
    build_text_token_masks: bool = False
    text_encoder_type: str = "bert-base-uncased"
    max_text_len: int = 256
    text_mask_warn_limit: int = 20
    text_mask_skip_invalid_canonical: bool = False
    text_mask_audit_jsonl: Optional[str] = None
    use_tn_category_weights: bool = True
    default_tn_category_weight: float = 1.0
    skip_tn_if_neg_overlaps_canonical: bool = True
    skip_ambiguous_tn: bool = True
    skip_tn_if_changed_span_not_found: bool = True
    skip_tn_if_changed_span_empty_after_filter: bool = True
    skip_relation_like_tn_in_v1: bool = False
    tn_balance_sampling: bool = True
    tn_balance_cap: float = 5.0
    sam3_tn_image_root: Optional[str] = None
    sam3_tn_bbox_key: str = "sam_bbox"
    sam3_tn_keep_failed: bool = False
    require_global_tn_verified: bool = False
    require_fixed_stagea_topk_exact_verified: bool = False
    fixed_stagea_topk_exact_audit: Optional[str] = None
    fixed_stagea_topk_expected_contract: Optional[Dict[str, Any]] = None
    require_proposalset_proxy_verified: bool = False
    require_benchmark_dataft_alltn: bool = False
    require_vlm_strict_tn: bool = False
    require_single_edit_token_provenance: bool = False
    table_b_id: Optional[str] = None
    table_b_scope: Optional[str] = None
    table_b_audit_sha256: Optional[str] = None
    stage_b_gdino_adapter_ref_eval: bool = False
    stage_b_gdino_adapter_no_support: bool = False
    native_patch_category_row_locked_support: bool = False
    native_patch_category_variant: Optional[str] = None
    native_patch_category_source_dataset: Optional[str] = None
    native_patch_category_alias_bridges: Optional[Dict[int, int]] = None
    strict_sample_identity: bool = False
    lazy_jsonl: bool = False


class PatchEpisodeJsonlDataset(VisionDataset):
    """
    Patch-only episode dataset.

    Each sample returns:
      - image: query image (after DETR-style transforms)
      - target:
          boxes:  (N,4) normalized cxcywh (done by Normalize transform)
          labels: (N,) canonical class_id (int64)
          support_class: (1,) support canonical class_id
          patch: (3, S, S) support crop (ImageNet normalized)
          is_negative_episode: (1,) 1 if support_class not in image
          caption/cap_list: dummy text prompt ("object")

    Expected annotation format (jsonl or json list):
      {
        "filename" or "file_name": "relative/path.jpg",
        "width": int, "height": int, (optional; inferred from image if missing)
        "instances": [{"bbox": [..4..], "class_id": int}, ...]
      }

    For `source="vg_region_descriptions"` (VAW/VG):
      - the dataset will store regions with raw `phrase` and resolve `class_id` at runtime via:
          - prefix match over canonical aliases (`vg_phrase_labeler="prefix"`), or
          - classifier (`vg_phrase_labeler="classifier"` / "hybrid") using `phrase_classifier_ckpt`.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        transforms: Optional[Any] = None,
        box_format: str = "xyxy",
        neg_episode_prob: float = 0.2,
        support_min_count: int = 2,
        support_patch_size: int = 224,
        support_num_patches_min: int = 1,
        support_num_patches_max: int = 1,
        support_use_all_gt_classes: bool = False,
        lvis_neg_category_only: bool = False,
        support_patch_max_per_class: int = 0,
        negative_max_tries: int = 50,
        support_patch_tsv: Optional[str] = None,
        support_patch_bucket: Optional[str] = None,
        support_patch_class_map_json: Optional[str] = None,
        canonical_classes_json: Optional[str] = None,
        source: Optional[str] = None,
        lvis_image_root: Optional[str] = None,
        coco_image_root: Optional[str] = None,
        vg_image_roots: Optional[List[str]] = None,
        support_patch_use_embedding: bool = False,
        patch_emb_cache_size: int = 4096,
        support_patch_image_root: Optional[str] = None,
        keep_only_support_gt: bool = False,
        keep_only_patchset_gt: bool = True,
        patch_text_augment: bool = False,
        patch_text_aug_p_object: float = 0.5,
        patch_text_aug_p_alias: float = 0.25,
        patch_text_aug_p_vg: float = 0.25,
        patch_text_aug_vg_jsonl: Optional[str] = None,
        patch_text_aug_vg_pool_size: int = 50000,
        patch_text_aug_max_words: int = 6,
        vg_phrase_labeler: str = "prefix",
        phrase_classifier_ckpt: Optional[str] = None,
        phrase_classifier_device: str = "cpu",
        phrase_classifier_max_length: int = 24,
        phrase_classifier_batch_size: int = 64,
        phrase_classifier_min_conf: float = 0.0,
        phrase_cache_size: int = 50000,
        patch_bank_cache: bool = True,
        patch_bank_cache_path: Optional[str] = None,
        patch_bank_cache_write: bool = True,
        anno_cache: bool = True,
        anno_cache_path: Optional[str] = None,
        anno_cache_write: bool = True,
        build_text_token_masks: bool = False,
        text_encoder_type: str = "bert-base-uncased",
        max_text_len: int = 256,
        text_mask_warn_limit: int = 20,
        text_mask_skip_invalid_canonical: bool = False,
        text_mask_audit_jsonl: Optional[str] = None,
        use_tn_category_weights: bool = True,
        default_tn_category_weight: float = 1.0,
        skip_tn_if_neg_overlaps_canonical: bool = True,
        skip_ambiguous_tn: bool = True,
        skip_tn_if_changed_span_not_found: bool = True,
        skip_tn_if_changed_span_empty_after_filter: bool = True,
        skip_relation_like_tn_in_v1: bool = False,
        tn_balance_sampling: bool = True,
        tn_balance_cap: float = 5.0,
        sam3_tn_image_root: Optional[str] = None,
        sam3_tn_bbox_key: str = "sam_bbox",
        sam3_tn_keep_failed: bool = False,
        require_global_tn_verified: bool = False,
        require_fixed_stagea_topk_exact_verified: bool = False,
        fixed_stagea_topk_exact_audit: Optional[str] = None,
        fixed_stagea_topk_expected_contract: Optional[Dict[str, Any]] = None,
        require_proposalset_proxy_verified: bool = False,
        require_benchmark_dataft_alltn: bool = False,
        require_vlm_strict_tn: bool = False,
        require_single_edit_token_provenance: bool = False,
        table_b_id: Optional[str] = None,
        table_b_scope: Optional[str] = None,
        table_b_audit_sha256: Optional[str] = None,
        stage_b_gdino_adapter_ref_eval: bool = False,
        stage_b_gdino_adapter_no_support: bool = False,
        native_patch_category_row_locked_support: bool = False,
        native_patch_category_variant: Optional[str] = None,
        native_patch_category_source_dataset: Optional[str] = None,
        native_patch_category_alias_bridges: Optional[Dict[int, int]] = None,
        strict_sample_identity: bool = False,
        lazy_jsonl: bool = False,
    ) -> None:
        root = _expand_path_like(root)
        anno = _expand_path_like(anno)
        canonical_classes_json = _expand_path_like(canonical_classes_json)
        support_patch_class_map_json = _expand_path_like(support_patch_class_map_json)
        lvis_image_root = _expand_path_like(lvis_image_root)
        coco_image_root = _expand_path_like(coco_image_root)
        vg_image_roots = _expand_path_like(vg_image_roots or [])
        support_patch_tsv = _expand_path_like(support_patch_tsv)
        support_patch_image_root = _expand_path_like(support_patch_image_root)
        patch_text_aug_vg_jsonl = _expand_path_like(patch_text_aug_vg_jsonl)
        phrase_classifier_ckpt = _expand_path_like(phrase_classifier_ckpt)
        patch_bank_cache_path = _expand_path_like(patch_bank_cache_path)
        anno_cache_path = _expand_path_like(anno_cache_path)
        sam3_tn_image_root = _expand_path_like(sam3_tn_image_root)
        fixed_stagea_topk_exact_audit = _expand_path_like(
            fixed_stagea_topk_exact_audit
        )

        super().__init__(root=root, transforms=transforms)
        self.root = str(root)
        self.anno = str(anno)
        self.source = str(source) if source else None
        self._canonical_classes_json = canonical_classes_json
        self._support_patch_class_map_json = support_patch_class_map_json
        self._alt_image_roots = [Path(p) for p in (vg_image_roots or [])]
        if not isinstance(lazy_jsonl, bool):
            raise ValueError("lazy_jsonl must be an exact boolean")
        if lazy_jsonl and (
            not strict_sample_identity
            or require_fixed_stagea_topk_exact_verified
            or native_patch_category_row_locked_support
            or lvis_neg_category_only
            or str(self.source or "").lower()
            in {"sam3_tn_pair", "sam3_paired_tn", "sam3_and_tn"}
        ):
            raise ValueError(
                "lazy_jsonl is restricted to immutable strict-identity JSONL rows"
            )
        if require_fixed_stagea_topk_exact_verified:
            if str(self.source or "").lower() not in {
                "sam3_tn_pair",
                "sam3_paired_tn",
                "sam3_and_tn",
            }:
                raise ValueError(
                    "fixed Stage-A exact Top-K rows require a paired TN source"
                )
            if not fixed_stagea_topk_exact_audit:
                raise ValueError(
                    "require_fixed_stagea_topk_exact_verified requires an exact "
                    "verification sidecar audit"
                )
            try:
                fixed_stagea_topk_expected_contract = normalize_exact_contract(
                    fixed_stagea_topk_expected_contract
                )
            except ExactTopKContractError as error:
                raise ValueError(
                    "require_fixed_stagea_topk_exact_verified requires a complete "
                    f"expected provenance contract: {error}"
                ) from error
            if (
                float(neg_episode_prob) != 0.0
                or int(support_num_patches_min) != 1
                or int(support_num_patches_max) != 1
                or bool(support_patch_use_embedding)
            ):
                raise ValueError(
                    "fixed Stage-A exact Top-K rows require neg_episode_prob=0 and "
                    "exactly one row-locked image support patch"
                )
        if require_single_edit_token_provenance:
            if str(self.source or "").lower() not in {
                "sam3_tn_pair",
                "sam3_paired_tn",
                "sam3_and_tn",
            }:
                raise ValueError(
                    "single-edit token provenance requires a paired TN source"
                )
            if not build_text_token_masks:
                raise ValueError(
                    "single-edit token provenance requires build_text_token_masks=True"
                )
        if stage_b_gdino_adapter_no_support and not (
            require_benchmark_dataft_alltn
            or require_global_tn_verified
            or require_fixed_stagea_topk_exact_verified
            or require_vlm_strict_tn
            or stage_b_gdino_adapter_ref_eval
        ):
            raise ValueError(
                "stage_b_gdino_adapter_no_support is restricted to explicitly "
                "verified adapter training or evaluation datasets"
            )
        if native_patch_category_row_locked_support and (
            float(neg_episode_prob) != 0.0
            or int(support_num_patches_min) != 1
            or int(support_num_patches_max) != 1
            or bool(support_patch_use_embedding)
            or not bool(build_text_token_masks)
            or not bool(strict_sample_identity)
        ):
            raise ValueError(
                "native patch-category row-locked support requires one pixel "
                "support, full-text masks, zero negative episodes, and strict identity"
            )
        native_patch_category_variant = str(
            native_patch_category_variant or ""
        ).strip().lower()
        if native_patch_category_row_locked_support and (
            native_patch_category_variant not in {"d1", "d2"}
        ):
            raise ValueError(
                "native patch-category rows require an exact d1/d2 variant"
            )
        self.cfg = PatchEpisodeConfig(
            box_format=box_format,
            neg_episode_prob=neg_episode_prob,
            support_min_count=support_min_count,
            support_patch_size=support_patch_size,
            support_num_patches_min=int(support_num_patches_min),
            support_num_patches_max=int(support_num_patches_max),
            support_use_all_gt_classes=bool(support_use_all_gt_classes),
            lvis_neg_category_only=bool(lvis_neg_category_only),
            support_patch_max_per_class=int(support_patch_max_per_class),
            negative_max_tries=negative_max_tries,
            support_patch_bucket=support_patch_bucket,
            support_patch_use_embedding=support_patch_use_embedding,
            patch_emb_cache_size=patch_emb_cache_size,
            support_patch_image_root=str(support_patch_image_root) if support_patch_image_root else None,
            keep_only_support_gt=bool(keep_only_support_gt),
            keep_only_patchset_gt=bool(keep_only_patchset_gt),
            patch_text_augment=bool(patch_text_augment),
            patch_text_aug_p_object=float(patch_text_aug_p_object),
            patch_text_aug_p_alias=float(patch_text_aug_p_alias),
            patch_text_aug_p_vg=float(patch_text_aug_p_vg),
            patch_text_aug_vg_jsonl=str(patch_text_aug_vg_jsonl) if patch_text_aug_vg_jsonl else None,
            patch_text_aug_vg_pool_size=int(patch_text_aug_vg_pool_size),
            patch_text_aug_max_words=int(patch_text_aug_max_words),
            vg_phrase_labeler=str(vg_phrase_labeler),
            phrase_classifier_ckpt=phrase_classifier_ckpt,
            phrase_classifier_device=str(phrase_classifier_device),
            phrase_classifier_max_length=int(phrase_classifier_max_length),
            phrase_classifier_batch_size=int(phrase_classifier_batch_size),
            phrase_classifier_min_conf=float(phrase_classifier_min_conf),
            phrase_cache_size=int(phrase_cache_size),
            patch_bank_cache=bool(patch_bank_cache),
            patch_bank_cache_path=patch_bank_cache_path,
            patch_bank_cache_write=bool(patch_bank_cache_write),
            anno_cache=bool(anno_cache),
            anno_cache_path=anno_cache_path,
            anno_cache_write=bool(anno_cache_write),
            build_text_token_masks=bool(build_text_token_masks),
            text_encoder_type=str(text_encoder_type),
            max_text_len=int(max_text_len),
            text_mask_warn_limit=int(text_mask_warn_limit),
            text_mask_skip_invalid_canonical=bool(text_mask_skip_invalid_canonical),
            text_mask_audit_jsonl=str(text_mask_audit_jsonl) if text_mask_audit_jsonl else None,
            use_tn_category_weights=bool(use_tn_category_weights),
            default_tn_category_weight=float(default_tn_category_weight),
            skip_tn_if_neg_overlaps_canonical=bool(skip_tn_if_neg_overlaps_canonical),
            skip_ambiguous_tn=bool(skip_ambiguous_tn),
            skip_tn_if_changed_span_not_found=bool(skip_tn_if_changed_span_not_found),
            skip_tn_if_changed_span_empty_after_filter=bool(skip_tn_if_changed_span_empty_after_filter),
            skip_relation_like_tn_in_v1=bool(skip_relation_like_tn_in_v1),
            tn_balance_sampling=bool(tn_balance_sampling),
            tn_balance_cap=float(tn_balance_cap),
            sam3_tn_image_root=str(sam3_tn_image_root) if sam3_tn_image_root else None,
            sam3_tn_bbox_key=str(sam3_tn_bbox_key),
            sam3_tn_keep_failed=bool(sam3_tn_keep_failed),
            require_global_tn_verified=bool(require_global_tn_verified),
            require_fixed_stagea_topk_exact_verified=bool(
                require_fixed_stagea_topk_exact_verified
            ),
            fixed_stagea_topk_exact_audit=(
                str(fixed_stagea_topk_exact_audit)
                if fixed_stagea_topk_exact_audit
                else None
            ),
            fixed_stagea_topk_expected_contract=(
                dict(fixed_stagea_topk_expected_contract)
                if fixed_stagea_topk_expected_contract is not None
                else None
            ),
            require_proposalset_proxy_verified=bool(
                require_proposalset_proxy_verified
            ),
            require_benchmark_dataft_alltn=bool(
                require_benchmark_dataft_alltn
            ),
            require_vlm_strict_tn=bool(require_vlm_strict_tn),
            require_single_edit_token_provenance=bool(
                require_single_edit_token_provenance
            ),
            table_b_id=(str(table_b_id) if table_b_id is not None else None),
            table_b_scope=(
                str(table_b_scope) if table_b_scope is not None else None
            ),
            table_b_audit_sha256=(
                str(table_b_audit_sha256)
                if table_b_audit_sha256 is not None
                else None
            ),
            stage_b_gdino_adapter_ref_eval=bool(
                stage_b_gdino_adapter_ref_eval
            ),
            stage_b_gdino_adapter_no_support=bool(
                stage_b_gdino_adapter_no_support
            ),
            native_patch_category_row_locked_support=bool(
                native_patch_category_row_locked_support
            ),
            native_patch_category_variant=(
                native_patch_category_variant
                if native_patch_category_row_locked_support
                else None
            ),
            native_patch_category_source_dataset=(
                native_patch_category_source_dataset
                if native_patch_category_variant == "d2"
                else None
            ),
            native_patch_category_alias_bridges=(
                dict(native_patch_category_alias_bridges)
                if native_patch_category_alias_bridges is not None
                else None
            ),
            strict_sample_identity=bool(strict_sample_identity),
            lazy_jsonl=bool(lazy_jsonl),
        )

        self.name2cid = _build_name_to_canonical_id(canonical_classes_json)
        self.cid_to_name, self.cid_to_aliases = _build_canonical_text_maps(canonical_classes_json)
        self.phrase_matcher = _PhraseMatcher(self.name2cid) if self.name2cid else None
        self._phrase_cache: "OrderedDict[str, Optional[int]]" = OrderedDict()
        self._phrase_cls_labeler: Optional[_PhraseClassifierLabeler] = None
        self._phrase_cls_cache: "OrderedDict[str, Tuple[Optional[int], float]]" = OrderedDict()
        self._text_mask_warn_count = 0
        self._text_tokenizer = None
        if self.cfg.build_text_token_masks:
            try:
                self._text_tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg.text_encoder_type, use_fast=True, local_files_only=True
                )
            except Exception:
                self._text_tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg.text_encoder_type, use_fast=True
                )
            if not getattr(self._text_tokenizer, "is_fast", False):
                raise RuntimeError(
                    f"build_text_token_masks=True requires a fast tokenizer, got {type(self._text_tokenizer)}"
                )

        self._fixed_stagea_exact_rows: Dict[str, Dict[str, Any]] = {}
        self._fixed_support_patch_sha_cache: Dict[Path, Tuple[int, int, str]] = {}
        self._native_patch_support_sha_cache: Dict[
            Path, Tuple[int, int, int, int, str]
        ] = {}
        anno_path = Path(anno)
        if (
            self.cfg.require_fixed_stagea_topk_exact_verified
            and anno_path.suffix.lower() != ".jsonl"
        ):
            raise ValueError(
                "fixed Stage-A exact Top-K datasets require canonical JSONL plus "
                "a sidecar audit"
            )
        if anno_path.suffix.lower() == ".jsonl":
            self.metas = (
                _LazyJsonlRows(anno_path)
                if self.cfg.lazy_jsonl
                else _read_jsonl(anno_path)
            )
            if self.cfg.require_fixed_stagea_topk_exact_verified:
                try:
                    validated = validate_exact_pair_collection(
                        self.metas,
                        annotation_path=anno_path,
                        audit_path=Path(self.cfg.fixed_stagea_topk_exact_audit),
                        expected_contract=self.cfg.fixed_stagea_topk_expected_contract,
                    )
                except ExactTopKContractError as error:
                    raise ValueError(
                        f"fixed Stage-A exact Top-K dataset failed closed: {error}"
                    ) from error
                self._fixed_stagea_exact_rows = {
                    str(row["sample_id"]): summary
                    for row, summary in zip(self.metas, validated)
                }
            if str(self.source or "").lower() in {"sam3_tn_pair", "sam3_paired_tn", "sam3_and_tn"}:
                self.metas = self._normalize_sam3_tn_pair_metas(self.metas)
                # Normalized metas retain only the runtime replay tensors and
                # support binding, not the full per-candidate judgment payload.
                self._fixed_stagea_exact_rows = {}
        elif anno_path.suffix.lower() == ".json":
            if source in {"lvis", "coco", "vg_region_descriptions"}:
                self.metas = self._load_metas_cached(
                    anno_path,
                    src=str(source),
                    lvis_image_root=lvis_image_root,
                    coco_image_root=coco_image_root,
                    vg_image_roots=vg_image_roots,
                )
            else:
                with anno_path.open("r", encoding="utf-8") as f:
                    anno_data = json.load(f)

                detected = None
                if isinstance(anno_data, dict) and all(k in anno_data for k in ("annotations", "images", "categories")):
                    detected = "coco"
                    try:
                        images = anno_data.get("images", []) or []
                        if images and isinstance(images[0], dict) and (
                            ("neg_category_ids" in images[0]) or ("not_exhaustive_category_ids" in images[0])
                        ):
                            detected = "lvis"
                    except Exception:
                        detected = "coco"
                elif (
                    isinstance(anno_data, list)
                    and anno_data
                    and isinstance(anno_data[0], dict)
                    and "regions" in anno_data[0]
                    and "id" in anno_data[0]
                ):
                    detected = "vg_region_descriptions"
                elif isinstance(anno_data, list):
                    detected = "prebuilt"

                src = source or detected
                if self.source is None and src is not None:
                    self.source = str(src)
                if src == "lvis":
                    self.metas = self._load_metas_cached(
                        anno_path,
                        src="lvis",
                        lvis_image_root=lvis_image_root,
                        coco_image_root=coco_image_root,
                        vg_image_roots=vg_image_roots,
                        anno_data=anno_data,
                    )
                elif src == "coco":
                    self.metas = self._load_metas_cached(
                        anno_path,
                        src="coco",
                        lvis_image_root=lvis_image_root,
                        coco_image_root=coco_image_root,
                        vg_image_roots=vg_image_roots,
                        anno_data=anno_data,
                    )
                elif src == "vg_region_descriptions":
                    self.metas = self._load_metas_cached(
                        anno_path,
                        src="vg_region_descriptions",
                        lvis_image_root=lvis_image_root,
                        coco_image_root=coco_image_root,
                        vg_image_roots=vg_image_roots,
                        anno_data=anno_data,
                    )
                elif src == "prebuilt":
                    self.metas = anno_data
                elif src in {"sam3_tn_pair", "sam3_paired_tn", "sam3_and_tn"}:
                    self.metas = self._normalize_sam3_tn_pair_metas(anno_data)
                else:
                    raise ValueError(f"Unsupported source={src} detected={detected} for anno={anno_path}")
        else:
            raise ValueError(f"Unsupported anno extension: {anno_path.suffix}")

        if self.cfg.native_patch_category_row_locked_support:
            for row_index, meta in enumerate(self.metas):
                if not isinstance(meta, dict):
                    raise ValueError(
                        f"native patch-category row {row_index} is not an object"
                    )
                _validate_native_patch_category_meta(
                    meta,
                    row_index,
                    variant=str(self.cfg.native_patch_category_variant),
                    expected_source_dataset=(
                        self.cfg.native_patch_category_source_dataset
                    ),
                    alias_bridges=self.cfg.native_patch_category_alias_bridges,
                )

        if self.cfg.require_benchmark_dataft_alltn:
            for row_index, meta in enumerate(self.metas):
                if (
                    not isinstance(meta, dict)
                    or meta.get("benchmark_dataft_alltn", None) is not True
                    or meta.get("tn_scope", None) != "benchmark_dataft_alltn"
                    or meta.get("proposalset_proxy_verified", None) is not False
                ):
                    raise ValueError(
                        "benchmark data-FT adapter rows require exact boolean "
                        "benchmark_dataft_alltn=true, "
                        "tn_scope='benchmark_dataft_alltn', and no proposal proxy; "
                        f"invalid row {row_index} in {self.anno}"
                    )

        if self.cfg.require_vlm_strict_tn:
            for row_index, meta in enumerate(self.metas):
                audit = meta.get("proposal_audit", None) if isinstance(meta, dict) else None
                if (
                    not isinstance(meta, dict)
                    or meta.get("manifest_schema", None)
                    != "stageb_vlm_verified_strict_tn_v2"
                    or meta.get("visual_verified_negative", None) is not True
                    or meta.get("coverage_pass", None) is not True
                    or meta.get("coverage_policy", None) != "target_plus_proposal"
                    or not isinstance(audit, dict)
                    or audit.get("target_verified_no", None) is not True
                    or audit.get("target_plus_proposal_covered", None) is not True
                ):
                    raise ValueError(
                        "VLM strict-v2 TN rows require the exact verified-negative "
                        f"coverage contract; invalid row {row_index} in {self.anno}"
                    )

        if self.cfg.stage_b_gdino_adapter_ref_eval:
            for row_index, meta in enumerate(self.metas):
                instances = meta.get("instances", None) if isinstance(meta, dict) else None
                instance = instances[0] if isinstance(instances, list) and len(instances) == 1 else None
                if (
                    not isinstance(meta, dict)
                    or not all(meta.get(key, None) is not None for key in ("image_id", "ann_id", "ref_id", "sent_id"))
                    or not isinstance(instance, dict)
                    or instance.get("text_is_negative", None) is not False
                    or not isinstance(instance.get("positive_phrase", None), str)
                    or not instance["positive_phrase"].strip()
                ):
                    raise ValueError(
                        "GDINO adapter Ref evaluation rows require one positive "
                        f"identified expression; invalid row {row_index} in {self.anno}"
                    )

        if (self.cfg.vg_phrase_labeler in {"classifier", "hybrid"}) and (not self.cfg.phrase_classifier_ckpt):
            raise ValueError("vg_phrase_labeler requires phrase_classifier_ckpt when using classifier/hybrid.")

        self.patch_tfm = TV.Compose(
            [
                TV.Resize(256),
                TV.CenterCrop(self.cfg.support_patch_size),
                TV.ToTensor(),
                TV.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.patch_bank: Optional[Dict[int, List[str]]] = None
        self.patch_class_map: Optional[Dict[str, int]] = None
        if support_patch_class_map_json and not self.cfg.stage_b_gdino_adapter_no_support:
            with Path(support_patch_class_map_json).open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("support_patch_class_map_json must be a JSON object mapping class_name -> canonical_id.")
            self.patch_class_map = {_norm_text(str(k)): int(v) for k, v in raw.items()}
        if support_patch_tsv and not self.cfg.stage_b_gdino_adapter_no_support:
            self.patch_bank = self._load_patch_bank_cached(Path(support_patch_tsv))
            if not self.patch_bank:
                print(
                    f"[WARN] Loaded support_patch_tsv={support_patch_tsv} but patch_bank is empty. "
                    "If your TSV 'class' column is a string name (e.g. from emb_index_from_quality.tsv), "
                    "provide support_patch_class_map_json to map class_name -> canonical_id."
                )
        self._patch_emb_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._filter_lvis_neg_category_metas_if_needed()
        self.tn_category_stats = self._compute_tn_category_stats()
        tn_sample_weights = self._build_tn_balanced_sample_weights()
        native_patch_sample_weights = (
            self._build_native_patch_category_sample_weights()
        )
        if tn_sample_weights is not None and native_patch_sample_weights is not None:
            raise ValueError(
                "TN balancing and native patch-category D2 balancing are mutually exclusive"
            )
        self.sample_weights = (
            native_patch_sample_weights
            if native_patch_sample_weights is not None
            else tn_sample_weights
        )
        if self.tn_category_stats["total_edits"] > 0:
            print("[INFO] TN category stats for {}:\n{}".format(self.anno, json.dumps(self.tn_category_stats, indent=2)))

        # Text augmentation pools (Stage A robustness): canonical aliases and VG raw phrases.
        self._text_aug_alias_pool: List[str] = []
        self._text_aug_vg_pool: List[str] = []
        if self.cfg.patch_text_augment:
            if canonical_classes_json and Path(canonical_classes_json).exists():
                try:
                    with Path(canonical_classes_json).open("r", encoding="utf-8") as f:
                        cc = json.load(f)
                    pool = set()
                    if isinstance(cc, list):
                        for item in cc:
                            if not isinstance(item, dict):
                                continue
                            for k in ("raw_name", "norm_name"):
                                v = item.get(k, None)
                                if isinstance(v, str) and v.strip():
                                    pool.add(v.strip())
                            syn = item.get("synonyms", None)
                            if isinstance(syn, list):
                                for s in syn:
                                    if isinstance(s, str) and s.strip():
                                        pool.add(s.strip())
                            aliases = item.get("aliases", None)
                            if isinstance(aliases, list):
                                for a in aliases:
                                    if isinstance(a, dict):
                                        for k in ("name", "norm_name"):
                                            v = a.get(k, None)
                                            if isinstance(v, str) and v.strip():
                                                pool.add(v.strip())
                                    elif isinstance(a, str) and a.strip():
                                        pool.add(a.strip())
                    self._text_aug_alias_pool = sorted(pool)
                except Exception:
                    self._text_aug_alias_pool = []

            vg_path = self.cfg.patch_text_aug_vg_jsonl
            if vg_path and Path(vg_path).exists():
                # Reservoir sample a subset to avoid loading a huge file into each worker.
                import random as _r

                k = int(self.cfg.patch_text_aug_vg_pool_size)
                if k > 0:
                    buf: List[str] = []
                    n = 0
                    try:
                        with Path(vg_path).open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                phrase = obj.get("raw_phrase", None) or obj.get("head_phrase", None)
                                if not isinstance(phrase, str) or (not phrase.strip()):
                                    continue
                                phrase = phrase.strip()
                                n += 1
                                if len(buf) < k:
                                    buf.append(phrase)
                                else:
                                    j = _r.randrange(n)
                                    if j < k:
                                        buf[j] = phrase
                        self._text_aug_vg_pool = buf
                    except Exception:
                        self._text_aug_vg_pool = []

    def __len__(self) -> int:
        return len(self.metas)

    def _eligible_lvis_neg_cids_for_meta(self, meta: Dict[str, Any]) -> List[int]:
        if self.patch_bank is None:
            return []
        _, labels = self._extract_instances(meta)
        annotated = set(int(x) for x in labels.tolist())
        forbidden = set(int(x) for x in (meta.get("not_exhaustive_cids", []) or []))
        out: List[int] = []
        seen = set()
        for cid_raw in meta.get("neg_cids", []) or []:
            try:
                cid = int(cid_raw)
            except Exception:
                continue
            if cid in seen or cid in annotated or cid in forbidden:
                continue
            if len(self.patch_bank.get(cid, [])) <= 0:
                continue
            seen.add(cid)
            out.append(cid)
        return out

    def _filter_lvis_neg_category_metas_if_needed(self) -> None:
        if not bool(self.cfg.lvis_neg_category_only):
            return
        if self.patch_bank is None:
            raise ValueError("lvis_neg_category_only=True requires support_patch_tsv / patch_bank.")
        kept: List[Dict[str, Any]] = []
        for meta in self.metas:
            eligible_neg_cids = self._eligible_lvis_neg_cids_for_meta(meta)
            if not eligible_neg_cids:
                continue
            meta = dict(meta)
            meta["eligible_neg_cids"] = eligible_neg_cids
            kept.append(meta)
        if not kept:
            raise ValueError(
                "lvis_neg_category_only=True produced an empty subset. "
                "Check LVIS neg_category_ids, canonical class mapping, not_exhaustive filtering, and support patch bank coverage."
            )
        old_len = len(self.metas)
        self.metas = kept
        print(f"[INFO] LVIS neg_category_only subset: kept {len(self.metas)} / {old_len} images with patch-backed neg_category_ids.")

    def _iter_meta_tn_records(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        instances = meta.get("instances", None)
        if instances is None and isinstance(meta.get("detection", None), dict):
            instances = meta["detection"].get("instances", None)
        if instances is None and isinstance(meta.get("grounding", None), dict):
            instances = meta["grounding"].get("regions", None)
        if not instances:
            return []
        out = []
        for obj in instances:
            if not isinstance(obj, dict):
                continue
            if bool(obj.get("text_is_negative", obj.get("is_text_negative", False))) or bool(
                obj.get("sam3_tn_pair", False)
            ):
                out.append(obj)
        return out

    def _meta_primary_tn_group(self, meta: Dict[str, Any]) -> Optional[str]:
        for obj in self._iter_meta_tn_records(meta):
            categories = _as_list(obj.get("replace_category", None))
            if categories:
                return _tn_category_group(categories[0])
            return "other"
        return None

    def _compute_tn_category_stats(self) -> Dict[str, Any]:
        raw_counts: Counter = Counter()
        group_counts: Counter = Counter()
        rows_with_category = 0
        total_edits = 0
        tn_rows = 0
        for meta in self.metas:
            row_has_category = False
            for obj in self._iter_meta_tn_records(meta):
                tn_rows += 1
                categories = _as_list(obj.get("replace_category", None))
                replace_from_values = _as_list(obj.get("replace_from", None))
                replace_to_values = _as_list(obj.get("replace_to", None))
                num_edits = max(len(replace_from_values), len(replace_to_values), len(categories))
                if categories:
                    row_has_category = True
                for ridx in range(num_edits):
                    category = (
                        categories[ridx]
                        if ridx < len(categories)
                        else (categories[-1] if categories else "")
                    )
                    raw = _normalize_tn_category(category)
                    group = _tn_category_group(raw)
                    if raw:
                        raw_counts[raw] += 1
                    group_counts[group] += 1
                total_edits += int(num_edits)
            if row_has_category:
                rows_with_category += 1
        return {
            "tn_rows": int(tn_rows),
            "rows_with_category": int(rows_with_category),
            "total_edits": int(total_edits),
            "raw_category_counts": dict(raw_counts.most_common()),
            "normalized_group_counts": {name: int(group_counts.get(name, 0)) for name in _TN_GROUP_NAMES},
        }

    def _build_tn_balanced_sample_weights(self) -> Optional[List[float]]:
        if not bool(self.cfg.tn_balance_sampling):
            return None
        groups = [self._meta_primary_tn_group(meta) for meta in self.metas]
        counts = Counter(g for g in groups if g is not None)
        if not counts:
            self.tn_balance_stats = {"enabled": False, "reason": "no_tn_rows"}
            return None
        max_count = max(counts.values())
        cap = max(1.0, float(self.cfg.tn_balance_cap))
        group_weights = {
            group: min(cap, math.sqrt(float(max_count) / max(1.0, float(count))))
            for group, count in counts.items()
        }
        weights = [float(group_weights.get(group, 1.0)) if group is not None else 1.0 for group in groups]
        mean_weight = sum(weights) / max(1, len(weights))
        if mean_weight > 0:
            weights = [w / mean_weight for w in weights]
        self.tn_balance_stats = {
            "enabled": True,
            "cap": cap,
            "group_counts": {name: int(counts.get(name, 0)) for name in _TN_GROUP_NAMES},
            "group_weights": {name: float(group_weights.get(name, 1.0)) for name in _TN_GROUP_NAMES},
        }
        if counts:
            print("[INFO] TN balanced sampling for {}:\n{}".format(self.anno, json.dumps(self.tn_balance_stats, indent=2)))
        return weights

    def _build_native_patch_category_sample_weights(
        self,
    ) -> Optional[List[float]]:
        if self.cfg.native_patch_category_variant != "d2":
            self.native_patch_category_sampling_stats = {
                "enabled": False,
                "reason": "not_d2",
            }
            return None
        if bool(self.cfg.tn_balance_sampling):
            raise ValueError(
                "native patch-category D2 requires tn_balance_sampling=False"
            )
        weights: List[float] = []
        sources: Counter[str] = Counter()
        classes: Counter[int] = Counter()
        groups: set[str] = set()
        for row_index, meta in enumerate(self.metas):
            value = meta.get(_NATIVE_PATCH_CATEGORY_D2_WEIGHT_FIELD)
            if (
                not isinstance(value, float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"native patch-category D2 row {row_index} has an invalid sampling weight"
                )
            source = meta.get("native_patch_category_source_dataset")
            class_id = meta.get("native_patch_category_class_id")
            group_id = meta.get("native_patch_category_group_id")
            if (
                source not in {"refcoco", "refcocoplus", "refcocog"}
                or type(class_id) is not int
                or not isinstance(group_id, str)
                or not group_id
            ):
                raise ValueError(
                    f"native patch-category D2 row {row_index} lost sampling identity"
                )
            weights.append(float(value))
            sources[str(source)] += 1
            classes[int(class_id)] += 1
            groups.add(group_id)
        mean_weight = math.fsum(weights) / len(weights) if weights else 0.0
        if not weights or not math.isclose(
            mean_weight, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "native patch-category D2 per-source sampling weights must have exact mean one"
            )
        self.native_patch_category_sampling_stats = {
            "enabled": True,
            "contract": _NATIVE_PATCH_CATEGORY_D2_SAMPLING_CONTRACT,
            "rows": len(weights),
            "groups": len(groups),
            "classes": len(classes),
            "source_counts": dict(sorted(sources.items())),
            "weight_mean": mean_weight,
            "weight_min": min(weights),
            "weight_max": max(weights),
        }
        expected_source = self.cfg.native_patch_category_source_dataset
        if set(sources) != {expected_source}:
            raise ValueError(
                "native patch-category D2 rows do not match their bound source manifest"
            )
        print(
            "[INFO] Native patch-category D2 sampling for {}:\n{}".format(
                self.anno,
                json.dumps(
                    self.native_patch_category_sampling_stats,
                    indent=2,
                    sort_keys=True,
                ),
            )
        )
        return weights

    def _clean_caption_phrase(self, s: str) -> str:
        s = str(s).replace("_", " ").replace(".", " ").strip()
        s = " ".join(s.split())
        if not s:
            return "object"
        max_words = int(self.cfg.patch_text_aug_max_words)
        if (not self.cfg.build_text_token_masks) and max_words > 0:
            s = " ".join(s.split()[:max_words])
        return s

    def _sam3_tn_image_path(self, row: Dict[str, Any]) -> str:
        image_root = self.cfg.sam3_tn_image_root or self.root
        image_path = row.get("image_path", None)
        if isinstance(image_path, str) and image_path.strip() and Path(image_path).exists():
            return image_path
        image_id = row.get("image_id", None)
        if image_id is not None:
            try:
                return str(Path(image_root) / f"COCO_train2014_{int(image_id):012d}.jpg")
            except Exception:
                pass
        file_name = row.get("file_name", row.get("filename", None))
        if isinstance(file_name, str) and file_name.strip():
            return str(Path(image_root) / Path(file_name).name)
        return str(Path(image_root) / "")

    def _sam3_tn_bbox(self, row: Dict[str, Any]) -> Optional[List[float]]:
        preferred = str(self.cfg.sam3_tn_bbox_key or "sam_bbox")
        keys = [preferred]
        for key in ("sam_bbox", "bbox", "gt_bbox", "target_bbox_used"):
            if key not in keys:
                keys.append(key)
        for key in keys:
            value = row.get(key, None)
            if not isinstance(value, (list, tuple)) or len(value) != 4:
                continue
            try:
                box = [float(x) for x in value]
            except Exception:
                continue
            if box[2] <= 0 or box[3] <= 0:
                continue
            return box
        return None

    def _sam3_tn_canonical_id(self, row: Dict[str, Any], positive_phrase: str) -> Optional[int]:
        class_id = row.get("class_id", None)
        if class_id is not None:
            try:
                return int(class_id)
            except Exception:
                pass
        for key in ("class_norm_name", "category_name", "try_tn_head", "try_tn_head_phrase", "sent"):
            value = row.get(key, None)
            if isinstance(value, str) and value.strip():
                cid = self._phrase_to_canonical_id(value)
                if cid is not None:
                    return int(cid)
        return self._phrase_to_canonical_id(positive_phrase)

    def _normalize_sam3_tn_pair_metas(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        metas: List[Dict[str, Any]] = []
        skipped = 0
        table_b_id = getattr(self.cfg, "table_b_id", None)
        table_b_scope = getattr(self.cfg, "table_b_scope", None)
        table_b_audit_sha256 = getattr(
            self.cfg, "table_b_audit_sha256", None
        )
        for row in rows:
            if not isinstance(row, dict):
                skipped += 1
                continue
            if table_b_id is not None:
                if row.get("table_b_pair_schema") != TABLE_B_PAIR_SCHEMA:
                    raise ValueError(
                        "Table-B TN row has an unexpected pair schema at "
                        f"{row.get('sample_id', f'row {len(metas)}')}"
                    )
                if row.get("table_b_id") != table_b_id:
                    raise ValueError(
                        "Table-B TN row ID does not match the enabled dataset "
                        f"contract: {row.get('table_b_id')!r} != "
                        f"{table_b_id!r}"
                    )
                if row.get("tn_scope") != table_b_scope:
                    raise ValueError(
                        "Table-B TN row scope does not match the enabled dataset "
                        f"contract: {row.get('tn_scope')!r} != "
                        f"{table_b_scope!r}"
                    )
                if row.get("global_tn_verified") is not False:
                    raise ValueError(
                        "Table-B weak-scope rows must retain exact boolean "
                        "global_tn_verified=false"
                    )
            if self.cfg.require_single_edit_token_provenance:
                _validate_single_edit_token_provenance(
                    row,
                    context=str(row.get("sample_id", f"row {len(metas)}")),
                )
            tn_scope = row.get("tn_scope", None)
            semantic_global_verified = (
                row.get("global_tn_verified", None) is True
                and tn_scope == "image_global_topk_verified"
            )
            exact_summary = self._fixed_stagea_exact_rows.get(
                str(row.get("sample_id", ""))
            )
            exact_global_verified = (
                self.cfg.require_fixed_stagea_topk_exact_verified
                and exact_summary is not None
                and row.get("global_tn_verified", None) is True
                and row.get("fixed_stagea_topk_exact_verified", None) is True
                and tn_scope == EXACT_TOPK_TN_SCOPE
            )
            global_verified = semantic_global_verified or exact_global_verified
            proposalset_proxy_verified = (
                row.get("proposalset_proxy_verified", None) is True
                and tn_scope == "proposal_set_verified"
            )
            # This runtime flag is granted by the datasetinfo contract after
            # validating the immutable row, never by a row-authored boolean.
            token_supervision_valid = bool(
                self.cfg.require_single_edit_token_provenance
            )
            data_driven_trace = None
            if token_supervision_valid:
                data_driven_trace = dict(row["tn_edits"][0])
            candidate_trace_scope = str(
                row.get("candidate_trace_scope", "expression_only")
            ).strip().lower()
            if candidate_trace_scope not in {
                "expression_only",
                "target_only",
                "global_word_absent",
                "candidate_verified",
            }:
                raise ValueError(
                    "SAM3 TN row has an unknown candidate trace scope: "
                    f"{candidate_trace_scope!r}"
                )
            changed_word_global_absent_verified = (
                row.get("changed_word_global_absent_verified", None) is True
            )
            candidate_verified_indices = row.get(
                "changed_word_candidate_verified_indices", None
            )
            if candidate_trace_scope == "global_word_absent" and not (
                changed_word_global_absent_verified
            ):
                raise ValueError(
                    "GLOBAL_WORD_ABSENT trace scope requires exact changed-word "
                    "absence verification"
                )
            if candidate_trace_scope == "candidate_verified":
                if (
                    not isinstance(candidate_verified_indices, (list, tuple))
                    or not candidate_verified_indices
                    or any(
                        isinstance(value, bool) or int(value) < 0
                        for value in candidate_verified_indices
                    )
                ):
                    raise ValueError(
                        "CANDIDATE_VERIFIED trace scope requires non-negative "
                        "original-query indices"
                    )
                candidate_verified_indices = [
                    int(value) for value in candidate_verified_indices
                ]
            if bool(getattr(self.cfg, "require_global_tn_verified", False)) and not global_verified:
                raise ValueError(
                    "SAM3 TN row must carry exact boolean global_tn_verified=true "
                    "and tn_scope='image_global_topk_verified' while "
                    f"require_global_tn_verified is enabled: {self.anno}"
                )
            if (
                bool(self.cfg.require_fixed_stagea_topk_exact_verified)
                and not exact_global_verified
            ):
                raise ValueError(
                    "SAM3 TN row must carry the exact fixed Stage-A Top-K scope, "
                    "sidecar-validated provenance, and exact boolean "
                    "fixed_stagea_topk_exact_verified=true while the strict "
                    f"loader is enabled: {self.anno}"
                )
            if (
                bool(getattr(self.cfg, "require_proposalset_proxy_verified", False))
                and not proposalset_proxy_verified
            ):
                raise ValueError(
                    "SAM3 TN row must carry exact boolean "
                    "proposalset_proxy_verified=true and "
                    "tn_scope='proposal_set_verified' while "
                    f"require_proposalset_proxy_verified is enabled: {self.anno}"
                )
            raw_negative_phrase = row.get("try_tn", None)
            if not isinstance(raw_negative_phrase, str) or not raw_negative_phrase.strip():
                skipped += 1
                continue
            negative_phrase = self._clean_caption_phrase(raw_negative_phrase)
            if not negative_phrase or negative_phrase.lower() in {"object", "none", "null"}:
                skipped += 1
                continue
            # Train the verifier on the complete referring expression. The generated
            # head phrase may omit relation/action words that still appear in the TN.
            raw_positive_phrase = row.get("sent", None) or row.get("try_tn_head_phrase", None)
            if not isinstance(raw_positive_phrase, str) or not raw_positive_phrase.strip():
                skipped += 1
                continue
            positive_phrase = self._clean_caption_phrase(raw_positive_phrase)
            if not positive_phrase or positive_phrase.lower() in {"object", "none", "null"}:
                skipped += 1
                continue
            canonical_id = self._sam3_tn_canonical_id(row, positive_phrase)
            if canonical_id is None:
                skipped += 1
                continue
            bbox = self._sam3_tn_bbox(row)
            if bbox is None:
                skipped += 1
                continue
            instance = {
                "bbox": bbox,
                "class_id": int(canonical_id),
                "raw_phrase": positive_phrase,
                "phrase": positive_phrase,
                "head_phrase": row.get("class_norm_name", None)
                or row.get("category_name", None)
                or row.get("try_tn_head", None),
                "head": row.get("class_norm_name", None)
                or row.get("category_name", None)
                or row.get("try_tn_head", None),
                "canonical_name": row.get("class_norm_name", None) or row.get("category_name", None),
                "positive_phrase": positive_phrase,
                "negative_phrase": negative_phrase,
                "try_tn": negative_phrase,
                "try_tn_head": row.get("try_tn_head", None),
                "try_tn_head_phrase": positive_phrase,
                "replace_from": row.get("replace_from", None),
                "replace_to": row.get("replace_to", None),
                "replace_category": row.get("replace_category", None),
                "replace_span": row.get("replace_span", None),
                "text_is_negative": False,
                "pair_source": row.get("pair_source", None),
                "category_name": row.get("category_name", None),
                "visual_filter_status": row.get("visual_filter_status", None),
                "global_tn_verified": global_verified,
                "fixed_stagea_topk_exact_verified": exact_global_verified,
                "proposalset_proxy_verified": proposalset_proxy_verified,
                "stage_b_v21_token_supervision_valid": token_supervision_valid,
                "stage_b_candidate_trace_scope": candidate_trace_scope,
                "stage_b_changed_word_global_absent_verified": (
                    changed_word_global_absent_verified
                ),
                "stage_b_changed_word_candidate_verified_indices": (
                    candidate_verified_indices
                ),
                "tn_scope": tn_scope,
                "sam3_tn_pair": True,
            }
            if table_b_id is not None:
                instance["table_b_id"] = table_b_id
                instance["table_b_audit_sha256"] = table_b_audit_sha256
            if exact_global_verified:
                instance["fixed_stagea_support_patch"] = row[
                    "fixed_stagea_support_patch"
                ]
            normalized_meta = {
                    "filename": self._sam3_tn_image_path(row),
                    "source": row.get("pair_source", row.get("dataset", self.source)),
                    "dataset_name": row.get("dataset", self.source),
                    "image_id": row.get("image_id", None),
                    "ann_id": row.get("ann_id", None),
                    "ref_id": row.get("ref_id", None),
                    "sent_id": row.get("sent_id", None),
                    "split": row.get("split", "train"),
                    "instances": [instance],
                    "sam3_tn_pair": True,
                    "global_tn_verified": global_verified,
                    "fixed_stagea_topk_exact_verified": exact_global_verified,
                    "proposalset_proxy_verified": proposalset_proxy_verified,
                    "stage_b_v21_token_supervision_valid": token_supervision_valid,
                    "stage_b_candidate_trace_scope": candidate_trace_scope,
                    "stage_b_changed_word_global_absent_verified": (
                        changed_word_global_absent_verified
                    ),
                    "stage_b_changed_word_candidate_verified_indices": (
                        candidate_verified_indices
                    ),
                    "stage_b_data_driven_trace": data_driven_trace,
                    "tn_scope": tn_scope,
                    **(
                        {
                            "fixed_stagea_candidate_indices": exact_summary[
                                "candidate_indices"
                            ],
                            "fixed_stagea_candidate_boxes": exact_summary[
                                "candidate_boxes"
                            ],
                            "fixed_stagea_candidate_box_atol": exact_summary[
                                "contract"
                            ]["candidate_box_atol"],
                            "fixed_stagea_candidate_set_sha256": row[
                                "fixed_stagea_candidate_set_sha256"
                            ],
                            "fixed_stagea_support_patch": row[
                                "fixed_stagea_support_patch"
                            ],
                        }
                        if exact_global_verified
                        else {}
                    ),
                }
            if table_b_id is not None:
                normalized_meta["table_b_id"] = table_b_id
                normalized_meta["table_b_audit_sha256"] = table_b_audit_sha256
            metas.append(normalized_meta)
        print(
            f"[INFO] normalized SAM3 paired TN metas for {self.anno}: "
            f"kept {len(metas)} / {len(rows)} rows, skipped {skipped}."
        )
        return metas

    def _sample_text_phrases(self, k: int) -> List[str]:
        k = max(1, int(k))
        if not bool(self.cfg.patch_text_augment):
            return ["object"] * k
        p_object = float(self.cfg.patch_text_aug_p_object)
        p_alias = float(self.cfg.patch_text_aug_p_alias)
        p_vg = float(self.cfg.patch_text_aug_p_vg)
        total = max(0.0, p_object) + max(0.0, p_alias) + max(0.0, p_vg)
        if total <= 0:
            return ["object"] * k
        r = random.random() * total
        if r < p_object:
            return ["object"] * k
        r -= p_object
        if r < p_alias and self._text_aug_alias_pool:
            return [self._clean_caption_phrase(random.choice(self._text_aug_alias_pool)) for _ in range(k)]
        if self._text_aug_vg_pool:
            return [self._clean_caption_phrase(random.choice(self._text_aug_vg_pool)) for _ in range(k)]
        return ["object"] * k

    def _warn_text_mask(self, message: str) -> None:
        if self._text_mask_warn_count >= int(self.cfg.text_mask_warn_limit):
            return
        self._text_mask_warn_count += 1
        print(f"[WARN] {message}")

    def _append_text_mask_audit(self, record: Dict[str, Any]) -> None:
        path = getattr(self.cfg, "text_mask_audit_jsonl", None)
        if not path:
            return
        audit_path = Path(path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload.setdefault("ts", time.time())
        with audit_path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _get_canonical_name(self, canonical_id: int) -> str:
        name = self.cid_to_name.get(int(canonical_id), None)
        if isinstance(name, str) and name.strip():
            return self._clean_caption_phrase(name)
        return "object"

    def _get_canonical_aliases(self, canonical_id: int) -> List[str]:
        aliases: List[str] = []
        seen = set()
        for alias in self.cid_to_aliases.get(int(canonical_id), []) or []:
            cleaned = self._clean_caption_phrase(alias)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            aliases.append(cleaned)
        return aliases

    def _build_caption_from_phrases(self, phrases: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
        caption_parts: List[str] = []
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for i, phrase in enumerate(phrases):
            if i > 0:
                caption_parts.append(" ")
                cursor += 1
            start = cursor
            caption_parts.append(phrase)
            cursor += len(phrase)
            spans.append((start, cursor))
            caption_parts.append(" .")
            cursor += 2
        return "".join(caption_parts), spans

    def _find_word_span(self, text: str, candidate: str, *, ignore_case: bool = False) -> Optional[Tuple[int, int]]:
        if not text or not candidate:
            return None
        flags = re.IGNORECASE if ignore_case else 0
        pat = re.compile(rf"(?<![0-9A-Za-z]){re.escape(candidate)}(?![0-9A-Za-z])", flags=flags)
        m = pat.search(text)
        if m is None:
            return None
        return m.start(), m.end()

    def _char_span_to_token_mask(
        self, tokenized, span: Tuple[int, int], max_text_len: int
    ) -> torch.Tensor:
        mask = torch.zeros((int(max_text_len),), dtype=torch.bool)
        start, end = int(span[0]), int(span[1])
        if end <= start:
            return mask

        beg_pos = tokenized.char_to_token(start)
        if beg_pos is None:
            for delta in (1, 2):
                if start + delta < end:
                    beg_pos = tokenized.char_to_token(start + delta)
                    if beg_pos is not None:
                        break
        end_pos = tokenized.char_to_token(end - 1)
        if end_pos is None:
            for delta in (2, 3):
                if end - delta >= start:
                    end_pos = tokenized.char_to_token(end - delta)
                    if end_pos is not None:
                        break
        if beg_pos is None or end_pos is None:
            return mask
        beg_pos = int(beg_pos)
        end_pos = int(end_pos)
        if beg_pos < 0 or end_pos < 0 or end_pos < beg_pos:
            return mask
        beg_pos = min(beg_pos, int(max_text_len) - 1)
        end_pos = min(end_pos, int(max_text_len) - 1)
        mask[beg_pos : end_pos + 1] = True
        return mask

    def _is_relation_like_category(self, category: str) -> bool:
        return _normalize_tn_category(category) in {
            "spatial",
            "spatial relation",
            "spatial position",
            "position",
            "location",
            "distance",
            "action",
            "posture",
            "pose",
        }

    def _tn_category_weight(self, category: str, changed_token_norms: List[str]) -> float:
        group = _tn_category_group(category)
        if not bool(self.cfg.use_tn_category_weights):
            return float(self.cfg.default_tn_category_weight)
        if group in _TN_CATEGORY_WEIGHTS:
            return float(_TN_CATEGORY_WEIGHTS[group])
        return float(self.cfg.default_tn_category_weight)

    def _is_visual_content_token(self, token_norm: str) -> bool:
        if not token_norm:
            return False
        if token_norm in _CONTENT_EXCLUDED_TOKENS:
            return False
        return True

    def _is_relation_action_token(self, token_norm: str) -> bool:
        return token_norm in _RELATION_ACTION_WORDS

    def _mask_from_phrase_local_spans(
        self,
        tokenized,
        phrase_span: Tuple[int, int],
        phrase_mask: torch.Tensor,
        local_spans: List[Tuple[int, int]],
        max_text_len: int,
    ) -> torch.Tensor:
        out = torch.zeros_like(phrase_mask)
        span_start = int(phrase_span[0])
        for local_start, local_end in local_spans:
            if int(local_end) <= int(local_start):
                continue
            out = out | self._char_span_to_token_mask(
                tokenized,
                (span_start + int(local_start), span_start + int(local_end)),
                max_text_len,
            )
        return out & phrase_mask

    def _find_token_subsequence_start(
        self,
        haystack_tokens: List[Dict[str, Any]],
        needle_tokens: List[Dict[str, Any]],
    ) -> Optional[int]:
        if not needle_tokens or len(needle_tokens) > len(haystack_tokens):
            return None
        hay = [t["norm"] for t in haystack_tokens]
        needle = [t["norm"] for t in needle_tokens]
        starts: List[int] = []
        n = len(needle)
        for i in range(0, len(hay) - n + 1):
            if hay[i : i + n] == needle:
                starts.append(i)
        if not starts:
            return None
        if len(starts) > 1 and bool(self.cfg.skip_ambiguous_tn):
            return None
        return int(starts[0])

    def _changed_attribute_token_spans(
        self,
        phrase_text: str,
        replace_from: Any,
        replace_to: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        from_text = _clean_for_alignment(replace_from)
        to_text = _clean_for_alignment(replace_to)
        from_tokens = _tokenize_with_offsets(from_text)
        to_tokens = _tokenize_with_offsets(to_text)
        if not to_tokens:
            return []

        from_norms = [t["norm"] for t in from_tokens]
        to_norms = [t["norm"] for t in to_tokens]
        changed_to_indices: List[int] = []
        for tag, _i1, _i2, j1, j2 in SequenceMatcher(None, from_norms, to_norms).get_opcodes():
            if tag in {"replace", "insert"}:
                changed_to_indices.extend(range(int(j1), int(j2)))
        if not changed_to_indices:
            return []

        phrase_text_clean = _clean_for_alignment(phrase_text)
        to_text_clean = _clean_for_alignment(to_text)
        local_to_span = self._find_word_span(phrase_text_clean, to_text_clean, ignore_case=False)
        if local_to_span is None:
            local_to_span = self._find_word_span(phrase_text_clean, to_text_clean, ignore_case=True)

        out: List[Dict[str, Any]] = []
        if local_to_span is not None:
            base = int(local_to_span[0])
            for idx in changed_to_indices:
                tok = to_tokens[idx]
                out.append(
                    {
                        "text": tok["text"],
                        "norm": tok["norm"],
                        "start": base + int(tok["start"]),
                        "end": base + int(tok["end"]),
                    }
                )
            return out

        phrase_tokens = _tokenize_with_offsets(phrase_text_clean)
        start_idx = self._find_token_subsequence_start(phrase_tokens, to_tokens)
        if start_idx is None:
            return None
        for idx in changed_to_indices:
            tok = phrase_tokens[start_idx + idx]
            out.append(
                {
                    "text": tok["text"],
                    "norm": tok["norm"],
                    "start": int(tok["start"]),
                    "end": int(tok["end"]),
                }
            )
        return out

    def _build_relation_token_mask(
        self,
        tokenized,
        phrase_text: str,
        phrase_span: Tuple[int, int],
        phrase_mask: torch.Tensor,
        max_text_len: int,
    ) -> torch.Tensor:
        spans = [
            (int(t["start"]), int(t["end"]))
            for t in _tokenize_with_offsets(phrase_text)
            if self._is_relation_action_token(str(t["norm"]))
        ]
        return self._mask_from_phrase_local_spans(tokenized, phrase_span, phrase_mask, spans, max_text_len)

    def _build_content_attr_mask(
        self,
        tokenized,
        phrase_text: str,
        phrase_span: Tuple[int, int],
        phrase_mask: torch.Tensor,
        canonical_mask: torch.Tensor,
        max_text_len: int,
    ) -> torch.Tensor:
        spans = []
        for tok in _tokenize_with_offsets(phrase_text):
            norm = str(tok["norm"])
            if not self._is_visual_content_token(norm):
                continue
            spans.append((int(tok["start"]), int(tok["end"])))
        mask = self._mask_from_phrase_local_spans(tokenized, phrase_span, phrase_mask, spans, max_text_len)
        return mask & (~canonical_mask)

    def _build_shared_attr_mask(
        self,
        tokenized,
        phrase_text: str,
        positive_phrase: Optional[str],
        phrase_span: Tuple[int, int],
        phrase_mask: torch.Tensor,
        canonical_mask: torch.Tensor,
        relation_mask: torch.Tensor,
        attr_neg_mask: torch.Tensor,
        max_text_len: int,
    ) -> torch.Tensor:
        if not isinstance(positive_phrase, str) or not positive_phrase.strip():
            return self._build_content_attr_mask(
                tokenized, phrase_text, phrase_span, phrase_mask, canonical_mask, max_text_len
            ) & (~attr_neg_mask)

        pos_tokens = _tokenize_with_offsets(positive_phrase)
        tn_tokens = _tokenize_with_offsets(phrase_text)
        pos_norms = [t["norm"] for t in pos_tokens]
        tn_norms = [t["norm"] for t in tn_tokens]
        spans: List[Tuple[int, int]] = []
        for tag, _i1, _i2, j1, j2 in SequenceMatcher(None, pos_norms, tn_norms).get_opcodes():
            if tag != "equal":
                continue
            for tok in tn_tokens[int(j1) : int(j2)]:
                norm = str(tok["norm"])
                if not self._is_visual_content_token(norm):
                    continue
                spans.append((int(tok["start"]), int(tok["end"])))
        mask = self._mask_from_phrase_local_spans(tokenized, phrase_span, phrase_mask, spans, max_text_len)
        return mask & (~canonical_mask) & (~attr_neg_mask)

    def _build_slot_text_masks(
        self,
        slot_phrases: List[str],
        slot_canonical_texts: List[str],
        slot_aliases: List[List[str]],
        slot_records: Optional[List[Dict[str, Any]]] = None,
    ):
        slot_phrases = [self._clean_caption_phrase(p) for p in slot_phrases]
        slot_canonical_texts = [self._clean_caption_phrase(p) for p in slot_canonical_texts]
        slot_records = list(slot_records or [{} for _ in slot_phrases])
        caption, slot_spans = self._build_caption_from_phrases(slot_phrases)
        if not self.cfg.build_text_token_masks:
            return (
                caption,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                [],
                [],
            )
        if self._text_tokenizer is None:
            raise RuntimeError("build_text_token_masks=True but tokenizer is not initialized.")

        tokenized = self._text_tokenizer(
            caption,
            truncation=True,
            max_length=int(self.cfg.max_text_len),
        )
        K = len(slot_phrases)
        T = int(self.cfg.max_text_len)
        phrase_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        canonical_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        attr_pos_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        attr_neg_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        relation_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        content_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        tn_group_ids = torch.full((K,), int(_TN_GROUP_TO_ID["other"]), dtype=torch.long)
        attr_neg_weight_mask = torch.zeros((K, T), dtype=torch.float32)
        is_tn = torch.zeros((K,), dtype=torch.bool)
        rank_positive_phrase_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        rank_positive_canonical_to_token_mask = torch.zeros((K, T), dtype=torch.bool)
        has_rank_positive = torch.zeros((K,), dtype=torch.bool)
        rank_positive_captions: List[Optional[str]] = [None for _ in range(K)]
        invalid_records: List[Dict[str, Any]] = []

        for k, ((span_start, _span_end), phrase_text, canonical_text, aliases) in enumerate(
            zip(slot_spans, slot_phrases, slot_canonical_texts, slot_aliases)
        ):
            record = slot_records[k] if k < len(slot_records) and isinstance(slot_records[k], dict) else {}
            is_text_negative = bool(record.get("text_is_negative", record.get("is_text_negative", False)))
            is_tn[k] = bool(is_text_negative)

            candidates: List[str] = []
            seen = set()
            for cand in [canonical_text] + list(aliases or []):
                cleaned = self._clean_caption_phrase(cand)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(cleaned)

            # Positive counterfactual scoring is an independent path.  It must
            # survive even when TN changed-span supervision below fails closed.
            positive_phrase = record.get("positive_phrase", None)
            positive_phrase_clean = self._clean_caption_phrase(positive_phrase)
            if positive_phrase_clean:
                positive_caption, positive_spans = self._build_caption_from_phrases(
                    [positive_phrase_clean]
                )
                positive_tokenized = self._text_tokenizer(
                    positive_caption,
                    truncation=True,
                    max_length=int(self.cfg.max_text_len),
                )
                positive_phrase_mask = self._char_span_to_token_mask(
                    positive_tokenized, positive_spans[0], T
                )
                positive_canonical_span = None
                for cand in candidates:
                    positive_canonical_span = self._find_word_span(
                        positive_phrase_clean, cand, ignore_case=False
                    )
                    if positive_canonical_span is not None:
                        break
                if positive_canonical_span is None:
                    for cand in candidates:
                        positive_canonical_span = self._find_word_span(
                            positive_phrase_clean, cand, ignore_case=True
                        )
                        if positive_canonical_span is not None:
                            break
                if positive_phrase_mask.any():
                    positive_canonical_mask = torch.zeros_like(
                        positive_phrase_mask
                    )
                    if positive_canonical_span is not None:
                        pos_abs_span = (
                            positive_spans[0][0]
                            + int(positive_canonical_span[0]),
                            positive_spans[0][0]
                            + int(positive_canonical_span[1]),
                        )
                        positive_canonical_mask = self._char_span_to_token_mask(
                            positive_tokenized, pos_abs_span, T
                        )
                        positive_canonical_mask = (
                            positive_canonical_mask & positive_phrase_mask
                        )
                    rank_positive_phrase_to_token_mask[k] = positive_phrase_mask
                    rank_positive_canonical_to_token_mask[k] = (
                        positive_canonical_mask
                    )
                    has_rank_positive[k] = True
                    rank_positive_captions[k] = positive_caption

            phrase_mask = self._char_span_to_token_mask(tokenized, slot_spans[k], T)
            phrase_to_token_mask[k] = phrase_mask
            if not phrase_mask.any():
                self._warn_text_mask(
                    f"phrase_to_token_mask is empty for slot={k} phrase={phrase_text!r} caption={caption!r}"
                )
                invalid_records.append(
                    {
                        "reason": "empty_phrase_to_token_mask",
                        "slot_idx": int(k),
                        "phrase": phrase_text,
                        "canonical": canonical_text,
                        "caption": caption,
                    }
                )
                continue

            local_span = None
            for cand in candidates:
                local_span = self._find_word_span(phrase_text, cand, ignore_case=False)
                if local_span is not None:
                    break
            if local_span is None:
                for cand in candidates:
                    local_span = self._find_word_span(phrase_text, cand, ignore_case=True)
                    if local_span is not None:
                        break

            canonical_mask = torch.zeros_like(phrase_mask)
            if local_span is None:
                self._warn_text_mask(
                    f"canonical_to_token_mask fallback to zero for slot={k} phrase={phrase_text!r} canonical={canonical_text!r}"
                )
                invalid_records.append(
                    {
                        "reason": "canonical_to_token_mask_fallback_zero",
                        "slot_idx": int(k),
                        "phrase": phrase_text,
                        "canonical": canonical_text,
                        "caption": caption,
                        "alias_candidates": candidates,
                    }
                )
            else:
                canonical_span = (span_start + int(local_span[0]), span_start + int(local_span[1]))
                canonical_mask = self._char_span_to_token_mask(tokenized, canonical_span, T)
                canonical_mask = canonical_mask & phrase_mask
                if not canonical_mask.any():
                    self._warn_text_mask(
                        f"canonical_to_token_mask is empty after tokenization for slot={k} phrase={phrase_text!r} canonical={canonical_text!r}"
                    )
                    invalid_records.append(
                        {
                            "reason": "canonical_to_token_mask_empty_after_tokenization",
                            "slot_idx": int(k),
                            "phrase": phrase_text,
                            "canonical": canonical_text,
                            "caption": caption,
                            "alias_candidates": candidates,
                        }
                    )
            canonical_to_token_mask[k] = canonical_mask

            relation_mask = self._build_relation_token_mask(tokenized, phrase_text, slot_spans[k], phrase_mask, T)
            relation_to_token_mask[k] = relation_mask

            if not is_text_negative:
                content_mask = self._build_content_attr_mask(
                    tokenized, phrase_text, slot_spans[k], phrase_mask, canonical_mask, T
                )
                content_to_token_mask[k] = content_mask
                attr_pos_to_token_mask[k] = content_mask
                continue

            replace_from_values = _as_list(record.get("replace_from", None))
            replace_to_values = _as_list(record.get("replace_to", None))
            replace_category_values = _as_list(record.get("replace_category", None))
            max_replacements = max(
                len(replace_from_values),
                len(replace_to_values),
                len(replace_category_values),
            )
            if max_replacements <= 0:
                continue

            slot_invalid = False
            neg_mask = torch.zeros((T,), dtype=torch.bool)
            neg_weight = torch.zeros((T,), dtype=torch.float32)
            for ridx in range(max_replacements):
                replace_from = replace_from_values[ridx] if ridx < len(replace_from_values) else ""
                replace_to = replace_to_values[ridx] if ridx < len(replace_to_values) else ""
                category = (
                    replace_category_values[ridx]
                    if ridx < len(replace_category_values)
                    else (replace_category_values[-1] if replace_category_values else "")
                )
                category_norm = _normalize_tn_category(category)
                group_name = _tn_category_group(category_norm)
                if ridx == 0:
                    tn_group_ids[k] = int(_TN_GROUP_TO_ID[group_name])

                changed_tokens = self._changed_attribute_token_spans(phrase_text, replace_from, replace_to)
                if changed_tokens is None:
                    if bool(self.cfg.skip_tn_if_changed_span_not_found):
                        slot_invalid = True
                        invalid_records.append(
                            {
                                "reason": "tn_changed_span_not_found",
                                "tn_group": group_name,
                                "slot_idx": int(k),
                                "phrase": phrase_text,
                                "canonical": canonical_text,
                                "caption": caption,
                                "replace_from": replace_from,
                                "replace_to": replace_to,
                                "replace_category": category_norm,
                            }
                        )
                    continue

                filtered_tokens = []
                for tok in changed_tokens:
                    norm = str(tok.get("norm", ""))
                    if not self._is_visual_content_token(norm):
                        continue
                    filtered_tokens.append(tok)

                weight = self._tn_category_weight(category_norm, [str(t.get("norm", "")) for t in filtered_tokens])
                if weight <= 0.0:
                    continue
                if not filtered_tokens:
                    if bool(self.cfg.skip_tn_if_changed_span_empty_after_filter):
                        slot_invalid = True
                        invalid_records.append(
                            {
                                "reason": "tn_changed_span_empty_after_filter",
                                "tn_group": group_name,
                                "slot_idx": int(k),
                                "phrase": phrase_text,
                                "canonical": canonical_text,
                                "caption": caption,
                                "replace_from": replace_from,
                                "replace_to": replace_to,
                                "replace_category": category_norm,
                            }
                        )
                    continue

                token_spans = [(int(t["start"]), int(t["end"])) for t in filtered_tokens]
                changed_mask = self._mask_from_phrase_local_spans(
                    tokenized, slot_spans[k], phrase_mask, token_spans, T
                )
                if not changed_mask.any():
                    if bool(self.cfg.skip_tn_if_changed_span_empty_after_filter):
                        slot_invalid = True
                        invalid_records.append(
                            {
                                "reason": "tn_changed_mask_empty_after_tokenization",
                                "tn_group": group_name,
                                "slot_idx": int(k),
                                "phrase": phrase_text,
                                "canonical": canonical_text,
                                "caption": caption,
                                "replace_from": replace_from,
                                "replace_to": replace_to,
                                "replace_category": category_norm,
                            }
                        )
                    continue

                neg_mask = neg_mask | changed_mask
                neg_weight = torch.maximum(neg_weight, changed_mask.to(torch.float32) * float(weight))

            if slot_invalid:
                continue
            if (neg_mask & canonical_mask).any():
                if bool(self.cfg.skip_tn_if_neg_overlaps_canonical):
                    invalid_records.append(
                        {
                            "reason": "tn_neg_overlaps_canonical",
                            "tn_group": _TN_ID_TO_GROUP.get(int(tn_group_ids[k].item()), "other"),
                            "slot_idx": int(k),
                            "phrase": phrase_text,
                            "canonical": canonical_text,
                            "caption": caption,
                        }
                    )
                    continue
                neg_mask = neg_mask & (~canonical_mask)
                neg_weight = neg_weight * neg_mask.to(torch.float32)

            if not bool((neg_weight > 0).any().item()):
                # Keep masks empty when the changed span cannot produce valid content tokens.
                continue

            attr_neg_to_token_mask[k] = neg_mask
            attr_neg_weight_mask[k] = neg_weight
            content_mask = self._build_content_attr_mask(
                tokenized,
                phrase_text,
                slot_spans[k],
                phrase_mask,
                canonical_mask,
                T,
            )
            shared_positive_phrase = positive_phrase or record.get("try_tn_head_phrase", None)
            attr_pos_mask = self._build_shared_attr_mask(
                tokenized,
                phrase_text,
                shared_positive_phrase,
                slot_spans[k],
                phrase_mask,
                canonical_mask,
                relation_mask,
                neg_mask,
                T,
            )
            content_mask = content_mask | attr_pos_mask
            content_to_token_mask[k] = content_mask
            # Compatibility alias: attr_pos now means positive content tokens,
            # not only attribute words. TN negative tokens stay in attr_neg.
            attr_pos_to_token_mask[k] = content_mask & (~neg_mask)

        return (
            caption,
            phrase_to_token_mask,
            canonical_to_token_mask,
            attr_pos_to_token_mask,
            attr_neg_to_token_mask,
            relation_to_token_mask,
            content_to_token_mask,
            is_tn,
            attr_neg_weight_mask,
            tn_group_ids,
            rank_positive_phrase_to_token_mask,
            rank_positive_canonical_to_token_mask,
            has_rank_positive,
            rank_positive_captions,
            invalid_records,
        )

    def _load_patch_bank(self, tsv_path: Path) -> Dict[int, List[str]]:
        """
        Load a patch bank from a TSV file (e.g. emb_index_from_quality.tsv).
        Expected columns:
          - path: patch image path
          - class_id / canonical_class_id / class: support class id (int or int-like string)
          - bucket (optional): clean/borderline/bad
        Rows with non-int classes are skipped.
        """
        want_embedding = bool(self.cfg.support_patch_use_embedding)
        bank: Dict[int, List[str]] = {}
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for r in reader:
                p = r.get("path", None)
                emb_rel = r.get("emb_rel_path", None)
                if want_embedding:
                    if not emb_rel:
                        continue
                else:
                    if not p:
                        continue
                bucket = r.get("bucket", None)
                if self.cfg.support_patch_bucket and bucket and bucket != self.cfg.support_patch_bucket:
                    continue
                cls_raw = (
                    r.get("class_id", None)
                    or r.get("canonical_class_id", None)
                    or r.get("support_class", None)
                    or r.get("class", None)
                )
                if cls_raw is None:
                    continue
                try:
                    cls_id = int(cls_raw)
                except Exception:
                    name_key = _norm_text(str(cls_raw))
                    mapped = None
                    if self.patch_class_map is not None:
                        mapped = self.patch_class_map.get(name_key, None)
                    if mapped is None and self.name2cid:
                        mapped = self.name2cid.get(name_key, None)
                    if mapped is None:
                        continue
                    cls_id = int(mapped)
                if want_embedding:
                    emb_path = str(emb_rel)
                    if not os.path.isabs(emb_path):
                        emb_path = os.path.join(str(tsv_path.parent), emb_path)
                    bank.setdefault(cls_id, []).append(emb_path)
                else:
                    img_path = str(p)
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(str(tsv_path.parent), img_path)
                    bank.setdefault(cls_id, []).append(img_path)
        return bank

    @staticmethod
    def _patch_ce_positive_only_source_flag(source_name: Any) -> bool:
        norm = str(source_name or "").lower().replace("_", "").replace("-", "").replace("+", "plus")
        return any(tag in norm for tag in ("refcoco", "refcocoplus", "refcocog", "refexp"))

    def _default_patch_bank_cache_path(self, tsv_path: Path) -> Path:
        bucket = self.cfg.support_patch_bucket or "all"
        mode = "emb" if self.cfg.support_patch_use_embedding else "img"
        return Path(str(tsv_path) + f".bank.{bucket}.{mode}.pkl")

    def _default_anno_cache_path(self, anno_path: Path, src: str) -> Path:
        return Path(str(anno_path) + f".metas.{src}.pkl")

    def _load_metas_cached(
        self,
        anno_path: Path,
        src: str,
        *,
        lvis_image_root: Optional[str],
        coco_image_root: Optional[str],
        vg_image_roots: Optional[List[str]],
        anno_data: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cache the expensive `json.load` + meta-building for huge raw JSON sources (LVIS / VG region descriptions).
        """
        if not self.cfg.anno_cache:
            if anno_data is None:
                with anno_path.open("r", encoding="utf-8") as f:
                    anno_data = json.load(f)
            if src == "lvis":
                if not lvis_image_root:
                    raise ValueError("source='lvis' requires lvis_image_root.")
                return self._build_metas_from_lvis(anno_data, Path(lvis_image_root))
            if src == "coco":
                if not coco_image_root:
                    raise ValueError("source='coco' requires coco_image_root.")
                return self._build_metas_from_coco(anno_data, Path(coco_image_root))
            if src == "vg_region_descriptions":
                if not vg_image_roots:
                    raise ValueError("source='vg_region_descriptions' requires vg_image_roots (list).")
                return self._build_metas_from_vg_region_descriptions(anno_data, [Path(p) for p in vg_image_roots])
            raise ValueError(f"Unsupported cached source={src}")

        cache_path = (
            Path(self.cfg.anno_cache_path)
            if self.cfg.anno_cache_path
            else self._default_anno_cache_path(anno_path, src=src)
        )
        anno_mtime = anno_path.stat().st_mtime
        canonical_path = self._canonical_classes_json
        canonical_mtime = None
        if canonical_path:
            try:
                canonical_mtime = Path(canonical_path).stat().st_mtime
            except Exception:
                canonical_mtime = None

        meta = {
            "version": 5,
            "src": str(src),
            "anno_path": str(anno_path),
            "anno_mtime": anno_mtime,
            "canonical_classes_json": canonical_path,
            "canonical_mtime": canonical_mtime,
            "lvis_image_root": str(lvis_image_root) if lvis_image_root else None,
            "coco_image_root": str(coco_image_root) if coco_image_root else None,
            "vg_image_roots": [str(p) for p in (vg_image_roots or [])],
        }

        try:
            if cache_path.exists():
                with cache_path.open("rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and payload.get("meta") and payload.get("metas") is not None:
                    cached_meta = payload["meta"]
                    if cached_meta == meta:
                        print(f"[INFO] Loaded anno metas cache: {cache_path}")
                        return payload["metas"]
        except Exception as e:
            print(f"[WARN] Failed to load anno metas cache ({cache_path}), rebuilding: {e}")

        print(f"[INFO] Building metas from raw JSON (first run only; will cache): {anno_path} (src={src})")
        if anno_data is None:
            with anno_path.open("r", encoding="utf-8") as f:
                anno_data = json.load(f)

        if src == "lvis":
            if not lvis_image_root:
                raise ValueError("source='lvis' requires lvis_image_root.")
            metas = self._build_metas_from_lvis(anno_data, Path(lvis_image_root))
        elif src == "coco":
            if not coco_image_root:
                raise ValueError("source='coco' requires coco_image_root.")
            metas = self._build_metas_from_coco(anno_data, Path(coco_image_root))
        elif src == "vg_region_descriptions":
            if not vg_image_roots:
                raise ValueError("source='vg_region_descriptions' requires vg_image_roots (list).")
            metas = self._build_metas_from_vg_region_descriptions(anno_data, [Path(p) for p in vg_image_roots])
        else:
            raise ValueError(f"Unsupported cached source={src}")

        if self.cfg.anno_cache_write:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as f:
                    pickle.dump({"meta": meta, "metas": metas}, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[INFO] Wrote anno metas cache: {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to write anno metas cache ({cache_path}): {e}")
        return metas

    def _coerce_patch_bank_format(self, bank: Any) -> Dict[int, List[str]]:
        """
        Backward compatible loader for patch bank caches.

        - Current format: Dict[int, List[str]] (paths)
        - Legacy format:  Dict[int, List[{"img_path": str, "emb_path": Optional[str]}]]
        """
        if not isinstance(bank, dict):
            return {}
        want_embedding = bool(self.cfg.support_patch_use_embedding)
        out: Dict[int, List[str]] = {}
        for k, items in bank.items():
            try:
                cls_id = int(k)
            except Exception:
                continue
            if not isinstance(items, list) or not items:
                continue
            if all(isinstance(x, str) for x in items):
                paths = [str(x) for x in items if str(x)]
                if paths:
                    out[cls_id] = paths
                continue
            if all(isinstance(x, dict) for x in items):
                key = "emb_path" if want_embedding else "img_path"
                paths = []
                for it in items:
                    v = it.get(key, None)
                    if v:
                        paths.append(str(v))
                if paths:
                    out[cls_id] = paths
                continue
            # Best-effort for mixed types.
            key = "emb_path" if want_embedding else "img_path"
            paths = []
            for it in items:
                if isinstance(it, str) and it:
                    paths.append(it)
                elif isinstance(it, dict):
                    v = it.get(key, None)
                    if v:
                        paths.append(str(v))
            if paths:
                out[cls_id] = paths
        return out

    def _load_patch_bank_cached(self, tsv_path: Path) -> Dict[int, List[str]]:
        if not self.cfg.patch_bank_cache:
            return self._load_patch_bank_fast(tsv_path)

        cache_path = Path(self.cfg.patch_bank_cache_path) if self.cfg.patch_bank_cache_path else self._default_patch_bank_cache_path(tsv_path)
        tsv_mtime = tsv_path.stat().st_mtime
        canonical_path = self._canonical_classes_json
        canonical_mtime = None
        if canonical_path:
            try:
                canonical_mtime = Path(canonical_path).stat().st_mtime
            except Exception:
                canonical_mtime = None

        meta = {
            "version": 3,
            "tsv_path": str(tsv_path),
            "tsv_mtime": tsv_mtime,
            "bucket": self.cfg.support_patch_bucket,
            "use_embedding": self.cfg.support_patch_use_embedding,
            "max_per_class": int(self.cfg.support_patch_max_per_class),
            "support_patch_image_root": self.cfg.support_patch_image_root,
            "canonical_classes_json": canonical_path,
            "canonical_mtime": canonical_mtime,
            "patch_class_map_json": self._support_patch_class_map_json,
            "patch_class_map_mtime": None,
        }
        if self._support_patch_class_map_json:
            try:
                meta["patch_class_map_mtime"] = Path(self._support_patch_class_map_json).stat().st_mtime
            except Exception:
                meta["patch_class_map_mtime"] = None

        try:
            if cache_path.exists():
                with cache_path.open("rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and payload.get("meta") and payload.get("bank") is not None:
                    cached_meta = payload["meta"]
                    if (
                        cached_meta.get("version") == meta["version"]
                        and cached_meta.get("tsv_path") == meta["tsv_path"]
                        and cached_meta.get("tsv_mtime") == meta["tsv_mtime"]
                        and cached_meta.get("bucket") == meta["bucket"]
                        and cached_meta.get("use_embedding") == meta["use_embedding"]
                        and cached_meta.get("max_per_class") == meta["max_per_class"]
                        and cached_meta.get("support_patch_image_root") == meta["support_patch_image_root"]
                        and cached_meta.get("canonical_classes_json") == meta["canonical_classes_json"]
                        and cached_meta.get("canonical_mtime") == meta["canonical_mtime"]
                        and cached_meta.get("patch_class_map_json") == meta["patch_class_map_json"]
                        and cached_meta.get("patch_class_map_mtime") == meta["patch_class_map_mtime"]
                    ):
                        print(f"[INFO] Loaded patch bank cache: {cache_path}")
                        coerced = self._coerce_patch_bank_format(payload["bank"])
                        if coerced:
                            return coerced
        except Exception as e:
            print(f"[WARN] Failed to load patch bank cache ({cache_path}), rebuilding: {e}")

        bank = self._load_patch_bank_fast(tsv_path)
        if self.cfg.patch_bank_cache_write:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as f:
                    pickle.dump({"meta": meta, "bank": bank}, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[INFO] Wrote patch bank cache: {cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to write patch bank cache ({cache_path}): {e}")
        return bank

    def _load_patch_bank_fast(self, tsv_path: Path) -> Dict[int, List[str]]:
        """
        Fast TSV parser for large `emb_index_from_quality.tsv`.

        Returns a compact bank:
          - support_patch_use_embedding=True  -> Dict[class_id, List[emb_path]]
          - support_patch_use_embedding=False -> Dict[class_id, List[img_path]]
        """
        want_embedding = bool(self.cfg.support_patch_use_embedding)
        keep_per_class = int(self.cfg.support_patch_max_per_class)

        bank: Dict[int, List[str]] = {}
        seen_per_class: Dict[int, int] = {}
        class_cache: Dict[str, int] = {}

        print(
            f"[INFO] Building patch bank from TSV (first run only; will cache): {tsv_path} "
            f"(bucket={self.cfg.support_patch_bucket or 'all'}, mode={'emb' if want_embedding else 'img'})"
        )

        with tsv_path.open("r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            col = {k: i for i, k in enumerate(header)}

            idx_class = col.get("class_id", None)
            if idx_class is None:
                idx_class = col.get("canonical_class_id", None)
            if idx_class is None:
                idx_class = col.get("support_class", None)
            if idx_class is None:
                idx_class = col.get("class", None)
            if idx_class is None:
                raise ValueError(f"TSV missing required class column (class/class_id/...) ({tsv_path})")

            idx_bucket = col.get("bucket", None)
            idx_emb = col.get("emb_rel_path", None)
            idx_path = col.get("path", None)

            if want_embedding:
                if idx_emb is None:
                    raise ValueError(f"support_patch_use_embedding=True but TSV has no emb_rel_path column: {tsv_path}")
            else:
                if idx_path is None:
                    raise ValueError(f"support_patch_use_embedding=False but TSV has no path column: {tsv_path}")

            want_bucket = self.cfg.support_patch_bucket
            tsv_parent = str(tsv_path.parent)
            patch_img_root = self.cfg.support_patch_image_root

            for line_i, line in enumerate(f, start=1):
                parts = line.rstrip("\n").split("\t")
                if idx_bucket is not None and want_bucket is not None:
                    if parts[idx_bucket] != want_bucket:
                        continue

                cls_raw = parts[idx_class]
                if not cls_raw:
                    continue

                cls_id = class_cache.get(cls_raw, None)
                if cls_id is None:
                    try:
                        cls_id = int(cls_raw)
                    except Exception:
                        name_key = _norm_text(str(cls_raw))
                        mapped = None
                        if self.patch_class_map is not None:
                            mapped = self.patch_class_map.get(name_key, None)
                        if mapped is None and self.name2cid:
                            mapped = self.name2cid.get(name_key, None)
                        if mapped is None:
                            cls_id = -1
                        else:
                            cls_id = int(mapped)
                    class_cache[cls_raw] = int(cls_id)

                if int(cls_id) < 0:
                    continue

                if want_embedding:
                    emb_rel = parts[idx_emb]
                    if not emb_rel:
                        continue
                    path = str(emb_rel)
                    if not os.path.isabs(path):
                        path = os.path.join(tsv_parent, path)
                else:
                    path = ""
                    # Prefer the clean-image mirror rooted at `support_patch_image_root` when available:
                    # use `emb_rel_path` (e.g. clean/vg_patches/.../xxx.npy) -> clean/vg_patches/.../xxx.jpg
                    if patch_img_root and (idx_emb is not None):
                        emb_rel = parts[idx_emb]
                        if emb_rel:
                            cand = os.path.join(str(patch_img_root), str(emb_rel))
                            if cand.endswith(".npy"):
                                cand = cand[: -len(".npy")] + ".jpg"
                            if os.path.exists(cand):
                                path = cand
                    if not path:
                        p = parts[idx_path]
                        if not p:
                            continue
                        path = str(p)
                        if not os.path.isabs(path):
                            path = os.path.join(tsv_parent, path)

                if keep_per_class > 0:
                    seen = seen_per_class.get(int(cls_id), 0) + 1
                    seen_per_class[int(cls_id)] = seen
                    lst = bank.get(int(cls_id), None)
                    if lst is None:
                        lst = []
                        bank[int(cls_id)] = lst
                    if len(lst) < keep_per_class:
                        lst.append(path)
                    else:
                        j = random.randrange(seen)
                        if j < keep_per_class:
                            lst[j] = path
                else:
                    bank.setdefault(int(cls_id), []).append(path)

                if line_i % 500000 == 0:
                    print(f"[INFO] Patch bank parsing progress: {line_i} lines...")

        return bank

    def _build_metas_from_lvis(self, lvis_data: Dict[str, Any], image_root: Path) -> List[Dict[str, Any]]:
        cat_id_to_name = {int(c["id"]): str(c["name"]) for c in lvis_data.get("categories", [])}
        cat_id_to_cid: Dict[int, int] = {}
        for cat_id, name in cat_id_to_name.items():
            cid = self.name2cid.get(_norm_text(name), None) if self.name2cid else None
            if cid is not None:
                cat_id_to_cid[cat_id] = int(cid)

        img_id_to_not_exhaustive_cids: Dict[int, List[int]] = {}
        img_id_to_neg_cids: Dict[int, List[int]] = {}
        for img in lvis_data.get("images", []) or []:
            try:
                img_id = int(img["id"])
            except Exception:
                continue
            ne_ids = img.get("not_exhaustive_category_ids", None)
            if ne_ids:
                cids: List[int] = []
                seen = set()
                for cat_id in ne_ids:
                    try:
                        cat_id_int = int(cat_id)
                    except Exception:
                        continue
                    cid = cat_id_to_cid.get(cat_id_int, None)
                    if cid is None:
                        continue
                    if cid in seen:
                        continue
                    seen.add(cid)
                    cids.append(int(cid))
                if cids:
                    img_id_to_not_exhaustive_cids[img_id] = cids
            neg_ids = img.get("neg_category_ids", None)
            if neg_ids:
                neg_cids: List[int] = []
                neg_seen = set()
                for cat_id in neg_ids:
                    try:
                        cat_id_int = int(cat_id)
                    except Exception:
                        continue
                    cid = cat_id_to_cid.get(cat_id_int, None)
                    if cid is None:
                        continue
                    if cid in neg_seen:
                        continue
                    neg_seen.add(cid)
                    neg_cids.append(int(cid))
                if neg_cids:
                    img_id_to_neg_cids[img_id] = neg_cids

        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for a in lvis_data.get("annotations", []):
            img_id = int(a["image_id"])
            cat_id = int(a["category_id"])
            cid = cat_id_to_cid.get(cat_id, None)
            if cid is None:
                continue
            x, y, w, h = a["bbox"]
            inst = {"bbox": [x, y, x + w, y + h], "class_id": int(cid)}
            anns_by_img.setdefault(img_id, []).append(inst)

        metas: List[Dict[str, Any]] = []
        for img in lvis_data.get("images", []):
            img_id = int(img["id"])
            coco_url = img.get("coco_url", None)
            if isinstance(coco_url, str) and coco_url.strip():
                file_name = coco_url.strip().split("/")[-1]
            else:
                file_name = img.get("file_name", None) or f"{img_id:012d}.jpg"
            instances = anns_by_img.get(img_id, [])
            if not instances:
                continue
            abs_path = (image_root / file_name).resolve()
            if not abs_path.exists():
                continue
            meta = {"filename": str(abs_path), "instances": instances}
            # LVIS may include per-image "not_exhaustive_category_ids". For patch-only training we must
            # avoid choosing a support class that is marked non-exhaustive for this image, otherwise
            # positives/negatives become unreliable. Keep this internal-only (do not put into targets).
            ne_cids = img_id_to_not_exhaustive_cids.get(img_id, None)
            if ne_cids:
                meta["not_exhaustive_cids"] = ne_cids
            neg_cids = img_id_to_neg_cids.get(img_id, None)
            if neg_cids:
                meta["neg_cids"] = neg_cids
            metas.append(meta)
        return metas

    def _build_metas_from_coco(self, coco_data: Dict[str, Any], image_root: Path) -> List[Dict[str, Any]]:
        cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco_data.get("categories", [])}
        cat_id_to_cid: Dict[int, int] = {}
        for cat_id, name in cat_id_to_name.items():
            cid = self.name2cid.get(_norm_text(name), None) if self.name2cid else None
            if cid is not None:
                cat_id_to_cid[cat_id] = int(cid)

        anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
        for a in coco_data.get("annotations", []) or []:
            if int(a.get("iscrowd", 0)) == 1:
                continue
            img_id = int(a["image_id"])
            cat_id = int(a["category_id"])
            cid = cat_id_to_cid.get(cat_id, None)
            if cid is None:
                continue
            x, y, w, h = a["bbox"]
            inst = {"bbox": [x, y, x + w, y + h], "class_id": int(cid)}
            anns_by_img.setdefault(img_id, []).append(inst)

        metas: List[Dict[str, Any]] = []
        # Some COCO archives unpack as .../train2017/train2017/*.jpg; allow one nested folder fallback.
        nested_root = image_root / image_root.name
        for img in coco_data.get("images", []) or []:
            try:
                img_id = int(img["id"])
            except Exception:
                continue
            file_name = img.get("file_name", None) or f"{img_id:012d}.jpg"
            instances = anns_by_img.get(img_id, [])
            if not instances:
                continue
            p1 = image_root / file_name
            if p1.exists():
                metas.append({"filename": str(p1.resolve()), "instances": instances})
                continue
            p2 = nested_root / file_name
            if p2.exists():
                metas.append({"filename": str(p2.resolve()), "instances": instances})
                continue
        return metas

    def _build_metas_from_vg_region_descriptions(
        self, vg_data: List[Dict[str, Any]], image_roots: List[Path]
    ) -> List[Dict[str, Any]]:
        if not image_roots:
            raise ValueError("vg_image_roots is empty.")

        metas: List[Dict[str, Any]] = []
        for item in vg_data:
            img_id = item.get("id", None)
            if img_id is None:
                continue
            img_id_int = int(img_id)
            # Keep filename relative; _open_image will resolve against root and alt roots.
            img_path = f"{img_id_int}.jpg"

            instances: List[Dict[str, Any]] = []
            for r in item.get("regions", []) or []:
                try:
                    x = float(r["x"])
                    y = float(r["y"])
                    w = float(r["width"])
                    h = float(r["height"])
                except Exception:
                    continue
                # Store phrase and (optional) class_id; actual label resolution happens in _extract_instances.
                phrase = r.get("phrase", "")
                inst: Dict[str, Any] = {"bbox": [x, y, x + w, y + h], "phrase": phrase}
                if "class_id" in r:
                    try:
                        inst["class_id"] = int(r["class_id"])
                    except Exception:
                        pass
                instances.append(inst)

            if not instances:
                continue
            metas.append({"filename": str(img_path), "instances": instances})
        return metas

    def _extract_instance_records(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        instances = meta.get("instances", None)
        if instances is None and isinstance(meta.get("detection", None), dict):
            instances = meta["detection"].get("instances", None)
        if instances is None and isinstance(meta.get("grounding", None), dict):
            instances = meta["grounding"].get("regions", None)
        if not instances:
            return []

        records: List[Dict[str, Any]] = []
        for obj in instances:
            if "bbox" not in obj:
                continue
            cls = obj.get("class_id", obj.get("label", None))
            if cls is None:
                phrase = obj.get("raw_phrase", None) or obj.get("phrase", None) or obj.get("head_phrase", None)
                if isinstance(phrase, str) and phrase.strip():
                    cls = self._phrase_to_canonical_id(phrase)
            if cls is None:
                continue
            rec = {
                "bbox": obj["bbox"],
                "class_id": int(cls),
                "phrase": obj.get("raw_phrase", None) or obj.get("phrase", None),
                "head": obj.get("head", None),
                "head_phrase": obj.get("head_phrase", None)
                or obj.get("canonical_text", None)
                or obj.get("canonical_name", None),
                "text_is_negative": bool(
                    obj.get("text_is_negative", obj.get("is_text_negative", False))
                ),
            }
            for key in (
                "coco_ann_id",
                "refcoco_category_id",
                "category_complete_primary",
                "category_complete_auxiliary",
                "positive_phrase",
                "replace_from",
                "replace_to",
                "replace_category",
                "try_tn_head",
                "try_tn_head_phrase",
                "negative_phrase",
                "try_tn",
                "tn_type",
                "visual_filter_status",
                "global_tn_verified",
                "fixed_stagea_topk_exact_verified",
                "fixed_stagea_support_patch",
                "proposalset_proxy_verified",
                "stage_b_v21_token_supervision_valid",
                "benchmark_dataft_alltn",
                "tn_scope",
                "replace_span",
                "sam3_tn_pair",
                "sam3_tn_pair_positive",
                "sam3_tn_pair_negative",
            ):
                if key in obj:
                    rec[key] = obj.get(key)
            records.append(rec)

        return records

    def _extract_instances(self, meta: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        records = self._extract_instance_records(meta)
        if not records:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        boxes = [r["bbox"] for r in records]
        labels = [int(r["class_id"]) for r in records]

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64).reshape(-1)

        if self.cfg.box_format == "xywh":
            boxes_t = _xywh_to_xyxy(boxes_t)
        elif self.cfg.box_format != "xyxy":
            raise ValueError(f"Unsupported box_format: {self.cfg.box_format}")

        return boxes_t, labels_t

    @staticmethod
    def _resolve_primary_support_instance(
        meta: Dict[str, Any],
        labels: torch.Tensor,
        forbidden_classes: Optional[set[int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Resolve an explicitly designated expression-bearing support instance."""
        raw_index = meta.get("primary_support_instance_index", None)
        if raw_index is None:
            return None
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("primary_support_instance_index must be an integer")
        support_i = int(raw_index)
        if support_i < 0 or support_i >= int(labels.numel()):
            raise ValueError(
                "primary_support_instance_index is outside the instance list: "
                f"index={support_i}, instances={int(labels.numel())}"
            )
        support_class = int(labels[support_i].item())
        if support_class in (forbidden_classes or set()):
            raise ValueError(
                "primary_support_instance_index selects a forbidden class: "
                f"class_id={support_class}"
            )
        return support_class, support_i

    def _get_slot_text_for_record(self, record: Dict[str, Any], canonical_id: int) -> Tuple[str, str, List[str]]:
        canonical_source = record.get("head", None)
        if bool(record.get("text_is_negative", record.get("is_text_negative", False))):
            canonical_source = record.get("try_tn_head", None) or canonical_source
        canonical_source = canonical_source or record.get("head_phrase", None) or self._get_canonical_name(canonical_id)
        canonical_text = self._clean_caption_phrase(canonical_source)
        phrase_text = record.get("phrase", None)
        if isinstance(phrase_text, str) and phrase_text.strip():
            phrase_text = self._clean_caption_phrase(phrase_text)
        else:
            phrase_text = canonical_text

        alias_candidates: List[str] = []
        seen = set()
        for cand in [
            canonical_text,
            record.get("head", None),
            record.get("try_tn_head", None),
            record.get("head_phrase", None),
        ] + self._get_canonical_aliases(canonical_id):
            cleaned = self._clean_caption_phrase(cand)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            alias_candidates.append(cleaned)
        return phrase_text, canonical_text, alias_candidates

    def _paired_tn_slot_payload(
        self,
        slot_record: Dict[str, Any],
        phrase_text: str,
        canonical_text: str,
        alias_candidates: List[str],
    ) -> Optional[Tuple[List[str], List[str], List[List[str]], List[Dict[str, Any]], List[bool]]]:
        if not bool(slot_record.get("sam3_tn_pair", False)):
            return None
        raw_negative_phrase = slot_record.get("negative_phrase", slot_record.get("try_tn", ""))
        if not isinstance(raw_negative_phrase, str) or not raw_negative_phrase.strip():
            return None
        negative_phrase = self._clean_caption_phrase(raw_negative_phrase)
        if not negative_phrase or negative_phrase.lower() in {"object", "none", "null"}:
            return None
        pos_record = dict(slot_record)
        pos_record.update(
            {
                "phrase": phrase_text,
                "raw_phrase": phrase_text,
                "head": canonical_text,
                "head_phrase": canonical_text,
                "text_is_negative": False,
                "is_text_negative": False,
                "positive_phrase": phrase_text,
                "sam3_tn_pair_positive": True,
            }
        )
        neg_record = dict(slot_record)
        neg_record.update(
            {
                "phrase": negative_phrase,
                "raw_phrase": negative_phrase,
                "text_is_negative": True,
                "is_text_negative": True,
                "positive_phrase": phrase_text,
                "try_tn_head_phrase": phrase_text,
                "sam3_tn_pair_negative": True,
            }
        )
        return (
            [phrase_text, negative_phrase],
            [canonical_text, canonical_text],
            [list(alias_candidates), list(alias_candidates)],
            [pos_record, neg_record],
            [False, True],
        )

    def _get_phrase_classifier(self) -> _PhraseClassifierLabeler:
        if self._phrase_cls_labeler is not None:
            return self._phrase_cls_labeler
        if not self.cfg.phrase_classifier_ckpt:
            raise RuntimeError("phrase_classifier_ckpt is not set.")
        worker = torch.utils.data.get_worker_info()
        if worker is not None and str(self.cfg.phrase_classifier_device).startswith("cuda"):
            raise RuntimeError(
                "phrase_classifier_device is CUDA but DataLoader is using workers; "
                "set --num_workers 0 (or run phrase labeling offline) to avoid one GPU model per worker."
            )
        self._phrase_cls_labeler = _PhraseClassifierLabeler(
            ckpt_path=self.cfg.phrase_classifier_ckpt,
            device=self.cfg.phrase_classifier_device,
            max_length=self.cfg.phrase_classifier_max_length,
            batch_size=self.cfg.phrase_classifier_batch_size,
            min_conf=self.cfg.phrase_classifier_min_conf,
        )
        return self._phrase_cls_labeler

    def _phrase_to_canonical_id(self, phrase: str) -> Optional[int]:
        phrase_key = _norm_text(phrase)
        if not phrase_key:
            return None

        if phrase_key in self._phrase_cache:
            v = self._phrase_cache.pop(phrase_key)
            self._phrase_cache[phrase_key] = v
            return v

        labeler = (self.cfg.vg_phrase_labeler or "prefix").lower()

        cid: Optional[int] = None
        if labeler in {"prefix", "hybrid"}:
            cid = self.phrase_matcher.match_prefix(phrase) if self.phrase_matcher else None

        if cid is None and labeler in {"classifier", "hybrid"}:
            # classifier cache (separate because it may keep conf scores)
            if phrase_key in self._phrase_cls_cache:
                pred, _conf = self._phrase_cls_cache.pop(phrase_key)
                self._phrase_cls_cache[phrase_key] = (pred, _conf)
                cid = pred
            else:
                clf = self._get_phrase_classifier()
                pred, conf = clf.predict_top1([phrase])[0]
                self._phrase_cls_cache[phrase_key] = (pred, conf)
                if len(self._phrase_cls_cache) > int(self.cfg.phrase_cache_size):
                    self._phrase_cls_cache.popitem(last=False)
                cid = pred

        self._phrase_cache[phrase_key] = cid
        if len(self._phrase_cache) > int(self.cfg.phrase_cache_size):
            self._phrase_cache.popitem(last=False)
        return cid

    def _open_image(self, rel_path: str) -> Image.Image:
        rel_p = Path(rel_path)
        abs_path = rel_p if rel_p.is_absolute() else (Path(self.root) / rel_p)
        abs_path = remap_legacy_path(abs_path)
        if not abs_path.exists() and (not rel_p.is_absolute()) and self._alt_image_roots:
            for r in self._alt_image_roots:
                cand = r / rel_p
                if cand.exists():
                    abs_path = cand
                    break
        if not abs_path.exists():
            raise FileNotFoundError(str(abs_path))
        return Image.open(abs_path).convert("RGB")

    def _sample_support_from_image(
        self, img: Image.Image, boxes_xyxy: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        support_class, support_i = self._choose_support(labels, forbidden_classes=None)
        crop = _safe_crop(img, boxes_xyxy[support_i])
        if crop is None:
            # Fallback: center crop of the whole image.
            crop = img
        patch = self.patch_tfm(crop)
        return torch.as_tensor([support_class], dtype=torch.int64), patch

    def _choose_support(self, labels: torch.Tensor, forbidden_classes: Optional[set[int]] = None) -> Tuple[int, int]:
        forbidden = forbidden_classes or set()
        counts = Counter(labels.tolist())
        candidates = [c for c, cnt in counts.items() if (cnt >= self.cfg.support_min_count and int(c) not in forbidden)]
        if not candidates:
            candidates = [int(c) for c in counts.keys() if int(c) not in forbidden]
        if not candidates:
            raise RuntimeError("No eligible support_class after excluding forbidden_classes (e.g. not_exhaustive).")
        support_class = int(random.choice(candidates))
        idxs = (labels == support_class).nonzero(as_tuple=False).flatten()
        support_i = int(idxs[torch.randint(len(idxs), (1,)).item()].item())
        return support_class, support_i

    def _choose_support_classes(self, labels: torch.Tensor, forbidden_classes: Optional[set[int]] = None) -> List[int]:
        """
        Choose a set of distinct support classes for multi-patch episodes.
        Returns canonical class_ids.
        """
        forbidden = forbidden_classes or set()
        counts = Counter(labels.tolist())
        use_all_gt_classes = bool(self.cfg.support_use_all_gt_classes)
        if use_all_gt_classes:
            candidates = [int(c) for c in sorted(counts.keys()) if int(c) not in forbidden]
        else:
            candidates = [
                int(c)
                for c, cnt in counts.items()
                if (cnt >= self.cfg.support_min_count and int(c) not in forbidden)
            ]
            if not candidates:
                candidates = [int(c) for c in counts.keys() if int(c) not in forbidden]
        if self.patch_bank is not None and ((not use_all_gt_classes) or self.cfg.support_patch_use_embedding):
            candidates = [c for c in candidates if len(self.patch_bank.get(int(c), [])) > 0]
        if not candidates:
            raise RuntimeError("No eligible support classes for multi-patch episode.")

        if use_all_gt_classes:
            return candidates

        k_max = min(int(self.cfg.support_num_patches_max), len(candidates))
        k_min = min(max(1, int(self.cfg.support_num_patches_min)), k_max)
        k = random.randint(k_min, k_max) if k_min < k_max else k_max
        return random.sample(candidates, k=k)

    def _choose_lvis_neg_support_classes(
        self,
        labels: torch.Tensor,
        neg_cids: Any,
        forbidden_classes: Optional[set[int]] = None,
    ) -> List[int]:
        """
        Choose LVIS verified-negative classes that are absent from current annotations.
        These slots intentionally have no positive GT in the target.
        """
        if self.patch_bank is None:
            raise RuntimeError("LVIS neg_category_only episodes require a support patch bank.")
        forbidden = forbidden_classes or set()
        annotated = set(int(x) for x in labels.tolist())
        seen = set()
        candidates: List[int] = []
        for cid_raw in neg_cids or []:
            try:
                cid = int(cid_raw)
            except Exception:
                continue
            if cid in seen or cid in annotated or cid in forbidden:
                continue
            if self.patch_bank is not None and len(self.patch_bank.get(int(cid), [])) <= 0:
                continue
            seen.add(cid)
            candidates.append(cid)
        if not candidates:
            raise RuntimeError("No eligible LVIS neg_category_ids for negative support episode.")
        if int(self.cfg.support_num_patches_max) > 0:
            candidates = candidates[: int(self.cfg.support_num_patches_max)]
        return candidates

    def _sample_support_patch_for_class(
        self,
        support_class: int,
        fallback_img: Optional[Image.Image] = None,
        fallback_boxes_xyxy: Optional[torch.Tensor] = None,
        fallback_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.stage_b_gdino_adapter_no_support:
            # The ordinary GDINO adapter never consumes support pixels.  Keep a
            # private sentinel only long enough to reuse the episode text/box
            # construction; it is removed from the returned target below.
            return torch.empty((0,), dtype=torch.float32)
        if self.patch_bank is not None:
            candidates = self.patch_bank.get(int(support_class), [])
            if candidates:
                if self.cfg.support_patch_use_embedding:
                    emb_path = random.choice(candidates)
                    return self._load_patch_embedding(str(emb_path))
                img_path = random.choice(candidates)
                img = Image.open(str(img_path)).convert("RGB")
                return self.patch_tfm(img)
        if fallback_img is None or fallback_boxes_xyxy is None or fallback_labels is None:
            raise RuntimeError("No patch_bank entry and no fallback crop inputs were provided.")
        _, patch = self._sample_support_from_image(fallback_img, fallback_boxes_xyxy, fallback_labels)
        return patch

    def _load_native_patch_category_support(
        self, meta: Dict[str, Any], support_class: int
    ) -> torch.Tensor:
        if not self.cfg.native_patch_category_row_locked_support:
            raise RuntimeError("native patch-category row-locked support is disabled")
        witness = meta.get("support_patch_witness")
        if not isinstance(witness, dict) or witness.get("class_id") != int(
            support_class
        ):
            raise RuntimeError("native patch-category support class drifted")
        path = Path(str(witness.get("path", ""))).expanduser().resolve(strict=True)
        if not path.is_file():
            raise RuntimeError(
                f"native patch-category support is not a file: {path}"
            )
        before = path.stat()
        expected_size = witness.get("size_bytes")
        expected_sha = witness.get("content_sha256")
        if before.st_size != expected_size:
            raise RuntimeError(
                f"native patch-category support size drifted: {path}"
            )
        cache_key = (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
        )
        cached = self._native_patch_support_sha_cache.get(path)
        if cached is not None and cached[:4] == cache_key:
            observed_sha = cached[4]
        else:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            observed_sha = digest.hexdigest()
            after = path.stat()
            after_key = (
                int(after.st_dev),
                int(after.st_ino),
                int(after.st_size),
                int(after.st_mtime_ns),
            )
            if after_key != cache_key:
                raise RuntimeError(
                    f"native patch-category support changed while hashing: {path}"
                )
            self._native_patch_support_sha_cache[path] = (
                cache_key[0],
                cache_key[1],
                cache_key[2],
                cache_key[3],
                observed_sha,
            )
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"native patch-category support content hash drifted: {path}"
            )
        with Image.open(path) as image:
            return self.patch_tfm(image.convert("RGB"))

    def _load_fixed_stagea_support_patch(
        self, support_record: Optional[Dict[str, Any]], support_class: int
    ) -> torch.Tensor:
        if not isinstance(support_record, dict):
            raise RuntimeError("fixed Stage-A exact row has no support record")
        value = support_record.get("fixed_stagea_support_patch")
        if not isinstance(value, dict):
            raise RuntimeError("fixed Stage-A exact row has no support patch binding")
        if int(value.get("class_id", -1)) != int(support_class):
            raise RuntimeError("fixed Stage-A support patch class drifted at runtime")
        expected_contract = self.cfg.fixed_stagea_topk_expected_contract
        if (
            not isinstance(expected_contract, dict)
            or value.get("transform_contract_sha256")
            != expected_contract.get("support_transform_contract_sha256")
        ):
            raise RuntimeError("fixed Stage-A support transform contract drifted")
        path = remap_legacy_path(str(value.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"fixed Stage-A support patch is missing: {path}")
        before = path.stat()
        cache_key = (int(before.st_size), int(before.st_mtime_ns))
        cached = self._fixed_support_patch_sha_cache.get(path)
        if cached is not None and cached[:2] == cache_key:
            observed_sha = cached[2]
        else:
            observed_sha = sha256_file(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (
                before.st_size,
                before.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"fixed Stage-A support patch changed while hashing: {path}"
                )
            self._fixed_support_patch_sha_cache[path] = (
                cache_key[0],
                cache_key[1],
                observed_sha,
            )
        if observed_sha != value.get("sha256"):
            raise RuntimeError(f"fixed Stage-A support patch hash drifted: {path}")
        with Image.open(path) as image:
            return self.patch_tfm(image.convert("RGB"))

    def _load_patch_embedding(self, emb_path: str) -> torch.Tensor:
        if emb_path in self._patch_emb_cache:
            v = self._patch_emb_cache.pop(emb_path)
            self._patch_emb_cache[emb_path] = v
            return v
        arr = np.load(emb_path)
        emb = torch.from_numpy(arr).to(torch.float32)
        if emb.dim() != 1:
            emb = emb.view(-1)
        self._patch_emb_cache[emb_path] = emb
        if len(self._patch_emb_cache) > int(self.cfg.patch_emb_cache_size):
            self._patch_emb_cache.popitem(last=False)
        return emb

    def _sample_negative_support(
        self, query_labels: torch.Tensor, forbidden_classes: Optional[set[int]] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # Sample a support patch from another image, making sure its class does not
        # appear in the query image labels (and optionally not in forbidden_classes).
        query_set = set(query_labels.tolist())
        forbidden = forbidden_classes or set()
        if self.patch_bank is not None:
            keys = [
                k
                for k in self.patch_bank.keys()
                if (k not in query_set and k not in forbidden and len(self.patch_bank.get(k, [])) > 0)
            ]
            if keys:
                support_class = int(random.choice(keys))
                patch = self._sample_support_patch_for_class(support_class)
                return torch.as_tensor([support_class], dtype=torch.int64), patch

        if len(self.metas) <= 1:
            return None
        for _ in range(self.cfg.negative_max_tries):
            j = random.randrange(0, len(self.metas))
            meta_j = self.metas[j]
            rel_j = meta_j.get("filename", meta_j.get("file_name", None))
            if rel_j is None:
                continue
            boxes_j, labels_j = self._extract_instances(meta_j)
            if labels_j.numel() == 0:
                continue
            support_class = int(labels_j[torch.randint(labels_j.numel(), (1,)).item()].item())
            if support_class in query_set or support_class in forbidden:
                continue
            img_j = self._open_image(rel_j)
            idxs = (labels_j == support_class).nonzero(as_tuple=False).flatten()
            support_i = int(idxs[torch.randint(len(idxs), (1,)).item()].item())
            crop = _safe_crop(img_j, boxes_j[support_i])
            if crop is None:
                continue
            patch = self.patch_tfm(crop)
            return torch.as_tensor([support_class], dtype=torch.int64), patch
        return None

    def __getitem__(self, index: int):
        # Some LVIS images mark certain category_ids as "not_exhaustive" (not guaranteed fully annotated).
        # For patch-only training we must skip episodes where support_class is in that list.
        # Formal data-driven runs bind the sampler ledger to the requested row.
        # A bad row must fail instead of silently returning another identity.
        max_resample = 1 if self.cfg.strict_sample_identity else 20
        for attempt in range(max_resample):
            meta = self.metas[index] if attempt == 0 else self.metas[random.randrange(0, len(self.metas))]
            rel_path = meta.get("filename", meta.get("file_name", None))
            if rel_path is None:
                continue

            not_exhaustive = set(int(x) for x in (meta.get("not_exhaustive_cids", []) or []))
            lvis_neg_category_only = bool(self.cfg.lvis_neg_category_only)

            try:
                img = self._open_image(rel_path)
            except FileNotFoundError:
                continue
            w, h = img.size
            instance_records = self._extract_instance_records(meta)
            if not instance_records:
                continue
            boxes_xyxy, labels = self._extract_instances(meta)
            primary_support = self._resolve_primary_support_instance(
                meta, labels, forbidden_classes=not_exhaustive
            )
            primary_instance_mask = torch.zeros_like(labels, dtype=torch.bool)
            if primary_support is not None:
                primary_instance_mask[int(primary_support[1])] = True
            if primary_support is not None and int(self.cfg.support_num_patches_max) > 1:
                raise ValueError(
                    "primary_support_instance_index currently requires a single support patch"
                )

            # Dummy/augmented caption to keep the standard text pipeline intact.
            # NOTE: mask builder uses "." as a separator; we repeat one phrase per patch so Stage A is
            # shape-compatible with Stage B (phrase/patch one-to-one).
            caption = "object ."

            is_negative = False
            support = None
            slot_phrases: List[str] = []
            slot_canonical_texts: List[str] = []
            slot_aliases: List[List[str]] = []
            slot_text_is_negative: List[bool] = []
            slot_records: List[Dict[str, Any]] = []
            support_record: Optional[Dict[str, Any]] = None

            if labels.numel() == 0:
                # Metas for LVIS/COCO should not have empty instances; if they do, just resample.
                continue
            else:
                if self.cfg.native_patch_category_row_locked_support:
                    if primary_support is None:
                        raise RuntimeError(
                            "native patch-category row lost its primary support"
                        )
                    support_class, support_i = primary_support
                    patch = self._load_native_patch_category_support(
                        meta, support_class
                    )
                    support = (
                        torch.as_tensor([support_class], dtype=torch.int64),
                        patch,
                    )
                    support_record = instance_records[support_i]
                    is_negative = False
                elif int(self.cfg.support_num_patches_max) > 1:
                    # Multi-patch: choose multiple support classes (canonical ids) and sample one patch per class.
                    try:
                        if lvis_neg_category_only:
                            support_classes = self._choose_lvis_neg_support_classes(
                                labels,
                                meta.get("eligible_neg_cids", meta.get("neg_cids", [])),
                                forbidden_classes=not_exhaustive,
                            )
                        else:
                            support_classes = self._choose_support_classes(labels, forbidden_classes=not_exhaustive)
                    except Exception:
                        continue
                    support_classes_t = torch.as_tensor(support_classes, dtype=torch.int64)

                    patch_list: List[torch.Tensor] = []
                    ok = True
                    for cid in support_classes:
                        if lvis_neg_category_only:
                            try:
                                patch_c = self._sample_support_patch_for_class(int(cid))
                            except Exception:
                                ok = False
                                break
                            support_i = None
                        else:
                            idxs = (labels == int(cid)).nonzero(as_tuple=False).flatten()
                            if idxs.numel() == 0:
                                ok = False
                                break
                            support_i = int(idxs[torch.randint(len(idxs), (1,)).item()].item())
                            patch_c = self._sample_support_patch_for_class(
                                int(cid),
                                fallback_img=img,
                                fallback_boxes_xyxy=boxes_xyxy[support_i : support_i + 1],
                                fallback_labels=labels[support_i : support_i + 1],
                            )
                        patch_list.append(patch_c)
                        if lvis_neg_category_only:
                            canonical_text = self._get_canonical_name(int(cid))
                            phrase_text = canonical_text
                            alias_candidates = self._get_canonical_aliases(int(cid))
                            slot_record = {
                                "phrase": phrase_text,
                                "head_phrase": canonical_text,
                                "text_is_negative": False,
                            }
                        else:
                            slot_record = instance_records[int(support_i)]
                            phrase_text, canonical_text, alias_candidates = self._get_slot_text_for_record(
                                slot_record, int(cid)
                            )
                        paired_payload = self._paired_tn_slot_payload(
                            slot_record, phrase_text, canonical_text, alias_candidates
                        )
                        if paired_payload is not None:
                            pair_phrases, pair_canonical_texts, pair_aliases, pair_records, pair_is_negative = paired_payload
                            slot_phrases.extend(pair_phrases)
                            slot_canonical_texts.extend(pair_canonical_texts)
                            slot_aliases.extend(pair_aliases)
                            slot_text_is_negative.extend(pair_is_negative)
                            slot_records.extend(pair_records)
                        else:
                            slot_phrases.append(phrase_text)
                            slot_canonical_texts.append(canonical_text)
                            slot_aliases.append(alias_candidates)
                            slot_text_is_negative.append(bool(slot_record.get("text_is_negative", False)))
                            slot_records.append(slot_record)
                    if (not ok) or (len(patch_list) != int(support_classes_t.numel())):
                        continue

                    # Optionally keep only GT boxes belonging to the selected support classes.
                    if lvis_neg_category_only:
                        boxes_xyxy = boxes_xyxy.new_zeros((0, 4))
                        labels = labels.new_zeros((0,))
                    elif bool(self.cfg.keep_only_patchset_gt):
                        m = torch.zeros((labels.shape[0],), dtype=torch.bool)
                        for cid in support_classes:
                            m = m | (labels == int(cid))
                        boxes_xyxy = boxes_xyxy[m]
                        labels = labels[m]
                        if labels.numel() == 0:
                            continue

                    if self.cfg.support_patch_use_embedding:
                        patches_or_emb = torch.stack(patch_list, dim=0).to(torch.float32)  # (K,D)
                    else:
                        patches_or_emb = torch.stack(patch_list, dim=0)  # (K,3,S,S)
                    support = (support_classes_t, patches_or_emb)
                    is_negative = bool(lvis_neg_category_only)
                else:
                    if (self.cfg.neg_episode_prob > 0) and (random.random() < self.cfg.neg_episode_prob):
                        support = self._sample_negative_support(query_labels=labels, forbidden_classes=not_exhaustive)
                        if support is not None:
                            is_negative = True
                    if support is None:
                        if self.cfg.support_patch_use_embedding and (self.patch_bank is not None):
                            uniq = set(labels.tolist())
                            eligible = [
                                int(c)
                                for c in uniq
                                if (int(c) not in not_exhaustive and len(self.patch_bank.get(int(c), [])) > 0)
                            ]
                            if not eligible:
                                # No trustworthy positive class in this image; resample another image.
                                continue
                            else:
                                if primary_support is not None:
                                    support_class, support_i = primary_support
                                    if support_class not in eligible:
                                        raise ValueError(
                                            "primary support class has no eligible support patch: "
                                            f"class_id={support_class}"
                                        )
                                else:
                                    support_class = int(random.choice(eligible))
                                    idxs = (labels == support_class).nonzero(as_tuple=False).flatten()
                                    support_i = int(idxs[torch.randint(len(idxs), (1,)).item()].item())
                                patch = self._sample_support_patch_for_class(
                                    support_class,
                                    fallback_img=img,
                                    fallback_boxes_xyxy=boxes_xyxy[support_i : support_i + 1],
                                    fallback_labels=labels[support_i : support_i + 1],
                                )
                                support = (torch.as_tensor([support_class], dtype=torch.int64), patch)
                                support_record = instance_records[support_i]
                                is_negative = False
                        else:
                            if primary_support is not None:
                                support_class, support_i = primary_support
                            else:
                                try:
                                    support_class, support_i = self._choose_support(labels, forbidden_classes=not_exhaustive)
                                except Exception:
                                    continue
                            if self.cfg.require_fixed_stagea_topk_exact_verified:
                                support_record = instance_records[support_i]
                                patch = self._load_fixed_stagea_support_patch(
                                    support_record, support_class
                                )
                            else:
                                patch = self._sample_support_patch_for_class(
                                    support_class,
                                    fallback_img=img,
                                    fallback_boxes_xyxy=boxes_xyxy[support_i : support_i + 1],
                                    fallback_labels=labels[support_i : support_i + 1],
                                )
                            support = (torch.as_tensor([support_class], dtype=torch.int64), patch)
                            support_record = instance_records[support_i]
                            is_negative = False

            if support is None:
                continue
            phrase_to_token_mask = None
            canonical_to_token_mask = None
            negative_to_token_mask = None
            attr_pos_to_token_mask = None
            attr_neg_to_token_mask = None
            relation_to_token_mask = None
            content_to_token_mask = None
            verifier_pair_stride = 1
            verifier_num_patch_slots = 1
            is_tn = None
            attr_neg_weight_mask = None
            tn_group_ids = None
            rank_positive_phrase_to_token_mask = None
            rank_positive_canonical_to_token_mask = None
            has_rank_positive = None
            rank_positive_captions: List[Optional[str]] = []
            invalid_text_mask_records: List[Dict[str, Any]] = []
            if int(self.cfg.support_num_patches_max) > 1:
                support_classes_t, patches_or_emb = support
                if any(int(x) in not_exhaustive for x in support_classes_t.tolist()):
                    continue
                K = int(support_classes_t.numel())
                verifier_num_patch_slots = K
                if self.cfg.build_text_token_masks:
                    verifier_pair_stride = 2 if (
                        len(slot_phrases) == K * 2
                        and len(slot_records) == K * 2
                        and all(
                            bool(slot_records[2 * idx].get("sam3_tn_pair_positive", False))
                            and bool(slot_records[2 * idx + 1].get("sam3_tn_pair_negative", False))
                            for idx in range(K)
                        )
                    ) else 1
                    if verifier_pair_stride == 1 and len(slot_phrases) != K:
                        slot_phrases = [self._get_canonical_name(int(cid)) for cid in support_classes_t.tolist()]
                        slot_canonical_texts = [self._get_canonical_name(int(cid)) for cid in support_classes_t.tolist()]
                        slot_aliases = [self._get_canonical_aliases(int(cid)) for cid in support_classes_t.tolist()]
                        slot_text_is_negative = [False for _ in support_classes_t.tolist()]
                        slot_records = [
                            {"phrase": p, "head_phrase": c, "text_is_negative": False}
                            for p, c in zip(slot_phrases, slot_canonical_texts)
                        ]
                    phrases = list(slot_phrases)
                    (
                        caption,
                        phrase_to_token_mask,
                        canonical_to_token_mask,
                        attr_pos_to_token_mask,
                        attr_neg_to_token_mask,
                        relation_to_token_mask,
                        content_to_token_mask,
                        is_tn,
                        attr_neg_weight_mask,
                        tn_group_ids,
                        rank_positive_phrase_to_token_mask,
                        rank_positive_canonical_to_token_mask,
                        has_rank_positive,
                        rank_positive_captions,
                        invalid_text_mask_records,
                    ) = self._build_slot_text_masks(
                        phrases, slot_canonical_texts, slot_aliases, slot_records=slot_records
                    )
                    negative_to_token_mask = attr_neg_to_token_mask
                else:
                    phrases = self._sample_text_phrases(K)
                    caption = " ".join([f"{p} ." for p in phrases])
                    verifier_pair_stride = 1
            else:
                support_class, patch = support
                if int(support_class.item()) in not_exhaustive:
                    continue
                if self.cfg.build_text_token_masks:
                    support_cid = int(support_class.item())
                    if support_record is None:
                        phrase_text = self._get_canonical_name(support_cid)
                        canonical_text = self._get_canonical_name(support_cid)
                        alias_candidates = self._get_canonical_aliases(support_cid)
                    else:
                        phrase_text, canonical_text, alias_candidates = self._get_slot_text_for_record(
                            support_record, support_cid
                        )
                    slot_text_is_negative = [bool(support_record.get("text_is_negative", False))] if support_record is not None else [False]
                    slot_phrases_for_masks = [phrase_text]
                    slot_canonical_texts_for_masks = [canonical_text]
                    slot_aliases_for_masks = [alias_candidates]
                    slot_records = [
                        support_record
                        if support_record is not None
                        else {"phrase": phrase_text, "head_phrase": canonical_text, "text_is_negative": False}
                    ]
                    if support_record is not None and bool(support_record.get("sam3_tn_pair", False)):
                        negative_phrase = self._clean_caption_phrase(
                            support_record.get("negative_phrase", support_record.get("try_tn", ""))
                        )
                        if isinstance(negative_phrase, str) and negative_phrase.strip():
                            neg_record = dict(support_record)
                            neg_record.update(
                                {
                                    "phrase": negative_phrase,
                                    "raw_phrase": negative_phrase,
                                    "text_is_negative": True,
                                    "is_text_negative": True,
                                    "positive_phrase": phrase_text,
                                    "try_tn_head_phrase": phrase_text,
                                    "sam3_tn_pair_negative": True,
                                }
                            )
                            slot_records = [
                                {
                                    "phrase": phrase_text,
                                    "raw_phrase": phrase_text,
                                    "head": canonical_text,
                                    "head_phrase": canonical_text,
                                    "text_is_negative": False,
                                    "positive_phrase": phrase_text,
                                    "sam3_tn_pair_positive": True,
                                },
                                neg_record,
                            ]
                            slot_phrases_for_masks = [phrase_text, negative_phrase]
                            slot_canonical_texts_for_masks = [canonical_text, canonical_text]
                            slot_aliases_for_masks = [alias_candidates, alias_candidates]
                            slot_text_is_negative = [False, True]
                            verifier_pair_stride = 2
                    phrases = list(slot_phrases_for_masks)
                    (
                        caption,
                        phrase_to_token_mask,
                        canonical_to_token_mask,
                        attr_pos_to_token_mask,
                        attr_neg_to_token_mask,
                        relation_to_token_mask,
                        content_to_token_mask,
                        is_tn,
                        attr_neg_weight_mask,
                        tn_group_ids,
                        rank_positive_phrase_to_token_mask,
                        rank_positive_canonical_to_token_mask,
                        has_rank_positive,
                        rank_positive_captions,
                        invalid_text_mask_records,
                    ) = self._build_slot_text_masks(
                        phrases,
                        slot_canonical_texts_for_masks,
                        slot_aliases_for_masks,
                        slot_records=slot_records,
                    )
                    negative_to_token_mask = attr_neg_to_token_mask
                else:
                    phrases = self._sample_text_phrases(1)
                    caption = f"{phrases[0]} ."

                # Optionally keep only GT boxes of the support class (patch-only training).
                if self.cfg.keep_only_support_gt and labels.numel() > 0:
                    m = labels == int(support_class.item())
                    boxes_xyxy = boxes_xyxy[m]
                    labels = labels[m]
                    primary_instance_mask = primary_instance_mask[m]
                    if labels.numel() == 0:
                        # Shouldn't happen for positive episodes; treat as invalid and resample.
                        continue

            if self.cfg.build_text_token_masks and invalid_text_mask_records and self.cfg.text_mask_skip_invalid_canonical:
                support_class_payload = None
                if int(self.cfg.support_num_patches_max) > 1:
                    support_class_payload = [int(x) for x in support_classes_t.tolist()]
                else:
                    support_class_payload = int(support_class.item())
                for rec in invalid_text_mask_records:
                    audit_rec = dict(rec)
                    slot_idx = int(audit_rec.get("slot_idx", 0))
                    audit_rec.update(
                        {
                            "filename": str(rel_path),
                            "source": meta.get("source", None),
                            "image_id": meta.get("image_id", None),
                            "ann_id": meta.get("ann_id", None),
                            "ref_id": meta.get("ref_id", None),
                            "sent_id": meta.get("sent_id", None),
                            "is_negative_text": bool(slot_text_is_negative[slot_idx]) if slot_text_is_negative and slot_idx < len(slot_text_is_negative) else False,
                            "support_class": support_class_payload,
                            "positive_phrase": (
                                support_record.get("phrase", None)
                                if support_record is not None
                                else (
                                    slot_records[slot_idx].get("positive_phrase", slot_records[slot_idx].get("phrase", None))
                                    if slot_idx < len(slot_records)
                                    else (slot_phrases[slot_idx] if slot_idx < len(slot_phrases) else None)
                                )
                            ),
                            "head_phrase": (
                                support_record.get("head_phrase", None)
                                if support_record is not None
                                else (
                                    slot_records[slot_idx].get("head_phrase", None)
                                    if slot_idx < len(slot_records)
                                    else (slot_canonical_texts[slot_idx] if slot_idx < len(slot_canonical_texts) else None)
                                )
                            ),
                        }
                    )
                    self._append_text_mask_audit(audit_rec)
                continue

            target: Dict[str, Any] = {}
            target["boxes"] = boxes_xyxy
            target["labels"] = labels
            if primary_support is not None:
                if int(primary_instance_mask.sum().item()) != 1:
                    raise ValueError(
                        "primary support instance was lost before geometric transforms"
                    )
                target["primary_instance_mask"] = primary_instance_mask
            if meta.get("stage_b_data_driven_assignment_pair") is True:
                pair = meta.get("assignment_pair")
                anchor = pair.get("anchor") if isinstance(pair, dict) else None
                partner = pair.get("partner") if isinstance(pair, dict) else None
                pair_valid = meta.get("assignment_pair_valid")
                if not (
                    meta.get("stage_b_data_driven_assignment_pair_schema")
                    == _DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA
                    and isinstance(pair_valid, bool)
                    and isinstance(anchor, dict)
                    and isinstance(anchor.get("expression"), str)
                    and bool(anchor["expression"].strip())
                    and len(instance_records) == int(labels.numel())
                ):
                    raise ValueError("official assignment runtime payload drifted")
                roles = torch.full_like(labels, -1, dtype=torch.int64)
                anchor_ann_id = int(anchor["coco_ann_id"])
                anchor_matches = [
                    record_index
                    for record_index, record in enumerate(instance_records)
                    if int(record.get("coco_ann_id", -1)) == anchor_ann_id
                ]
                if len(anchor_matches) != 1:
                    raise ValueError(
                        "official assignment anchor is not one exact instance"
                    )
                roles[anchor_matches[0]] = 0
                partner_expression = anchor["expression"]
                if pair_valid:
                    if not (
                        isinstance(partner, dict)
                        and isinstance(partner.get("expression"), str)
                        and bool(partner["expression"].strip())
                    ):
                        raise ValueError(
                            "valid official assignment row lost its partner"
                        )
                    partner_ann_id = int(partner["coco_ann_id"])
                    partner_matches = [
                        record_index
                        for record_index, record in enumerate(instance_records)
                        if int(record.get("coco_ann_id", -1))
                        == partner_ann_id
                    ]
                    if len(partner_matches) != 1 or partner_matches == anchor_matches:
                        raise ValueError(
                            "official assignment partner is not one distinct instance"
                        )
                    roles[partner_matches[0]] = 1
                    partner_expression = partner["expression"]
                elif partner is not None:
                    raise ValueError(
                        "invalid official assignment row unexpectedly has a partner"
                    )
                target["stage_b_data_driven_assignment_valid"] = torch.as_tensor(
                    [pair_valid], dtype=torch.bool
                )
                target["stage_b_data_driven_assignment_role"] = roles
                target["stage_b_data_driven_assignment_pair_schema"] = (
                    _DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA
                )
                target["stage_b_data_driven_assignment_expressions"] = [
                    anchor["expression"],
                    partner_expression,
                ]
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])
            for identity_key in ("image_id", "ann_id", "ref_id", "sent_id"):
                identity_value = meta.get(identity_key, None)
                if identity_value is not None:
                    target[identity_key] = torch.as_tensor(
                        [int(identity_value)], dtype=torch.int64
                    )
            sample_id = meta.get("sample_id", None)
            if isinstance(sample_id, str) and sample_id:
                target["sample_id"] = sample_id
            target["caption"] = caption
            target["verifier_caption"] = caption
            if int(self.cfg.support_num_patches_max) > 1:
                if verifier_pair_stride == 1 and len(slot_canonical_texts) == len(phrases):
                    canonical_phrases = list(slot_canonical_texts)
                else:
                    canonical_phrases = [self._get_canonical_name(int(cid)) for cid in support_classes_t.tolist()]
            else:
                canonical_phrases = [self._get_canonical_name(int(support_class.item()))]
            target["stage_a_caption"] = self._build_caption_from_phrases(canonical_phrases)[0]
            dataset_name = (
                meta.get("dataset_name", None)
                or meta.get("source", None)
                or meta.get("pair_source", None)
                or self.source
                or Path(self.anno).stem
            )
            target["dataset_name"] = dataset_name
            target["global_tn_verified"] = torch.as_tensor(
                [meta.get("global_tn_verified", None) is True], dtype=torch.bool
            )
            target["fixed_stagea_topk_exact_verified"] = torch.as_tensor(
                [meta.get("fixed_stagea_topk_exact_verified", None) is True],
                dtype=torch.bool,
            )
            if meta.get("fixed_stagea_topk_exact_verified", None) is True:
                target["fixed_stagea_candidate_indices"] = torch.as_tensor(
                    meta["fixed_stagea_candidate_indices"], dtype=torch.int64
                )
                target["fixed_stagea_candidate_boxes"] = torch.as_tensor(
                    meta["fixed_stagea_candidate_boxes"], dtype=torch.float32
                )
                target["fixed_stagea_candidate_box_atol"] = torch.as_tensor(
                    [float(meta["fixed_stagea_candidate_box_atol"])],
                    dtype=torch.float32,
                )
                target["fixed_stagea_candidate_set_sha256"] = meta[
                    "fixed_stagea_candidate_set_sha256"
                ]
            target["proposalset_proxy_verified"] = torch.as_tensor(
                [meta.get("proposalset_proxy_verified", None) is True],
                dtype=torch.bool,
            )
            target["stage_b_v21_token_supervision_valid"] = torch.as_tensor(
                [
                    self.cfg.require_single_edit_token_provenance
                    and meta.get("stage_b_v21_token_supervision_valid", None)
                    is True
                ],
                dtype=torch.bool,
            )
            target["stage_b_candidate_trace_scope"] = str(
                meta.get("stage_b_candidate_trace_scope", "expression_only")
            )
            target["stage_b_changed_word_global_absent_verified"] = (
                torch.as_tensor(
                    [
                        meta.get(
                            "stage_b_changed_word_global_absent_verified", None
                        )
                        is True
                    ],
                    dtype=torch.bool,
                )
            )
            candidate_verified_indices = meta.get(
                "stage_b_changed_word_candidate_verified_indices", None
            )
            if candidate_verified_indices is not None:
                target[
                    "stage_b_changed_word_candidate_verified_indices"
                ] = torch.as_tensor(candidate_verified_indices, dtype=torch.int64)
            if isinstance(meta.get("stage_b_data_driven_trace"), dict):
                target["stage_b_data_driven_trace"] = dict(
                    meta["stage_b_data_driven_trace"]
                )
            target["benchmark_dataft_alltn"] = torch.as_tensor(
                [meta.get("benchmark_dataft_alltn", None) is True],
                dtype=torch.bool,
            )
            target["tn_scope"] = meta.get("tn_scope", None)
            if meta.get("table_b_id") is not None:
                target["table_b_id"] = meta["table_b_id"]
                target["table_b_audit_sha256"] = meta[
                    "table_b_audit_sha256"
                ]
            target["patch_ce_positive_only"] = torch.as_tensor(
                [1 if self._patch_ce_positive_only_source_flag(dataset_name) else 0],
                dtype=torch.int64,
            )
            target["verifier_pair_stride"] = torch.as_tensor([int(verifier_pair_stride)], dtype=torch.int64)
            target["verifier_num_patch_slots"] = torch.as_tensor([int(verifier_num_patch_slots)], dtype=torch.int64)
            # cap_list is used by other pipelines; keep it aligned to the number of phrases in `caption`.
            target["cap_list"] = phrases
            if phrase_to_token_mask is not None:
                target["phrase_to_token_mask"] = phrase_to_token_mask
            if canonical_to_token_mask is not None:
                target["canonical_to_token_mask"] = canonical_to_token_mask
            if attr_pos_to_token_mask is not None:
                target["attr_pos_to_token_mask"] = attr_pos_to_token_mask
            if attr_neg_to_token_mask is not None:
                target["attr_neg_to_token_mask"] = attr_neg_to_token_mask
            if relation_to_token_mask is not None:
                target["relation_to_token_mask"] = relation_to_token_mask
            if content_to_token_mask is not None:
                target["content_to_token_mask"] = content_to_token_mask
                # Compatibility alias: older code may still read phrase_semantic_token_mask.
                target["phrase_semantic_token_mask"] = content_to_token_mask
            if is_tn is not None:
                target["is_tn"] = is_tn
            if attr_neg_weight_mask is not None:
                target["attr_neg_weight_mask"] = attr_neg_weight_mask
            if tn_group_ids is not None:
                target["tn_group_ids"] = tn_group_ids
            if negative_to_token_mask is not None:
                target["negative_to_token_mask"] = negative_to_token_mask
            if rank_positive_phrase_to_token_mask is not None:
                target["rank_positive_phrase_to_token_mask"] = rank_positive_phrase_to_token_mask
            if rank_positive_canonical_to_token_mask is not None:
                target["rank_positive_canonical_to_token_mask"] = rank_positive_canonical_to_token_mask
            if has_rank_positive is not None:
                target["has_rank_positive"] = has_rank_positive
            if rank_positive_captions:
                target["rank_positive_captions"] = rank_positive_captions
            if int(self.cfg.support_num_patches_max) > 1:
                target["support_classes"] = support_classes_t
                target["support_class"] = support_classes_t[:1]  # legacy logging key
                if self.cfg.support_patch_use_embedding:
                    target["patch_global"] = patches_or_emb
                else:
                    target["patches"] = patches_or_emb
            else:
                target["support_class"] = support_class
                if not self.cfg.stage_b_gdino_adapter_no_support:
                    if self.cfg.support_patch_use_embedding:
                        target["patch_global"] = patch
                    else:
                        target["patch"] = patch
            target["is_negative_episode"] = torch.as_tensor([1 if is_negative else 0], dtype=torch.int64)
            target["is_lvis_neg_category_episode"] = torch.as_tensor(
                [1 if lvis_neg_category_only else 0],
                dtype=torch.int64,
            )
            if primary_support is not None:
                if (
                    self.cfg.native_patch_category_row_locked_support
                    and self.cfg.native_patch_category_variant == "d2"
                ):
                    target["stage_b_native_patch_category_d2"] = torch.as_tensor(
                        [meta.get("stage_b_native_patch_category_d2") is True],
                        dtype=torch.bool,
                    )
                else:
                    target["stage_b_u2_category_complete"] = torch.as_tensor(
                        [meta.get("stage_b_u2_category_complete", None) is True],
                        dtype=torch.bool,
                    )
                    target["stage_b_native_patch_category_d1"] = torch.as_tensor(
                        [meta.get("stage_b_native_patch_category_d1", None) is True],
                        dtype=torch.bool,
                    )

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        if self.cfg.strict_sample_identity:
            raise RuntimeError(
                "requested patch-episode row is invalid while strict sample "
                f"identity is enabled: index={index}; resampling is forbidden"
            )
        raise RuntimeError(
            f"Failed to sample a valid episode after {max_resample} attempts (likely too many not_exhaustive-only images)."
        )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_data_driven_new_head_partition_receipt(
    receipt: Dict[str, Any],
    datasetinfo: Dict[str, Any],
    *,
    anno_path: Path,
    expected_variant: str,
    manifest_sha: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if datasetinfo.get("stage_b_data_driven_partition") != "train":
        raise ValueError(
            "new-head partition receipt is training-only and requires explicit "
            "stage_b_data_driven_partition='train'"
        )

    canonical_sha = receipt.get("canonical_payload_sha256")
    if not (
        isinstance(canonical_sha, str)
        and _LOWER_SHA256_RE.fullmatch(canonical_sha) is not None
    ):
        raise ValueError("new-head partition receipt canonical hash is invalid")
    canonical_payload = dict(receipt)
    del canonical_payload["canonical_payload_sha256"]
    try:
        canonical_bytes = json.dumps(
            canonical_payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(
            "new-head partition receipt canonical payload is invalid"
        ) from error
    if hashlib.sha256(canonical_bytes).hexdigest() != canonical_sha:
        raise ValueError("new-head partition receipt canonical hash drifted")

    invariants = receipt.get("invariants")
    if not isinstance(invariants, dict) or not invariants or any(
        value is not True for value in invariants.values()
    ):
        raise ValueError("new-head partition receipt invariants drifted")
    if receipt.get("output_layout") != "<variant>/<partition>/<source_manifest>":
        raise ValueError("new-head partition receipt output layout drifted")
    if (
        receipt.get("output_stream_encoding")
        != "raw_input_record_including_original_line_ending_v1"
    ):
        raise ValueError("new-head partition receipt stream encoding drifted")

    source_order = receipt.get("source_manifest_order")
    if source_order != list(_DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS):
        raise ValueError("new-head partition receipt source manifest order drifted")
    source_names = set(_DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS)
    source_manifests = receipt.get("source_manifests")
    if not isinstance(source_manifests, dict) or set(source_manifests) != source_names:
        raise ValueError("new-head partition receipt source manifests drifted")
    for name in _DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS:
        source = source_manifests[name]
        if (
            not isinstance(source, dict)
            or type(source.get("rows")) is not int
            or source["rows"] <= 0
        ):
            raise ValueError(
                f"new-head partition source manifest is invalid: {name}"
            )
        for variant in _DATA_DRIVEN_NEW_HEAD_VARIANTS:
            sealed = source.get(variant)
            if not isinstance(sealed, dict):
                raise ValueError(
                    f"new-head partition source binding is missing: {variant}/{name}"
                )
            sealed_path = sealed.get("path")
            if not (
                isinstance(sealed_path, str)
                and bool(sealed_path.strip())
                and Path(_expand_path_like(sealed_path)).expanduser().is_absolute()
                and Path(_expand_path_like(sealed_path)).name == name
                and type(sealed.get("size_bytes")) is int
                and sealed["size_bytes"] > 0
                and isinstance(sealed.get("sha256"), str)
                and _LOWER_SHA256_RE.fullmatch(sealed["sha256"]) is not None
            ):
                raise ValueError(
                    f"new-head partition source binding drifted: {variant}/{name}"
                )

    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(
        _DATA_DRIVEN_NEW_HEAD_VARIANTS
    ):
        raise ValueError("new-head partition receipt output variants drifted")
    records: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for variant in _DATA_DRIVEN_NEW_HEAD_VARIANTS:
        variant_outputs = outputs[variant]
        if not isinstance(variant_outputs, dict) or set(variant_outputs) != set(
            _DATA_DRIVEN_NEW_HEAD_PARTITIONS
        ):
            raise ValueError(
                f"new-head partition receipt output partitions drifted: {variant}"
            )
        records[variant] = {}
        for partition in _DATA_DRIVEN_NEW_HEAD_PARTITIONS:
            partition_outputs = variant_outputs[partition]
            if not isinstance(partition_outputs, dict) or set(
                partition_outputs
            ) != source_names:
                raise ValueError(
                    "new-head partition receipt output manifest set drifted: "
                    f"{variant}/{partition}"
                )
            records[variant][partition] = partition_outputs
            for name in _DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS:
                record = partition_outputs[name]
                if not isinstance(record, dict):
                    raise ValueError(
                        "new-head partition output record is invalid: "
                        f"{variant}/{partition}/{name}"
                    )
                record_path = record.get("path")
                path = (
                    Path(_expand_path_like(record_path)).expanduser()
                    if isinstance(record_path, str) and record_path.strip()
                    else None
                )
                rows = record.get("rows")
                identities = record.get("unique_identities")
                images = record.get("unique_image_keys")
                size_bytes = record.get("size_bytes")
                if not (
                    path is not None
                    and path.is_absolute()
                    and path.name == name
                    and path.parent.name == partition
                    and path.parent.parent.name == variant
                    and type(rows) is int
                    and rows >= 0
                    and type(identities) is int
                    and identities == rows
                    and type(images) is int
                    and 0 <= images <= rows
                    and (rows == 0 or images > 0)
                    and type(size_bytes) is int
                    and size_bytes >= 0
                    and isinstance(record.get("sha256"), str)
                    and _LOWER_SHA256_RE.fullmatch(record["sha256"]) is not None
                    and isinstance(
                        record.get("ordered_identity_stream_sha256"), str
                    )
                    and _LOWER_SHA256_RE.fullmatch(
                        record["ordered_identity_stream_sha256"]
                    )
                    is not None
                ):
                    raise ValueError(
                        "new-head partition output record drifted: "
                        f"{variant}/{partition}/{name}"
                    )

    partition_summary = receipt.get("partition_summary")
    if not isinstance(partition_summary, dict) or set(partition_summary) != set(
        _DATA_DRIVEN_NEW_HEAD_PARTITIONS
    ):
        raise ValueError("new-head partition summary drifted")
    for partition in _DATA_DRIVEN_NEW_HEAD_PARTITIONS:
        summary = partition_summary[partition]
        rows_by_manifest = (
            summary.get("rows_by_manifest") if isinstance(summary, dict) else None
        )
        if not (
            isinstance(summary, dict)
            and type(summary.get("rows")) is int
            and summary["rows"] >= 0
            and type(summary.get("unique_image_keys")) is int
            and 0 <= summary["unique_image_keys"] <= summary["rows"]
            and (summary["rows"] == 0 or summary["unique_image_keys"] > 0)
            and isinstance(summary.get("ordered_image_key_stream_sha256"), str)
            and _LOWER_SHA256_RE.fullmatch(
                summary["ordered_image_key_stream_sha256"]
            )
            is not None
            and isinstance(rows_by_manifest, dict)
            and set(rows_by_manifest) == source_names
            and all(
                type(value) is int and value >= 0
                for value in rows_by_manifest.values()
            )
            and summary["rows"] == sum(rows_by_manifest.values())
        ):
            raise ValueError(
                f"new-head partition summary is invalid: {partition}"
            )
        for name in _DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS:
            d0 = records["d0_ordinary_primary"][partition][name]
            d1 = records["d1_category_complete"][partition][name]
            if any(
                d0[field] != d1[field]
                for field in (
                    "rows",
                    "unique_identities",
                    "unique_image_keys",
                    "ordered_identity_stream_sha256",
                )
            ) or rows_by_manifest[name] != d0["rows"]:
                raise ValueError(
                    "new-head partition paired output summary drifted: "
                    f"{partition}/{name}"
                )
            if d0["unique_image_keys"] > summary["unique_image_keys"]:
                raise ValueError(
                    "new-head partition image summary drifted: "
                    f"{partition}/{name}"
                )

    for name in _DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS:
        main_rows = sum(
            records["d0_ordinary_primary"][partition][name]["rows"]
            for partition in ("train", "dev_full", "quarantine")
        )
        if main_rows != source_manifests[name]["rows"]:
            raise ValueError(
                f"new-head partition main row accounting drifted: {name}"
            )
        if (
            records["d0_ordinary_primary"]["dev_screen"][name]["rows"]
            > records["d0_ordinary_primary"]["dev_full"][name]["rows"]
        ):
            raise ValueError(
                f"new-head partition dev-screen accounting drifted: {name}"
            )
    if (
        partition_summary["train"]["rows"] <= 0
        or partition_summary["train"]["unique_image_keys"] <= 0
        or partition_summary["dev_screen"]["rows"]
        > partition_summary["dev_full"]["rows"]
        or partition_summary["dev_screen"]["unique_image_keys"]
        > partition_summary["dev_full"]["unique_image_keys"]
    ):
        raise ValueError("new-head partition summary accounting drifted")

    receipt_variant = _DATA_DRIVEN_NEW_HEAD_VARIANT_BY_DATASET_VARIANT.get(
        expected_variant
    )
    if receipt_variant is None:
        raise ValueError(
            f"new-head partition does not support variant {expected_variant!r}"
        )
    record = records[receipt_variant]["train"].get(anno_path.name)
    if not isinstance(record, dict):
        raise ValueError("new-head training manifest is absent from its receipt")
    record_path = Path(_expand_path_like(record["path"])).resolve(strict=True)
    if record_path != anno_path:
        raise ValueError("new-head training manifest path drifted")
    if not (
        record["rows"] > 0
        and record["unique_identities"] == record["rows"]
        and record["unique_image_keys"] > 0
        and record["size_bytes"] == anno_path.stat().st_size
        and record["sha256"] == manifest_sha
    ):
        raise ValueError("new-head training manifest binding drifted")
    return record, record


def _validate_data_driven_new_head_support_receipt(
    datasetinfo: Dict[str, Any],
    *,
    partition_receipt: Dict[str, Any],
    partition_receipt_path: Path,
    partition_receipt_sha: str,
) -> None:
    support_receipt_value = datasetinfo.get(
        "stage_b_data_driven_support_receipt"
    )
    support_receipt_sha = datasetinfo.get(
        "stage_b_data_driven_support_receipt_sha256"
    )
    if not (
        isinstance(support_receipt_value, str)
        and support_receipt_value.strip()
        and isinstance(support_receipt_sha, str)
        and _LOWER_SHA256_RE.fullmatch(support_receipt_sha) is not None
    ):
        raise ValueError("new-head support receipt/hash binding is incomplete")

    support_receipt_path = Path(
        _expand_path_like(support_receipt_value)
    ).resolve(strict=True)
    if _sha256_path(support_receipt_path) != support_receipt_sha:
        raise ValueError("new-head support receipt SHA drifted")
    support_receipt = json.loads(
        support_receipt_path.read_text(encoding="utf-8")
    )
    if not (
        isinstance(support_receipt, dict)
        and support_receipt.get("schema")
        == _DATA_DRIVEN_SUPPORT_PARTITION_RECEIPT_SCHEMA
    ):
        raise ValueError("new-head support receipt contract drifted")

    canonical_sha = support_receipt.get("canonical_payload_sha256")
    if not (
        isinstance(canonical_sha, str)
        and _LOWER_SHA256_RE.fullmatch(canonical_sha) is not None
    ):
        raise ValueError("new-head support receipt canonical hash is invalid")
    canonical_payload = dict(support_receipt)
    del canonical_payload["canonical_payload_sha256"]
    try:
        canonical_bytes = json.dumps(
            canonical_payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(
            "new-head support receipt canonical payload is invalid"
        ) from error
    if hashlib.sha256(canonical_bytes).hexdigest() != canonical_sha:
        raise ValueError("new-head support receipt canonical hash drifted")

    invariants = support_receipt.get("invariants")
    if not isinstance(invariants, dict) or not invariants or any(
        value is not True for value in invariants.values()
    ):
        raise ValueError("new-head support receipt invariants drifted")

    expected_partition_record = {
        "path": str(partition_receipt_path),
        "sha256": partition_receipt_sha,
        "size_bytes": partition_receipt_path.stat().st_size,
    }
    inputs = support_receipt.get("inputs")
    input_partition_record = (
        inputs.get("partition_receipt")
        if isinstance(inputs, dict)
        else None
    )
    partition = support_receipt.get("partition")
    summarized_partition_record = (
        partition.get("receipt")
        if isinstance(partition, dict)
        else None
    )
    summarized_partition_schema = (
        partition.get("schema") if isinstance(partition, dict) else None
    )
    summarized_partition_canonical_sha = (
        partition.get("canonical_payload_sha256")
        if isinstance(partition, dict)
        else None
    )
    if not (
        input_partition_record == expected_partition_record
        and summarized_partition_record == expected_partition_record
        and summarized_partition_schema
        == _DATA_DRIVEN_NEW_HEAD_PARTITION_RECEIPT_SCHEMA
        and summarized_partition_canonical_sha
        == partition_receipt.get("canonical_payload_sha256")
    ):
        raise ValueError("new-head support receipt partition lineage drifted")

    filter_contract = support_receipt.get("filter_contract")
    required_settings = (
        filter_contract.get("required_dataset_settings")
        if isinstance(filter_contract, dict)
        else None
    )
    if not (
        required_settings == _DATA_DRIVEN_SUPPORT_REQUIRED_SETTINGS
        and filter_contract.get("D0_and_D1_share_identical_runtime_bank") is True
        and filter_contract.get("bank_consumers") == ["D0", "D1"]
    ):
        raise ValueError("new-head support receipt runtime contract drifted")
    for key, expected in _DATA_DRIVEN_SUPPORT_REQUIRED_SETTINGS.items():
        observed = datasetinfo.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"new-head support dataset setting drifted: {key}"
            )
    if str(datasetinfo.get("patch_bank_cache_path", "") or "").strip():
        raise ValueError(
            "new-head support dataset must not retain a cache path"
        )

    runtime_bank = support_receipt.get("runtime_bank")
    class_counts = (
        runtime_bank.get("class_counts")
        if isinstance(runtime_bank, dict)
        else None
    )
    valid_class_counts = bool(
        isinstance(class_counts, dict)
        and class_counts
        and all(
            isinstance(class_id, str)
            and class_id.isdigit()
            and class_id == str(int(class_id))
            and int(class_id) >= 0
            and type(count) is int
            and 0 < count <= 200
            for class_id, count in class_counts.items()
        )
    )
    runtime_candidate_rows = (
        runtime_bank.get("candidate_rows")
        if isinstance(runtime_bank, dict)
        else None
    )
    if not (
        valid_class_counts
        and type(runtime_candidate_rows) is int
        and runtime_candidate_rows == sum(class_counts.values())
        and type(runtime_bank.get("class_count")) is int
        and runtime_bank["class_count"] == len(class_counts)
    ):
        raise ValueError("new-head support runtime-bank summary drifted")

    coverage = support_receipt.get("training_class_coverage")
    required_ids = (
        coverage.get("required_class_ids")
        if isinstance(coverage, dict)
        else None
    )
    covered_ids = (
        coverage.get("covered_class_ids")
        if isinstance(coverage, dict)
        else None
    )
    support_counts = (
        coverage.get("support_counts")
        if isinstance(coverage, dict)
        else None
    )
    if not (
        isinstance(required_ids, list)
        and required_ids
        and all(
            type(class_id) is int and class_id >= 0
            for class_id in required_ids
        )
        and required_ids == sorted(set(required_ids))
        and covered_ids == required_ids
        and coverage.get("missing_class_ids") == []
        and coverage.get("required_class_count") == len(required_ids)
        and coverage.get("covered_class_count") == len(required_ids)
        and isinstance(support_counts, dict)
        and set(support_counts) == {str(class_id) for class_id in required_ids}
        and all(
            type(support_counts[str(class_id)]) is int
            and support_counts[str(class_id)] > 0
            and support_counts[str(class_id)]
            == class_counts.get(str(class_id))
            for class_id in required_ids
        )
    ):
        raise ValueError("new-head support training-class coverage drifted")

    outputs = support_receipt.get("outputs")
    runtime_record = (
        outputs.get("runtime_support_tsv")
        if isinstance(outputs, dict)
        else None
    )
    support_value = datasetinfo.get("support_patch_tsv")
    if not (
        isinstance(runtime_record, dict)
        and isinstance(support_value, str)
        and support_value.strip()
        and isinstance(runtime_record.get("path"), str)
        and runtime_record["path"].strip()
        and isinstance(runtime_record.get("sha256"), str)
        and _LOWER_SHA256_RE.fullmatch(runtime_record["sha256"]) is not None
        and type(runtime_record.get("size_bytes")) is int
        and runtime_record["size_bytes"] > 0
        and type(runtime_record.get("rows")) is int
        and runtime_record["rows"] == runtime_candidate_rows
    ):
        raise ValueError("new-head runtime support TSV binding is incomplete")
    support_path = Path(_expand_path_like(support_value)).resolve(strict=True)
    runtime_path = Path(
        _expand_path_like(runtime_record["path"])
    ).resolve(strict=True)
    if not (
        support_path == runtime_path
        and support_path.stat().st_size == runtime_record["size_bytes"]
        and _sha256_path(support_path) == runtime_record["sha256"]
    ):
        raise ValueError("new-head runtime support TSV binding drifted")


def _validate_data_driven_role_routed_clean_assignment_receipt(
    receipt: Dict[str, Any],
    datasetinfo: Dict[str, Any],
    *,
    anno_path: Path,
    manifest_sha: str,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    Path,
    str,
]:
    if datasetinfo.get("stage_b_data_driven_partition") != "train":
        raise ValueError(
            "role-routed clean assignment is training-only and requires explicit "
            "stage_b_data_driven_partition='train'"
        )

    def validate_canonical_payload(value: Dict[str, Any], *, label: str) -> None:
        claimed = value.get("canonical_payload_sha256")
        if not (
            isinstance(claimed, str)
            and _LOWER_SHA256_RE.fullmatch(claimed) is not None
        ):
            raise ValueError(f"{label} canonical hash is invalid")
        payload = dict(value)
        payload.pop("canonical_payload_sha256", None)
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise ValueError(f"{label} canonical payload is invalid") from error
        if hashlib.sha256(encoded).hexdigest() != claimed:
            raise ValueError(f"{label} canonical hash drifted")

    def load_record(
        record: Any, *, label: str
    ) -> Tuple[Path, str, Dict[str, Any]]:
        if not (
            isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and record["path"].strip()
            and isinstance(record.get("sha256"), str)
            and _LOWER_SHA256_RE.fullmatch(record["sha256"]) is not None
            and type(record.get("size_bytes")) is int
            and record["size_bytes"] > 0
        ):
            raise ValueError(f"{label} file binding is invalid")
        path = Path(_expand_path_like(record["path"])).resolve(strict=True)
        if (
            path.stat().st_size != record["size_bytes"]
            or _sha256_path(path) != record["sha256"]
        ):
            raise ValueError(f"{label} file binding drifted")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} is unreadable") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be a JSON object")
        return path, record["sha256"], payload

    validate_canonical_payload(receipt, label="role-routed clean assignment receipt")
    invariants = receipt.get("invariants")
    selection = receipt.get("selection_contract")
    forbidden_inputs = (
        set(selection.get("forbidden_inputs") or [])
        if isinstance(selection, dict)
        else set()
    )
    expected_forbidden = {
        "teacher_scores",
        "teacher_logits",
        "model_scores",
        "model_logits",
        "checkpoint_outputs",
    }
    if not (
        receipt.get("schema")
        == _DATA_DRIVEN_ROLE_ROUTED_CLEAN_ASSIGNMENT_RECEIPT_SCHEMA
        and receipt.get("scope")
        == _DATA_DRIVEN_ROLE_ROUTED_CLEAN_ASSIGNMENT_SCOPE
        and receipt.get("row_schema") == _DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA
        and receipt.get("manifest_order")
        == list(_DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS)
        and receipt.get("rows") == 263661
        and receipt.get("valid_rows") == 224723
        and receipt.get("invalid_rows") == 38938
        and receipt.get("unique_identities") == 263661
        and receipt.get("unique_image_keys") == 22359
        and receipt.get("output_layout") == "<source_manifest>"
        and receipt.get("output_stream_encoding")
        == "raw_upstream_assignment_record_including_line_ending_v1"
        and isinstance(invariants, dict)
        and invariants
        and all(value is True for value in invariants.values())
        and isinstance(selection, dict)
        and selection.get("policy")
        == "exact_seven_field_identity_train_intersection_v1"
        and selection.get("clean_role") == "d1_category_complete/train"
        and selection.get("identity_fields")
        == ["source", "image_id", "ann_id", "ref_id", "sent_id", "split", "filename"]
        and selection.get("assignment_fields_removed_only_for_base_row_validation")
        == [
            "stage_b_data_driven_assignment_pair",
            "stage_b_data_driven_assignment_pair_schema",
            "assignment_pair_valid",
            "assignment_pair",
            "assignment_pair_invalid_reason",
        ]
        and selection.get("pair_reselection_or_repair_allowed") is False
        and selection.get("model_score_free") is True
        and forbidden_inputs == expected_forbidden
    ):
        raise ValueError("role-routed clean assignment receipt contract drifted")

    upstream_assignment = receipt.get("upstream_assignment_receipt")
    assignment_record = (
        upstream_assignment.get("record")
        if isinstance(upstream_assignment, dict)
        else None
    )
    assignment_receipt_path, _, assignment_receipt = load_record(
        assignment_record, label="upstream assignment receipt"
    )
    validate_canonical_payload(
        assignment_receipt, label="upstream assignment receipt"
    )
    assignment_invariants = assignment_receipt.get("invariants")
    if not (
        upstream_assignment.get("schema") == _DATA_DRIVEN_ASSIGNMENT_RECEIPT_SCHEMA
        and upstream_assignment.get("row_schema")
        == _DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA
        and upstream_assignment.get("rows") == 321327
        and upstream_assignment.get("valid_rows") == 274582
        and upstream_assignment.get("invalid_rows") == 46745
        and upstream_assignment.get("unique_identities") == 321327
        and upstream_assignment.get("canonical_payload_sha256")
        == assignment_receipt.get("canonical_payload_sha256")
        and assignment_receipt.get("schema")
        == _DATA_DRIVEN_ASSIGNMENT_RECEIPT_SCHEMA
        and assignment_receipt.get("row_schema")
        == _DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA
        and assignment_receipt.get("manifest_order")
        == list(_DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS)
        and isinstance(assignment_invariants, dict)
        and assignment_invariants
        and all(value is True for value in assignment_invariants.values())
    ):
        raise ValueError("upstream assignment receipt lineage drifted")

    upstream_partition = receipt.get("upstream_new_head_partition_receipt")
    partition_record = (
        upstream_partition.get("record")
        if isinstance(upstream_partition, dict)
        else None
    )
    partition_receipt_path, partition_receipt_sha, partition_receipt = load_record(
        partition_record, label="upstream new-head partition receipt"
    )
    if not (
        upstream_partition.get("schema")
        == _DATA_DRIVEN_NEW_HEAD_PARTITION_RECEIPT_SCHEMA
        and upstream_partition.get("canonical_payload_sha256")
        == partition_receipt.get("canonical_payload_sha256")
        and upstream_partition.get("train_rows") == 263661
        and upstream_partition.get("train_unique_image_keys") == 22359
    ):
        raise ValueError("upstream new-head partition receipt lineage drifted")

    manifests = receipt.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != set(
        _DATA_DRIVEN_NEW_HEAD_SOURCE_MANIFESTS
    ):
        raise ValueError("role-routed clean assignment manifest set drifted")
    name = anno_path.name
    manifest = manifests.get(name)
    if not isinstance(manifest, dict):
        raise ValueError("role-routed clean assignment manifest is absent")
    output = manifest.get("output")
    assignment_input = manifest.get("assignment_input")
    clean_input = manifest.get("clean_train_input")
    upstream_manifest = (assignment_receipt.get("manifests") or {}).get(name)
    upstream_output = (
        upstream_manifest.get("output")
        if isinstance(upstream_manifest, dict)
        else None
    )
    expected_assignment_path = (
        Path(_expand_path_like(assignment_input.get("path"))).resolve(strict=True)
        if isinstance(assignment_input, dict)
        and isinstance(assignment_input.get("path"), str)
        else None
    )
    if not (
        isinstance(output, dict)
        and isinstance(output.get("path"), str)
        and Path(_expand_path_like(output["path"])).resolve(strict=True) == anno_path
        and output.get("sha256") == manifest_sha
        and output.get("size_bytes") == anno_path.stat().st_size
        and type(manifest.get("rows")) is int
        and manifest["rows"] > 0
        and manifest.get("unique_identities") == manifest["rows"]
        and type(manifest.get("valid_rows")) is int
        and type(manifest.get("invalid_rows")) is int
        and manifest["valid_rows"] + manifest["invalid_rows"] == manifest["rows"]
        and manifest.get("valid_partner_rows_verified_in_clean_train")
        == manifest["valid_rows"]
        and isinstance(manifest.get("ordered_identity_stream_sha256"), str)
        and _LOWER_SHA256_RE.fullmatch(
            manifest["ordered_identity_stream_sha256"]
        )
        is not None
        and isinstance(manifest.get("base_row_stream_sha256"), str)
        and _LOWER_SHA256_RE.fullmatch(manifest["base_row_stream_sha256"])
        is not None
        and isinstance(assignment_input, dict)
        and assignment_input == upstream_output
        and expected_assignment_path is not None
        and _sha256_path(expected_assignment_path) == assignment_input.get("sha256")
        and expected_assignment_path.stat().st_size
        == assignment_input.get("size_bytes")
        and isinstance(clean_input, dict)
        and isinstance(clean_input.get("path"), str)
        and isinstance(clean_input.get("sha256"), str)
    ):
        raise ValueError("role-routed clean assignment manifest binding drifted")

    clean_path = Path(_expand_path_like(clean_input["path"])).resolve(strict=True)
    partition_manifest, _ = _validate_data_driven_new_head_partition_receipt(
        partition_receipt,
        datasetinfo,
        anno_path=clean_path,
        expected_variant="dd1_category_complete",
        manifest_sha=clean_input["sha256"],
    )
    if not (
        clean_input.get("size_bytes") == clean_path.stat().st_size
        and clean_input.get("sha256") == _sha256_path(clean_path)
        and partition_manifest.get("rows") == manifest["rows"]
        and partition_manifest.get("unique_identities")
        == manifest["unique_identities"]
        and partition_manifest.get("unique_image_keys")
        == manifest.get("unique_image_keys")
        and partition_manifest.get("ordered_identity_stream_sha256")
        == manifest["ordered_identity_stream_sha256"]
    ):
        raise ValueError("role-routed clean D1 lineage drifted")

    _validate_data_driven_new_head_support_receipt(
        datasetinfo,
        partition_receipt=partition_receipt,
        partition_receipt_path=partition_receipt_path,
        partition_receipt_sha=partition_receipt_sha,
    )
    if assignment_receipt_path == partition_receipt_path:
        raise ValueError("assignment and partition receipt lineages collapsed")
    return (
        manifest,
        output,
        partition_receipt,
        partition_receipt_path,
        partition_receipt_sha,
    )


def _validate_data_driven_ref_dataset_binding(
    args, datasetinfo: Dict[str, Any], *, image_set: str
) -> Optional[str]:
    if image_set != "train":
        return None
    if not bool(getattr(args, "stage_b_data_driven_score", False)) or str(
        getattr(args, "stage_b_data_driven_train_mode", "")
    ).strip() != "rank_patch_only":
        return None
    rank_supervision = str(
        getattr(
            args,
            "stage_b_data_driven_rank_supervision",
            "all_nonpositive_negative_v1",
        )
        or ""
    ).strip().lower()
    if rank_supervision not in {
        "all_nonpositive_negative_v1",
        *_DATA_DRIVEN_SAME_CATEGORY_RANK_SUPERVISIONS,
    }:
        raise ValueError(
            "data-driven rank supervision contract is unknown: "
            f"{rank_supervision!r}"
        )
    category_complete = bool(
        getattr(args, "stage_b_data_driven_category_complete", False)
    )
    assignment_supervision = (
        rank_supervision in _DATA_DRIVEN_ASSIGNMENT_RANK_SUPERVISIONS
    )
    role_routed_assignment = rank_supervision in {
        "role_routed_official_assignment_top1_v1",
        "role_routed_official_assignment_all_exclusive_nonowned_v2",
    }
    expected_variant = (
        "dd1_official_assignment_pair"
        if assignment_supervision
        else (
            "dd1_category_complete"
            if category_complete
            else "dd0_ordinary_primary"
        )
    )
    if (
        rank_supervision in _DATA_DRIVEN_SAME_CATEGORY_RANK_SUPERVISIONS
        and not category_complete
    ):
        raise ValueError(
            "same-category rank supervision requires the DD1 category-complete "
            "dataset"
        )
    variant = datasetinfo.get("stage_b_data_driven_variant")
    if variant != expected_variant:
        raise ValueError(
            "data-driven rank/patch dataset variant drifted: "
            f"expected={expected_variant!r}, got={variant!r}"
        )
    receipt_value = datasetinfo.get("stage_b_data_driven_receipt")
    receipt_sha = datasetinfo.get("stage_b_data_driven_receipt_sha256")
    manifest_sha = datasetinfo.get("stage_b_data_driven_manifest_sha256")
    if not all(
        isinstance(value, str) and bool(value.strip())
        for value in (receipt_value, receipt_sha, manifest_sha)
    ):
        raise ValueError("data-driven dataset receipt/hash binding is incomplete")
    receipt_path = Path(_expand_path_like(receipt_value)).resolve(strict=True)
    anno_path = Path(_expand_path_like(datasetinfo["anno"])).resolve(strict=True)
    if _sha256_path(receipt_path) != receipt_sha:
        raise ValueError("data-driven paired receipt SHA drifted")
    if _sha256_path(anno_path) != manifest_sha:
        raise ValueError("data-driven annotation SHA drifted")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("data-driven paired receipt contract drifted")
    receipt_schema = receipt.get("schema")
    new_head_partition = bool(
        not assignment_supervision
        and receipt_schema == _DATA_DRIVEN_NEW_HEAD_PARTITION_RECEIPT_SCHEMA
    )
    assignment_overfit = bool(
        assignment_supervision
        and receipt_schema == _DATA_DRIVEN_ASSIGNMENT_OVERFIT_RECEIPT_SCHEMA
    )
    clean_role_routed_assignment = bool(
        role_routed_assignment
        and receipt_schema
        == _DATA_DRIVEN_ROLE_ROUTED_CLEAN_ASSIGNMENT_RECEIPT_SCHEMA
    )
    if datasetinfo.get("lazy_jsonl") is True and not clean_role_routed_assignment:
        raise ValueError(
            "lazy_jsonl is restricted to the sealed role-routed clean assignment"
        )
    if clean_role_routed_assignment and datasetinfo.get("lazy_jsonl") is not True:
        raise ValueError(
            "role-routed clean assignment requires lazy_jsonl=true"
        )
    expected_receipt_schemas = (
        {_DATA_DRIVEN_ROLE_ROUTED_CLEAN_ASSIGNMENT_RECEIPT_SCHEMA}
        if role_routed_assignment
        else (
        {
            _DATA_DRIVEN_ASSIGNMENT_RECEIPT_SCHEMA,
            _DATA_DRIVEN_ASSIGNMENT_OVERFIT_RECEIPT_SCHEMA,
        }
        if assignment_supervision
        else {
            "pivot.stageb.data_driven_ref_pair_receipt/v1",
            _DATA_DRIVEN_NEW_HEAD_PARTITION_RECEIPT_SCHEMA,
        }
        )
    )
    if receipt_schema not in expected_receipt_schemas:
        raise ValueError("data-driven paired receipt contract drifted")
    invariants = receipt.get("invariants")
    if not isinstance(invariants, dict) or any(
        value is not True for value in invariants.values()
    ):
        raise ValueError("data-driven paired receipt contract drifted")
    if clean_role_routed_assignment:
        manifest, record, _, _, _ = (
            _validate_data_driven_role_routed_clean_assignment_receipt(
                receipt,
                datasetinfo,
                anno_path=anno_path,
                manifest_sha=manifest_sha,
            )
        )
    elif new_head_partition:
        manifest, record = _validate_data_driven_new_head_partition_receipt(
            receipt,
            datasetinfo,
            anno_path=anno_path,
            expected_variant=expected_variant,
            manifest_sha=manifest_sha,
        )
        _validate_data_driven_new_head_support_receipt(
            datasetinfo,
            partition_receipt=receipt,
            partition_receipt_path=receipt_path,
            partition_receipt_sha=receipt_sha,
        )
    elif assignment_overfit:
        if (
            receipt.get("row_schema") != _DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA
            or receipt.get("rows") != 64
            or receipt.get("valid_rows") != 64
            or receipt.get("invalid_rows") != 0
            or receipt.get("unique_images") != 64
            or receipt.get("unique_unordered_annotation_edges") != 64
            or receipt.get("unique_annotation_endpoints") != 128
            or receipt.get("output_manifest") != anno_path.name
        ):
            raise ValueError("data-driven Overfit64 receipt contract drifted")
        record = receipt.get("output")
        manifest = {"rows": receipt["rows"], "output": record}
        selection = receipt.get("selection_contract")
        support = receipt.get("support")
        mini_support = (
            support.get("mini_support_tsv")
            if isinstance(support, dict)
            else None
        )
        support_value = datasetinfo.get("support_patch_tsv")
        if not (
            isinstance(support_value, str)
            and support_value.strip()
            and isinstance(mini_support, dict)
            and isinstance(mini_support.get("path"), str)
            and mini_support["path"].strip()
        ):
            raise ValueError(
                "data-driven Overfit64 fixed external support binding is incomplete"
            )
        support_path = Path(_expand_path_like(support_value)).resolve(strict=True)
        if (
            not isinstance(selection, dict)
            or selection.get("model_score_free") is not True
            or selection.get("target_crop_fallback_allowed") is not False
            or selection.get("runtime_support_candidates_per_selected_class") != 1
            or not isinstance(mini_support, dict)
            or support_path
            != Path(_expand_path_like(mini_support.get("path"))).resolve(strict=True)
            or _sha256_path(support_path) != mini_support.get("sha256")
            or datasetinfo.get("support_patch_bucket") != "clean"
            or datasetinfo.get("support_patch_use_embedding") is not False
            or datasetinfo.get("support_patch_max_per_class") != 1
            or datasetinfo.get("patch_bank_cache") is not False
            or datasetinfo.get("patch_bank_cache_write") is not False
        ):
            raise ValueError(
                "data-driven Overfit64 fixed external support contract drifted"
            )
    else:
        if (
            receipt.get("rows") != 321327
            or receipt.get("unique_identities") != 321327
        ):
            raise ValueError("data-driven paired receipt contract drifted")
        manifest = (receipt.get("manifests") or {}).get(anno_path.name)
        if assignment_supervision:
            record = manifest.get("output") if isinstance(manifest, dict) else None
        else:
            role = (
                "category_complete"
                if expected_variant == "dd1_category_complete"
                else "ordinary_primary"
            )
            record = manifest.get(role) if isinstance(manifest, dict) else None
    if not isinstance(record, dict) or record.get("sha256") != manifest_sha:
        raise ValueError("data-driven manifest is absent from its paired receipt")
    if int(manifest.get("rows", -1)) <= 0:
        raise ValueError("data-driven receipt declares an empty manifest")
    if rank_supervision in _DATA_DRIVEN_SAME_CATEGORY_RANK_SUPERVISIONS:
        if clean_role_routed_assignment:
            return expected_variant
        complete_receipt_record = receipt.get(
            "upstream_category_complete_receipt"
            if assignment_overfit
            else "category_complete_receipt"
        )
        if not isinstance(complete_receipt_record, dict):
            raise ValueError(
                "same-category rank supervision requires the category-complete "
                "receipt"
            )
        complete_receipt_value = complete_receipt_record.get("path")
        complete_receipt_sha = complete_receipt_record.get("sha256")
        if not (
            isinstance(complete_receipt_value, str)
            and complete_receipt_value.strip()
            and isinstance(complete_receipt_sha, str)
            and len(complete_receipt_sha) == 64
        ):
            raise ValueError("category-complete receipt binding is incomplete")
        complete_receipt_path = Path(
            _expand_path_like(complete_receipt_value)
        ).resolve(strict=True)
        if _sha256_path(complete_receipt_path) != complete_receipt_sha:
            raise ValueError("category-complete receipt SHA drifted")
        complete_receipt = json.loads(
            complete_receipt_path.read_text(encoding="utf-8")
        )
        if assignment_overfit:
            complete_manifests = (
                complete_receipt.get("manifests")
                if isinstance(complete_receipt, dict)
                else None
            )
            upstream_record = receipt.get("upstream_assignment_receipt")
            if not (
                isinstance(upstream_record, dict)
                and isinstance(upstream_record.get("path"), str)
                and upstream_record["path"].strip()
            ):
                raise ValueError(
                    "Overfit64 receipt lost its upstream assignment binding"
                )
            upstream_path = Path(
                _expand_path_like(upstream_record.get("path"))
            ).resolve(strict=True)
            if _sha256_path(upstream_path) != upstream_record.get("sha256"):
                raise ValueError("Overfit64 upstream assignment receipt drifted")
            upstream_receipt = json.loads(
                upstream_path.read_text(encoding="utf-8")
            )
            upstream_complete = (
                upstream_receipt.get("category_complete_receipt")
                if isinstance(upstream_receipt, dict)
                else None
            )
            valid_complete = bool(
                isinstance(complete_receipt, dict)
                and complete_receipt.get("schema")
                == "pivot.stageb.u2_category_complete_receipt/v1"
                and isinstance(complete_manifests, dict)
                and complete_manifests
                and all(
                    isinstance(item, dict)
                    and item.get("rows") == item.get("multi_instance_rows")
                    and int(item.get("rows", 0)) > 0
                    for item in complete_manifests.values()
                )
                and isinstance(upstream_receipt, dict)
                and upstream_receipt.get("schema")
                == _DATA_DRIVEN_ASSIGNMENT_RECEIPT_SCHEMA
                and isinstance(upstream_complete, dict)
                and upstream_complete.get("sha256") == complete_receipt_sha
            )
        else:
            complete_manifest = (
                complete_receipt.get("manifests", {}).get(anno_path.name)
                if isinstance(complete_receipt, dict)
                else None
            )
            assignment_input = (
                manifest.get("input")
                if assignment_supervision and isinstance(manifest, dict)
                else None
            )
            valid_complete = bool(
                isinstance(complete_receipt, dict)
                and complete_receipt.get("schema")
                == "pivot.stageb.u2_category_complete_receipt/v1"
                and isinstance(complete_manifest, dict)
                and complete_manifest.get("rows")
                == complete_manifest.get("multi_instance_rows")
                and int(complete_manifest.get("rows", 0)) > 0
                and complete_manifest.get("output", {}).get("sha256")
                == (
                    assignment_input.get("sha256")
                    if isinstance(assignment_input, dict)
                    else manifest_sha
                )
            )
        if not valid_complete:
            raise ValueError(
                "same-category rank supervision is not backed by an all-row "
                "multi-instance receipt"
            )
    return expected_variant


def _validate_data_driven_ref_metas(
    metas: Sequence[Dict[str, Any]],
    variant: Optional[str],
    *,
    rank_supervision: str = "all_nonpositive_negative_v1",
) -> None:
    if variant is None:
        return
    for index, meta in enumerate(metas):
        instances = meta.get("instances")
        if meta.get("primary_support_instance_index") != 0 or not (
            isinstance(instances, list)
            and instances
            and all(isinstance(instance, dict) for instance in instances)
        ):
            raise ValueError(
                f"data-driven {variant} row {index} lost its primary instance"
            )
        if variant == "dd0_ordinary_primary":
            valid = (
                len(instances) == 1
                and meta.get("stage_b_data_driven_ordinary_primary") is True
                and meta.get("stage_b_data_driven_ordinary_primary_schema")
                == "pivot.stageb.data_driven_ordinary_primary/v1"
                and meta.get("stage_b_u2_category_complete") is not True
            )
        else:
            valid = (
                meta.get("stage_b_u2_category_complete") is True
                and meta.get("stage_b_u2_category_complete_schema")
                == "pivot.stageb.u2_category_complete_ref/v1"
                and all(
                    instance.get("class_id") == instances[0].get("class_id")
                    for instance in instances
                )
            )
            if rank_supervision in _DATA_DRIVEN_SAME_CATEGORY_RANK_SUPERVISIONS:
                valid = valid and len(instances) >= 2
            if variant == "dd1_official_assignment_pair":
                pair_valid = meta.get("assignment_pair_valid")
                pair = meta.get("assignment_pair")
                anchor = pair.get("anchor") if isinstance(pair, dict) else None
                partner = pair.get("partner") if isinstance(pair, dict) else None
                valid = valid and (
                    meta.get("stage_b_data_driven_assignment_pair") is True
                    and meta.get(
                        "stage_b_data_driven_assignment_pair_schema"
                    )
                    == _DATA_DRIVEN_ASSIGNMENT_ROW_SCHEMA
                    and isinstance(pair_valid, bool)
                    and isinstance(anchor, dict)
                    and int(anchor.get("coco_ann_id", -1))
                    == int(instances[0].get("coco_ann_id", -2))
                    and isinstance(anchor.get("expression"), str)
                    and _norm_text(anchor["expression"])
                    == _norm_text(
                        instances[0].get(
                            "raw_phrase",
                            instances[0].get("positive_phrase", ""),
                        )
                    )
                )
                if pair_valid is True:
                    partner_ann_id = (
                        int(partner.get("coco_ann_id", -1))
                        if isinstance(partner, dict)
                        else -1
                    )
                    matching = [
                        instance
                        for instance in instances
                        if int(instance.get("coco_ann_id", -2))
                        == partner_ann_id
                    ]
                    target_iou = (
                        partner.get("target_iou")
                        if isinstance(partner, dict)
                        else None
                    )
                    valid = valid and (
                        isinstance(partner, dict)
                        and partner_ann_id
                        != int(anchor.get("coco_ann_id", partner_ann_id))
                        and len(matching) == 1
                        and isinstance(partner.get("expression"), str)
                        and bool(_norm_text(partner["expression"]))
                        and _norm_text(partner["expression"])
                        != _norm_text(anchor["expression"])
                        and isinstance(target_iou, (int, float))
                        and not isinstance(target_iou, bool)
                        and math.isfinite(float(target_iou))
                        and 0.0 <= float(target_iou) < 0.3
                    )
                else:
                    valid = valid and (
                        partner is None
                        and isinstance(
                            meta.get("assignment_pair_invalid_reason"), str
                        )
                        and bool(meta["assignment_pair_invalid_reason"].strip())
                    )
        if not valid:
            raise ValueError(
                f"data-driven {variant} row {index} changed its supervision surface"
            )


def _validate_native_patch_category_dataset_binding(
    args, datasetinfo: Dict[str, Any], *, image_set: str
) -> Optional[Dict[int, int]]:
    enabled = bool(
        getattr(args, "stage_b_native_patch_category", False)
        or getattr(args, "stage_b_u0_gate_aligned_d10", False)
        or getattr(args, "stage_b_u0_gate_aligned_d11", False)
        or getattr(args, "stage_b_u0_gate_aligned_d12", False)
        or getattr(args, "stage_b_u0_gate_aligned_d13", False)
        or getattr(args, "stage_b_u2v2_training_dataset_binding", False)
        or getattr(args, "stage_b_u2v3_training_dataset_binding", False)
    )
    if not enabled:
        if datasetinfo.get("native_patch_category_row_locked_support") is True:
            raise ValueError(
                "row-locked native patch support requires its model training mode"
            )
        return None
    variant = str(
        datasetinfo.get("stage_b_native_patch_category_variant", "")
    ).strip().lower()
    if image_set != "train":
        raise ValueError(
            "native patch-category data are a training-only dataset binding"
        )
    if not (
        datasetinfo.get("native_patch_category_row_locked_support") is True
        and variant in {"d1", "d2"}
    ):
        raise ValueError("native patch-category dataset mode is incomplete")
    split = datasetinfo.get("stage_b_native_patch_category_split")
    if variant == "d1" and split not in {"train", "dev_screen", "dev_full"}:
        raise ValueError("native patch-category D1 split is invalid")
    if variant == "d2" and split != "train":
        raise ValueError(
            "native patch-category D2 training requires its weighted train split"
        )
    source_dataset = datasetinfo.get(
        "stage_b_native_patch_category_source_dataset"
    )
    if variant == "d2" and source_dataset not in {
        "refcoco",
        "refcocoplus",
        "refcocog",
    }:
        raise ValueError("native patch-category D2 source dataset is invalid")
    if variant == "d2" and not (
        datasetinfo.get("stage_b_native_patch_category_row_schema")
        == _NATIVE_PATCH_CATEGORY_D2_ROW_SCHEMA
        and datasetinfo.get(
            "stage_b_native_patch_category_sampling_contract"
        )
        == _NATIVE_PATCH_CATEGORY_D2_SAMPLING_CONTRACT
        and datasetinfo.get(
            "stage_b_native_patch_category_sampling_weight_field"
        )
        == _NATIVE_PATCH_CATEGORY_D2_WEIGHT_FIELD
    ):
        raise ValueError("native patch-category D2 declared schema drifted")
    receipt_value = datasetinfo.get("stage_b_native_patch_category_receipt")
    receipt_sha = datasetinfo.get("stage_b_native_patch_category_receipt_sha256")
    manifest_sha = datasetinfo.get("stage_b_native_patch_category_manifest_sha256")
    if not all(
        isinstance(value, str) and _LOWER_SHA256_RE.fullmatch(value) is not None
        for value in (receipt_sha, manifest_sha)
    ) or not isinstance(receipt_value, str):
        raise ValueError(
            f"native patch-category {variant.upper()} hash binding is incomplete"
        )
    receipt_path = Path(_expand_path_like(receipt_value)).resolve(strict=True)
    anno_path = Path(_expand_path_like(datasetinfo["anno"])).resolve(strict=True)
    if _sha256_path(receipt_path) != receipt_sha:
        raise ValueError(
            f"native patch-category {variant.upper()} receipt SHA drifted"
        )
    if _sha256_path(anno_path) != manifest_sha:
        raise ValueError(
            f"native patch-category {variant.upper()} annotation SHA drifted"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("native patch-category receipt must be a JSON object")
    canonical_sha = receipt.get("canonical_payload_sha256")
    canonical_payload = dict(receipt)
    canonical_payload.pop("canonical_payload_sha256", None)
    replay_sha = hashlib.sha256(
        json.dumps(
            canonical_payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    split_record = receipt.get("splits", {}).get(split)
    if variant == "d2" and isinstance(split_record, dict):
        split_record = split_record.get(source_dataset)
    output_record = (
        split_record.get("output") if isinstance(split_record, dict) else None
    )
    invariants = receipt.get("invariants")
    expected_receipt_schema = (
        _NATIVE_PATCH_CATEGORY_D1_RECEIPT_SCHEMA
        if variant == "d1"
        else _NATIVE_PATCH_CATEGORY_D2_RECEIPT_SCHEMA
    )
    d2_sampling_contract = receipt.get("sampling_contract")
    d2_binding_valid = variant == "d1" or bool(
        isinstance(split_record, dict)
        and receipt.get("row_schema") == _NATIVE_PATCH_CATEGORY_D2_ROW_SCHEMA
        and isinstance(d2_sampling_contract, dict)
        and d2_sampling_contract.get("name")
        == _NATIVE_PATCH_CATEGORY_D2_SAMPLING_CONTRACT
        and d2_sampling_contract.get("source_mix_weights", {}).get(
            source_dataset
        )
        == split_record.get("mix_weight")
        and split_record.get("mix_weight")
        == datasetinfo.get("mix_weight")
        and isinstance(split_record.get("sampling_weight_mean"), float)
        and math.isclose(
            split_record["sampling_weight_mean"],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    if not (
        receipt.get("schema") == expected_receipt_schema
        and isinstance(canonical_sha, str)
        and canonical_sha == replay_sha
        and isinstance(invariants, dict)
        and bool(invariants)
        and all(value is True for value in invariants.values())
        and isinstance(output_record, dict)
        and output_record.get("sha256") == manifest_sha
        and Path(output_record.get("path", "")).resolve(strict=True) == anno_path
        and output_record.get("size_bytes") == anno_path.stat().st_size
        and type(output_record.get("rows")) is int
        and output_record["rows"] > 0
        and d2_binding_valid
    ):
        raise ValueError(
            f"native patch-category {variant.upper()} receipt contract drifted"
        )
    support_record = receipt.get("inputs", {}).get("support_partition_receipt")
    if not (
        isinstance(support_record, dict)
        and set(support_record) == {"path", "sha256", "size_bytes"}
        and isinstance(support_record.get("path"), str)
        and isinstance(support_record.get("sha256"), str)
        and _LOWER_SHA256_RE.fullmatch(support_record["sha256"]) is not None
        and type(support_record.get("size_bytes")) is int
        and support_record["size_bytes"] > 0
    ):
        raise ValueError("native patch-category support receipt binding is incomplete")
    support_receipt_path = Path(support_record["path"]).resolve(strict=True)
    if (
        support_receipt_path.stat().st_size != support_record["size_bytes"]
        or _sha256_path(support_receipt_path) != support_record["sha256"]
    ):
        raise ValueError("native patch-category support receipt binding drifted")
    support_receipt = json.loads(
        support_receipt_path.read_text(encoding="utf-8")
    )
    support_payload = dict(support_receipt)
    support_canonical_sha = support_payload.pop("canonical_payload_sha256", None)
    support_replay_sha = hashlib.sha256(
        json.dumps(
            support_payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    support_invariants = support_receipt.get("invariants")
    if not (
        support_receipt.get("schema")
        == _DATA_DRIVEN_SUPPORT_PARTITION_RECEIPT_SCHEMA
        and support_canonical_sha == support_replay_sha
        and isinstance(support_invariants, dict)
        and support_invariants.get(
            "alias_bridges_are_unique_canonical_metadata_matches"
        )
        is True
        and support_invariants.get(
            "alias_bridges_reuse_only_filtered_base_paths"
        )
        is True
    ):
        raise ValueError("native patch-category support receipt payload drifted")
    alias_rows = support_receipt.get("alias_bridges")
    if not isinstance(alias_rows, list):
        raise ValueError("native patch-category alias bridge list is missing")
    alias_bridges: Dict[int, int] = {}
    for alias_index, alias in enumerate(alias_rows):
        if not isinstance(alias, dict):
            raise ValueError(
                f"native patch-category alias bridge {alias_index} is invalid"
            )
        target = alias.get("target_class_id")
        source = alias.get("source_cache_class_id")
        if (
            type(target) is not int
            or type(source) is not int
            or target == source
            or type(alias.get("candidate_rows")) is not int
            or alias["candidate_rows"] <= 0
            or target in alias_bridges
        ):
            raise ValueError(
                f"native patch-category alias bridge {alias_index} drifted"
            )
        alias_bridges[target] = source
    required_settings = {
        "neg_episode_prob": 0.0,
        "support_num_patches_min": 1,
        "support_num_patches_max": 1,
        "support_patch_use_embedding": False,
        "build_text_token_masks": True,
        "strict_sample_identity": True,
        "anno_cache": False,
        "anno_cache_write": False,
    }
    if variant == "d2":
        required_settings["tn_balance_sampling"] = False
    for key, expected in required_settings.items():
        observed = datasetinfo.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                "native patch-category "
                f"{variant.upper()} dataset setting drifted: {key}"
            )
    if datasetinfo.get("support_patch_tsv") not in {None, ""}:
        raise ValueError(
            "native patch-category must use only its row support witness"
        )
    return alias_bridges


def build_patch_episode(image_set: str, args, datasetinfo: Dict[str, Any]):
    table_b_contract = None
    if image_set == "train":
        try:
            table_b_contract = _validate_u2v5_matched_table_b_binding(
                args, datasetinfo
            ) or validate_table_b_dataset_binding(args, datasetinfo)
        except TableBContractError as error:
            raise ValueError(
                f"Table-B dataset contract failed closed: {error}"
            ) from error
    data_driven_ref_variant = _validate_data_driven_ref_dataset_binding(
        args, datasetinfo, image_set=image_set
    )
    native_patch_category_alias_bridges = (
        _validate_native_patch_category_dataset_binding(
            args, datasetinfo, image_set=image_set
        )
    )
    native_patch_category_row_locked_support = (
        native_patch_category_alias_bridges is not None
    )
    native_patch_category_variant = (
        str(datasetinfo.get("stage_b_native_patch_category_variant", ""))
        .strip()
        .lower()
        if native_patch_category_row_locked_support
        else None
    )
    native_patch_category_source_dataset = (
        datasetinfo.get("stage_b_native_patch_category_source_dataset")
        if native_patch_category_variant == "d2"
        else None
    )
    data_driven_rank_supervision = str(
        getattr(
            args,
            "stage_b_data_driven_rank_supervision",
            "all_nonpositive_negative_v1",
        )
    ).strip().lower()
    strict_sample_identity = datasetinfo.get(
        "strict_sample_identity",
        getattr(args, "stage_b_data_driven_strict_sample_identity", False),
    )
    if not isinstance(strict_sample_identity, bool):
        raise ValueError("strict_sample_identity must be an exact JSON boolean")
    if (
        data_driven_rank_supervision
        in _DATA_DRIVEN_SAME_CATEGORY_RANK_SUPERVISIONS
        and strict_sample_identity is not True
    ):
        raise ValueError(
            "same-category rank supervision requires strict sample identity"
        )
    lazy_jsonl = datasetinfo.get("lazy_jsonl", False)
    if not isinstance(lazy_jsonl, bool):
        raise ValueError("lazy_jsonl must be an exact JSON boolean")
    if lazy_jsonl and data_driven_ref_variant is None:
        raise ValueError(
            "lazy_jsonl is restricted to validated data-driven training rows"
        )
    root = datasetinfo["root"]
    anno = datasetinfo["anno"]
    source = datasetinfo.get("source", getattr(args, "patch_episode_source", None))
    canonical_classes_json = datasetinfo.get(
        "canonical_classes_json", getattr(args, "canonical_classes_json", None)
    )
    lvis_image_root = datasetinfo.get("lvis_image_root", getattr(args, "lvis_image_root", None))
    coco_image_root = datasetinfo.get("coco_image_root", getattr(args, "coco_image_root", None))
    vg_image_roots = datasetinfo.get("vg_image_roots", getattr(args, "vg_image_roots", None))
    box_format = datasetinfo.get("box_format", "xyxy")
    neg_episode_prob = float(datasetinfo.get("neg_episode_prob", getattr(args, "neg_episode_prob", 0.2)))
    support_min_count = int(datasetinfo.get("support_min_count", getattr(args, "support_min_count", 2)))
    support_patch_size = int(datasetinfo.get("support_patch_size", getattr(args, "support_patch_size", 224)))
    support_num_patches_min = int(
        datasetinfo.get("support_num_patches_min", getattr(args, "support_num_patches_min", 1))
    )
    support_num_patches_max = int(
        datasetinfo.get("support_num_patches_max", getattr(args, "support_num_patches_max", 1))
    )
    support_use_all_gt_classes = bool(
        datasetinfo.get("support_use_all_gt_classes", getattr(args, "support_use_all_gt_classes", False))
    )
    lvis_neg_category_only = bool(
        datasetinfo.get("lvis_neg_category_only", getattr(args, "lvis_neg_category_only", False))
    )
    support_patch_max_per_class = int(
        datasetinfo.get("support_patch_max_per_class", getattr(args, "support_patch_max_per_class", 0))
    )
    negative_max_tries = int(datasetinfo.get("negative_max_tries", getattr(args, "negative_max_tries", 50)))
    support_patch_tsv = datasetinfo.get("support_patch_tsv", getattr(args, "support_patch_tsv", None))
    support_patch_bucket = datasetinfo.get("support_patch_bucket", getattr(args, "support_patch_bucket", None))
    support_patch_class_map_json = datasetinfo.get(
        "support_patch_class_map_json", getattr(args, "support_patch_class_map_json", None)
    )
    support_patch_use_embedding = bool(
        datasetinfo.get("support_patch_use_embedding", getattr(args, "support_patch_use_embedding", False))
    )
    patch_emb_cache_size = int(datasetinfo.get("patch_emb_cache_size", getattr(args, "patch_emb_cache_size", 4096)))
    support_patch_image_root = datasetinfo.get(
        "support_patch_image_root", getattr(args, "support_patch_image_root", None)
    )
    keep_only_support_gt = bool(datasetinfo.get("keep_only_support_gt", getattr(args, "keep_only_support_gt", False)))
    keep_only_patchset_gt = bool(datasetinfo.get("keep_only_patchset_gt", getattr(args, "keep_only_patchset_gt", True)))
    patch_text_augment = bool(datasetinfo.get("patch_text_augment", getattr(args, "patch_text_augment", False)))
    patch_text_aug_p_object = float(
        datasetinfo.get("patch_text_aug_p_object", getattr(args, "patch_text_aug_p_object", 0.5))
    )
    patch_text_aug_p_alias = float(datasetinfo.get("patch_text_aug_p_alias", getattr(args, "patch_text_aug_p_alias", 0.25)))
    patch_text_aug_p_vg = float(datasetinfo.get("patch_text_aug_p_vg", getattr(args, "patch_text_aug_p_vg", 0.25)))
    patch_text_aug_vg_jsonl = datasetinfo.get(
        "patch_text_aug_vg_jsonl", getattr(args, "patch_text_aug_vg_jsonl", None)
    )
    patch_text_aug_vg_pool_size = int(
        datasetinfo.get("patch_text_aug_vg_pool_size", getattr(args, "patch_text_aug_vg_pool_size", 50000))
    )
    patch_text_aug_max_words = int(
        datasetinfo.get("patch_text_aug_max_words", getattr(args, "patch_text_aug_max_words", 6))
    )
    vg_phrase_labeler = datasetinfo.get("vg_phrase_labeler", getattr(args, "vg_phrase_labeler", "prefix"))
    phrase_classifier_ckpt = datasetinfo.get("phrase_classifier_ckpt", getattr(args, "phrase_classifier_ckpt", None))
    phrase_classifier_device = datasetinfo.get(
        "phrase_classifier_device", getattr(args, "phrase_classifier_device", "cpu")
    )
    phrase_classifier_max_length = int(
        datasetinfo.get("phrase_classifier_max_length", getattr(args, "phrase_classifier_max_length", 24))
    )
    phrase_classifier_batch_size = int(
        datasetinfo.get("phrase_classifier_batch_size", getattr(args, "phrase_classifier_batch_size", 64))
    )
    phrase_classifier_min_conf = float(
        datasetinfo.get("phrase_classifier_min_conf", getattr(args, "phrase_classifier_min_conf", 0.0))
    )
    phrase_cache_size = int(datasetinfo.get("phrase_cache_size", getattr(args, "phrase_cache_size", 50000)))
    patch_bank_cache = bool(datasetinfo.get("patch_bank_cache", getattr(args, "patch_bank_cache", True)))
    patch_bank_cache_path = datasetinfo.get("patch_bank_cache_path", getattr(args, "patch_bank_cache_path", None))
    patch_bank_cache_write = bool(datasetinfo.get("patch_bank_cache_write", getattr(args, "patch_bank_cache_write", True)))
    anno_cache = bool(datasetinfo.get("anno_cache", getattr(args, "anno_cache", True)))
    anno_cache_path = datasetinfo.get("anno_cache_path", getattr(args, "anno_cache_path", None))
    anno_cache_write = bool(datasetinfo.get("anno_cache_write", getattr(args, "anno_cache_write", True)))
    build_text_token_masks = bool(
        datasetinfo.get("build_text_token_masks", getattr(args, "build_text_token_masks", False))
    )
    text_encoder_type = datasetinfo.get("text_encoder_type", getattr(args, "text_encoder_type", "bert-base-uncased"))
    max_text_len = int(datasetinfo.get("max_text_len", getattr(args, "max_text_len", 256)))
    text_mask_warn_limit = int(
        datasetinfo.get("text_mask_warn_limit", getattr(args, "text_mask_warn_limit", 20))
    )
    text_mask_skip_invalid_canonical = bool(
        datasetinfo.get("text_mask_skip_invalid_canonical", getattr(args, "text_mask_skip_invalid_canonical", False))
    )
    text_mask_audit_jsonl = datasetinfo.get(
        "text_mask_audit_jsonl",
        getattr(args, "text_mask_audit_jsonl", None),
    )
    use_tn_category_weights = bool(
        datasetinfo.get("use_tn_category_weights", getattr(args, "use_tn_category_weights", True))
    )
    default_tn_category_weight = float(
        datasetinfo.get("default_tn_category_weight", getattr(args, "default_tn_category_weight", 1.0))
    )
    skip_tn_if_neg_overlaps_canonical = bool(
        datasetinfo.get(
            "skip_tn_if_neg_overlaps_canonical",
            getattr(args, "skip_tn_if_neg_overlaps_canonical", True),
        )
    )
    skip_ambiguous_tn = bool(
        datasetinfo.get("skip_ambiguous_tn", getattr(args, "skip_ambiguous_tn", True))
    )
    skip_tn_if_changed_span_not_found = bool(
        datasetinfo.get(
            "skip_tn_if_changed_span_not_found",
            getattr(args, "skip_tn_if_changed_span_not_found", True),
        )
    )
    skip_tn_if_changed_span_empty_after_filter = bool(
        datasetinfo.get(
            "skip_tn_if_changed_span_empty_after_filter",
            getattr(args, "skip_tn_if_changed_span_empty_after_filter", True),
        )
    )
    skip_relation_like_tn_in_v1 = bool(
        datasetinfo.get("skip_relation_like_tn_in_v1", getattr(args, "skip_relation_like_tn_in_v1", False))
    )
    tn_balance_sampling = bool(
        datasetinfo.get("tn_balance_sampling", getattr(args, "tn_balance_sampling", True))
    )
    tn_balance_cap = float(datasetinfo.get("tn_balance_cap", getattr(args, "tn_balance_cap", 5.0)))
    sam3_tn_image_root = datasetinfo.get("sam3_tn_image_root", getattr(args, "sam3_tn_image_root", None))
    sam3_tn_bbox_key = datasetinfo.get("sam3_tn_bbox_key", getattr(args, "sam3_tn_bbox_key", "sam_bbox"))
    sam3_tn_keep_failed = bool(
        datasetinfo.get("sam3_tn_keep_failed", getattr(args, "sam3_tn_keep_failed", False))
    )
    require_global_tn_verified = bool(
        datasetinfo.get(
            "require_global_tn_verified",
            getattr(args, "require_global_tn_verified", False),
        )
    )
    require_fixed_stagea_topk_exact_verified = bool(
        datasetinfo.get(
            "require_fixed_stagea_topk_exact_verified",
            getattr(args, "require_fixed_stagea_topk_exact_verified", False),
        )
    )
    fixed_stagea_topk_exact_audit = datasetinfo.get(
        "fixed_stagea_topk_exact_audit",
        getattr(args, "fixed_stagea_topk_exact_audit", None),
    )
    fixed_stagea_topk_expected_contract = datasetinfo.get(
        "fixed_stagea_topk_expected_contract",
        getattr(args, "fixed_stagea_topk_expected_contract", None),
    )
    require_proposalset_proxy_verified = bool(
        datasetinfo.get(
            "require_proposalset_proxy_verified",
            getattr(args, "require_proposalset_proxy_verified", False),
        )
    )
    require_benchmark_dataft_alltn = bool(
        datasetinfo.get(
            "require_benchmark_dataft_alltn",
            getattr(args, "require_benchmark_dataft_alltn", False),
        )
    )
    require_vlm_strict_tn = bool(
        datasetinfo.get(
            "require_vlm_strict_tn",
            getattr(args, "require_vlm_strict_tn", False),
        )
    )
    raw_single_edit_token_provenance = datasetinfo.get(
        "require_single_edit_token_provenance", False
    )
    if not isinstance(raw_single_edit_token_provenance, bool):
        raise ValueError(
            "require_single_edit_token_provenance must be an exact JSON boolean"
        )
    require_single_edit_token_provenance = raw_single_edit_token_provenance
    stage_b_gdino_adapter_ref_eval = bool(
        datasetinfo.get("stage_b_gdino_adapter_ref_eval", False)
    )
    stage_b_gdino_adapter_no_support = bool(
        datasetinfo.get("stage_b_gdino_adapter_no_support", False)
    )
    if stage_b_gdino_adapter_no_support and neg_episode_prob != 0.0:
        raise ValueError(
            "stage_b_gdino_adapter_no_support requires neg_episode_prob=0.0; "
            "internal negative-episode resampling would break paired captions "
            "and sample identity"
        )
    if stage_b_gdino_adapter_no_support and not (
        bool(getattr(args, "stage_b_gdino_score_adapter", False))
        and (
            require_benchmark_dataft_alltn
            or require_global_tn_verified
            or require_fixed_stagea_topk_exact_verified
            or require_vlm_strict_tn
            or stage_b_gdino_adapter_ref_eval
        )
    ):
        raise ValueError(
            "stage_b_gdino_adapter_no_support requires both the model adapter "
            "mode and an explicitly verified training or evaluation protocol"
        )
    if text_mask_audit_jsonl is None and build_text_token_masks:
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            text_mask_audit_jsonl = str(Path(output_dir) / "text_mask_invalid_samples.jsonl")

    tfm = make_query_transforms(
        image_set,
        fix_size=getattr(args, "fix_size", False),
        strong_aug=getattr(args, "strong_aug", False),
        args=args,
    )
    dataset = PatchEpisodeJsonlDataset(
        root=root,
        anno=anno,
        transforms=tfm,
        box_format=box_format,
        neg_episode_prob=neg_episode_prob,
        support_min_count=support_min_count,
        support_patch_size=support_patch_size,
        support_num_patches_min=support_num_patches_min,
        support_num_patches_max=support_num_patches_max,
        support_use_all_gt_classes=support_use_all_gt_classes,
        lvis_neg_category_only=lvis_neg_category_only,
        support_patch_max_per_class=support_patch_max_per_class,
        negative_max_tries=negative_max_tries,
        support_patch_tsv=support_patch_tsv,
        support_patch_bucket=support_patch_bucket,
        support_patch_class_map_json=support_patch_class_map_json,
        canonical_classes_json=canonical_classes_json,
        source=source,
        lvis_image_root=lvis_image_root,
        coco_image_root=coco_image_root,
        vg_image_roots=vg_image_roots,
        support_patch_use_embedding=support_patch_use_embedding,
        patch_emb_cache_size=patch_emb_cache_size,
        support_patch_image_root=support_patch_image_root,
        keep_only_support_gt=keep_only_support_gt,
        keep_only_patchset_gt=keep_only_patchset_gt,
        patch_text_augment=patch_text_augment,
        patch_text_aug_p_object=patch_text_aug_p_object,
        patch_text_aug_p_alias=patch_text_aug_p_alias,
        patch_text_aug_p_vg=patch_text_aug_p_vg,
        patch_text_aug_vg_jsonl=patch_text_aug_vg_jsonl,
        patch_text_aug_vg_pool_size=patch_text_aug_vg_pool_size,
        patch_text_aug_max_words=patch_text_aug_max_words,
        vg_phrase_labeler=vg_phrase_labeler,
        phrase_classifier_ckpt=phrase_classifier_ckpt,
        phrase_classifier_device=phrase_classifier_device,
        phrase_classifier_max_length=phrase_classifier_max_length,
        phrase_classifier_batch_size=phrase_classifier_batch_size,
        phrase_classifier_min_conf=phrase_classifier_min_conf,
        phrase_cache_size=phrase_cache_size,
        patch_bank_cache=patch_bank_cache,
        patch_bank_cache_path=patch_bank_cache_path,
        patch_bank_cache_write=patch_bank_cache_write,
        anno_cache=anno_cache,
        anno_cache_path=anno_cache_path,
        anno_cache_write=anno_cache_write,
        build_text_token_masks=build_text_token_masks,
        text_encoder_type=text_encoder_type,
        max_text_len=max_text_len,
        text_mask_warn_limit=text_mask_warn_limit,
        text_mask_skip_invalid_canonical=text_mask_skip_invalid_canonical,
        text_mask_audit_jsonl=text_mask_audit_jsonl,
        use_tn_category_weights=use_tn_category_weights,
        default_tn_category_weight=default_tn_category_weight,
        skip_tn_if_neg_overlaps_canonical=skip_tn_if_neg_overlaps_canonical,
        skip_ambiguous_tn=skip_ambiguous_tn,
        skip_tn_if_changed_span_not_found=skip_tn_if_changed_span_not_found,
        skip_tn_if_changed_span_empty_after_filter=skip_tn_if_changed_span_empty_after_filter,
        skip_relation_like_tn_in_v1=skip_relation_like_tn_in_v1,
        tn_balance_sampling=tn_balance_sampling,
        tn_balance_cap=tn_balance_cap,
        sam3_tn_image_root=sam3_tn_image_root,
        sam3_tn_bbox_key=sam3_tn_bbox_key,
        sam3_tn_keep_failed=sam3_tn_keep_failed,
        require_global_tn_verified=require_global_tn_verified,
        require_fixed_stagea_topk_exact_verified=(
            require_fixed_stagea_topk_exact_verified
        ),
        fixed_stagea_topk_exact_audit=fixed_stagea_topk_exact_audit,
        fixed_stagea_topk_expected_contract=fixed_stagea_topk_expected_contract,
        require_proposalset_proxy_verified=require_proposalset_proxy_verified,
        require_benchmark_dataft_alltn=require_benchmark_dataft_alltn,
        require_vlm_strict_tn=require_vlm_strict_tn,
        require_single_edit_token_provenance=(
            require_single_edit_token_provenance
        ),
        table_b_id=(
            table_b_contract.table_b_id if table_b_contract is not None else None
        ),
        table_b_scope=(
            table_b_contract.scope if table_b_contract is not None else None
        ),
        table_b_audit_sha256=(
            table_b_contract.audit_sha256
            if table_b_contract is not None
            else None
        ),
        stage_b_gdino_adapter_ref_eval=stage_b_gdino_adapter_ref_eval,
        stage_b_gdino_adapter_no_support=stage_b_gdino_adapter_no_support,
        native_patch_category_row_locked_support=(
            native_patch_category_row_locked_support
        ),
        native_patch_category_variant=native_patch_category_variant,
        native_patch_category_source_dataset=(
            native_patch_category_source_dataset
        ),
        native_patch_category_alias_bridges=native_patch_category_alias_bridges,
        strict_sample_identity=strict_sample_identity,
        lazy_jsonl=lazy_jsonl,
    )
    _validate_data_driven_ref_metas(
        dataset.metas,
        data_driven_ref_variant,
        rank_supervision=data_driven_rank_supervision,
    )
    return dataset
