import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
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

_WS_RE = re.compile(r"\s+")
_PUNC_RE = re.compile(r"[^a-z0-9 _-]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TN_CATEGORY_SEP_RE = re.compile(r"[_/,\-]+")


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


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


_TN_CATEGORY_WEIGHTS = {
    # Strong visual attributes.
    "color": 1.5,
    "hair color": 1.5,
    "clothing color": 1.5,
    "shirt color": 1.5,
    "pattern color": 1.5,
    "size color": 1.5,
    "color pattern": 1.5,
    "color and pattern": 1.5,
    "pattern and color": 1.5,
    "material color": 1.5,
    "color material": 1.5,
    "size and color": 1.5,
    "color and size": 1.5,
    # Medium visual attributes.
    "size": 1.3,
    "height": 1.2,
    "length": 1.2,
    "shape": 1.2,
    "pattern": 1.3,
    "material": 1.3,
    "texture": 1.2,
    "clothing": 1.3,
    "clothing type": 1.3,
    "clothing state": 1.0,
    "sleeve length": 1.0,
    "accessory": 1.2,
    "accessory type": 1.2,
    "state": 1.0,
    "condition": 1.0,
}

_TN_SKIP_CATEGORIES = {
    "spatial",
    "spatial relation",
    "spatial position",
    "position",
    "location",
    "distance",
    "action",
    "posture",
    "pose",
    "object",
    "object type",
    "type",
    "animal type",
    "vehicle type",
    "device type",
    "food type",
    "quantity",
    "number",
    "brand",
    "flavor",
    "topping",
    "content",
    "sport",
}

_TEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "to",
    "with",
    "without",
    "you",
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
        return out
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
    default_tn_category_weight: float = 0.0
    skip_tn_if_neg_overlaps_canonical: bool = True
    skip_ambiguous_tn: bool = True
    skip_tn_if_changed_span_not_found: bool = True
    skip_tn_if_changed_span_empty_after_filter: bool = True
    skip_relation_like_tn_in_v1: bool = True


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
        default_tn_category_weight: float = 0.0,
        skip_tn_if_neg_overlaps_canonical: bool = True,
        skip_ambiguous_tn: bool = True,
        skip_tn_if_changed_span_not_found: bool = True,
        skip_tn_if_changed_span_empty_after_filter: bool = True,
        skip_relation_like_tn_in_v1: bool = True,
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

        super().__init__(root=root, transforms=transforms)
        self.root = str(root)
        self.anno = str(anno)
        self._canonical_classes_json = canonical_classes_json
        self._support_patch_class_map_json = support_patch_class_map_json
        self._alt_image_roots = [Path(p) for p in (vg_image_roots or [])]
        self.cfg = PatchEpisodeConfig(
            box_format=box_format,
            neg_episode_prob=neg_episode_prob,
            support_min_count=support_min_count,
            support_patch_size=support_patch_size,
            support_num_patches_min=int(support_num_patches_min),
            support_num_patches_max=int(support_num_patches_max),
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

        anno_path = Path(anno)
        if anno_path.suffix.lower() == ".jsonl":
            self.metas = _read_jsonl(anno_path)
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
                else:
                    raise ValueError(f"Unsupported source={src} detected={detected} for anno={anno_path}")
        else:
            raise ValueError(f"Unsupported anno extension: {anno_path.suffix}")

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
        if support_patch_class_map_json:
            with Path(support_patch_class_map_json).open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("support_patch_class_map_json must be a JSON object mapping class_name -> canonical_id.")
            self.patch_class_map = {_norm_text(str(k)): int(v) for k, v in raw.items()}
        if support_patch_tsv:
            self.patch_bank = self._load_patch_bank_cached(Path(support_patch_tsv))
            if not self.patch_bank:
                print(
                    f"[WARN] Loaded support_patch_tsv={support_patch_tsv} but patch_bank is empty. "
                    "If your TSV 'class' column is a string name (e.g. from emb_index_from_quality.tsv), "
                    "provide support_patch_class_map_json to map class_name -> canonical_id."
                )
        self._patch_emb_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

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

    def _clean_caption_phrase(self, s: str) -> str:
        s = str(s).replace("_", " ").replace(".", " ").strip()
        s = " ".join(s.split())
        if not s:
            return "object"
        max_words = int(self.cfg.patch_text_aug_max_words)
        if (not self.cfg.build_text_token_masks) and max_words > 0:
            s = " ".join(s.split()[:max_words])
        return s

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
        category = _normalize_tn_category(category)
        if not bool(self.cfg.use_tn_category_weights):
            return float(self.cfg.default_tn_category_weight)
        if category in _TN_SKIP_CATEGORIES:
            return 0.0
        if category in _TN_CATEGORY_WEIGHTS:
            return float(_TN_CATEGORY_WEIGHTS[category])
        if category == "attribute":
            if any(t in _GENERIC_VISUAL_ATTRIBUTE_WORDS for t in changed_token_norms):
                return 1.0
            return 0.0
        return float(self.cfg.default_tn_category_weight)

    def _is_visual_content_token(self, token_norm: str) -> bool:
        if not token_norm:
            return False
        if token_norm in _TEXT_STOPWORDS:
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
        relation_mask: torch.Tensor,
        max_text_len: int,
    ) -> torch.Tensor:
        spans = []
        for tok in _tokenize_with_offsets(phrase_text):
            norm = str(tok["norm"])
            if not self._is_visual_content_token(norm):
                continue
            if self._is_relation_action_token(norm):
                continue
            spans.append((int(tok["start"]), int(tok["end"])))
        mask = self._mask_from_phrase_local_spans(tokenized, phrase_span, phrase_mask, spans, max_text_len)
        return mask & (~canonical_mask) & (~relation_mask)

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
                tokenized, phrase_text, phrase_span, phrase_mask, canonical_mask, relation_mask, max_text_len
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
                if self._is_relation_action_token(norm):
                    continue
                spans.append((int(tok["start"]), int(tok["end"])))
        mask = self._mask_from_phrase_local_spans(tokenized, phrase_span, phrase_mask, spans, max_text_len)
        return mask & (~canonical_mask) & (~relation_mask) & (~attr_neg_mask)

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
            return caption, None, None, None, None, None, None, None, None, []
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
        phrase_semantic_token_mask = torch.zeros((K, T), dtype=torch.bool)
        attr_neg_weight_mask = torch.zeros((K, T), dtype=torch.float32)
        is_tn = torch.zeros((K,), dtype=torch.bool)
        invalid_records: List[Dict[str, Any]] = []

        for k, ((span_start, _span_end), phrase_text, canonical_text, aliases) in enumerate(
            zip(slot_spans, slot_phrases, slot_canonical_texts, slot_aliases)
        ):
            record = slot_records[k] if k < len(slot_records) and isinstance(slot_records[k], dict) else {}
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
                continue

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
                continue
            canonical_to_token_mask[k] = canonical_mask

            relation_mask = self._build_relation_token_mask(tokenized, phrase_text, slot_spans[k], phrase_mask, T)
            relation_to_token_mask[k] = relation_mask

            is_text_negative = bool(record.get("text_is_negative", record.get("is_text_negative", False)))
            is_tn[k] = bool(is_text_negative)
            if not is_text_negative:
                attr_pos_mask = self._build_content_attr_mask(
                    tokenized, phrase_text, slot_spans[k], phrase_mask, canonical_mask, relation_mask, T
                )
                attr_pos_to_token_mask[k] = attr_pos_mask
                phrase_semantic_token_mask[k] = canonical_mask | attr_pos_mask
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
                if bool(self.cfg.skip_relation_like_tn_in_v1) and self._is_relation_like_category(category_norm):
                    continue
                if category_norm in _TN_SKIP_CATEGORIES:
                    continue
                if (
                    bool(self.cfg.use_tn_category_weights)
                    and category_norm not in _TN_CATEGORY_WEIGHTS
                    and category_norm != "attribute"
                    and float(self.cfg.default_tn_category_weight) <= 0.0
                ):
                    continue

                changed_tokens = self._changed_attribute_token_spans(phrase_text, replace_from, replace_to)
                if changed_tokens is None:
                    if bool(self.cfg.skip_tn_if_changed_span_not_found):
                        slot_invalid = True
                        invalid_records.append(
                            {
                                "reason": "tn_changed_span_not_found",
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
                    if self._is_relation_action_token(norm):
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
                # Unknown / relation-like / skipped TN: keep masks empty so the criterion
                # skips text supervision for this slot.
                continue

            attr_neg_to_token_mask[k] = neg_mask
            attr_neg_weight_mask[k] = neg_weight
            positive_phrase = record.get("positive_phrase", None) or record.get("try_tn_head_phrase", None)
            attr_pos_mask = self._build_shared_attr_mask(
                tokenized,
                phrase_text,
                positive_phrase,
                slot_spans[k],
                phrase_mask,
                canonical_mask,
                relation_mask,
                neg_mask,
                T,
            )
            attr_pos_to_token_mask[k] = attr_pos_mask
            phrase_semantic_token_mask[k] = canonical_mask | attr_pos_mask | neg_mask

        return (
            caption,
            phrase_to_token_mask,
            canonical_to_token_mask,
            attr_pos_to_token_mask,
            attr_neg_to_token_mask,
            relation_to_token_mask,
            phrase_semantic_token_mask,
            is_tn,
            attr_neg_weight_mask,
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
            "version": 4,
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
        for img in lvis_data.get("images", []) or []:
            try:
                img_id = int(img["id"])
            except Exception:
                continue
            ne_ids = img.get("not_exhaustive_category_ids", None)
            if not ne_ids:
                continue
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
                "positive_phrase",
                "replace_from",
                "replace_to",
                "replace_category",
                "try_tn_head",
                "try_tn_head_phrase",
                "tn_type",
                "visual_filter_status",
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
        candidates = [
            int(c)
            for c, cnt in counts.items()
            if (cnt >= self.cfg.support_min_count and int(c) not in forbidden)
        ]
        if not candidates:
            candidates = [int(c) for c in counts.keys() if int(c) not in forbidden]
        if self.patch_bank is not None:
            candidates = [c for c in candidates if len(self.patch_bank.get(int(c), [])) > 0]
        if not candidates:
            raise RuntimeError("No eligible support classes for multi-patch episode.")

        k_max = min(int(self.cfg.support_num_patches_max), len(candidates))
        k_min = min(max(1, int(self.cfg.support_num_patches_min)), k_max)
        k = random.randint(k_min, k_max) if k_min < k_max else k_max
        return random.sample(candidates, k=k)

    def _sample_support_patch_for_class(
        self,
        support_class: int,
        fallback_img: Optional[Image.Image] = None,
        fallback_boxes_xyxy: Optional[torch.Tensor] = None,
        fallback_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        max_resample = 20
        for attempt in range(max_resample):
            meta = self.metas[index] if attempt == 0 else self.metas[random.randrange(0, len(self.metas))]
            rel_path = meta.get("filename", meta.get("file_name", None))
            if rel_path is None:
                continue

            not_exhaustive = set(int(x) for x in (meta.get("not_exhaustive_cids", []) or []))

            try:
                img = self._open_image(rel_path)
            except FileNotFoundError:
                continue
            w, h = img.size
            instance_records = self._extract_instance_records(meta)
            if not instance_records:
                continue
            boxes_xyxy, labels = self._extract_instances(meta)

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
                if int(self.cfg.support_num_patches_max) > 1:
                    # Multi-patch: choose multiple support classes (canonical ids) and sample one patch per class.
                    try:
                        support_classes = self._choose_support_classes(labels, forbidden_classes=not_exhaustive)
                    except Exception:
                        continue
                    support_classes_t = torch.as_tensor(support_classes, dtype=torch.int64)

                    patch_list: List[torch.Tensor] = []
                    ok = True
                    for cid in support_classes:
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
                        phrase_text, canonical_text, alias_candidates = self._get_slot_text_for_record(
                            instance_records[support_i], int(cid)
                        )
                        slot_phrases.append(phrase_text)
                        slot_canonical_texts.append(canonical_text)
                        slot_aliases.append(alias_candidates)
                        slot_text_is_negative.append(bool(instance_records[support_i].get("text_is_negative", False)))
                        slot_records.append(instance_records[support_i])
                    if (not ok) or (len(patch_list) != int(support_classes_t.numel())):
                        continue

                    # Optionally keep only GT boxes belonging to the selected support classes.
                    if bool(self.cfg.keep_only_patchset_gt):
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
                    is_negative = False
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
                            try:
                                support_class, support_i = self._choose_support(labels, forbidden_classes=not_exhaustive)
                            except Exception:
                                continue
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
            phrase_semantic_token_mask = None
            is_tn = None
            attr_neg_weight_mask = None
            invalid_text_mask_records: List[Dict[str, Any]] = []
            if int(self.cfg.support_num_patches_max) > 1:
                support_classes_t, patches_or_emb = support
                if any(int(x) in not_exhaustive for x in support_classes_t.tolist()):
                    continue
                K = int(support_classes_t.numel())
                if self.cfg.build_text_token_masks:
                    if len(slot_phrases) != K:
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
                        phrase_semantic_token_mask,
                        is_tn,
                        attr_neg_weight_mask,
                        invalid_text_mask_records,
                    ) = self._build_slot_text_masks(
                        phrases, slot_canonical_texts, slot_aliases, slot_records=slot_records
                    )
                    negative_to_token_mask = attr_neg_to_token_mask
                else:
                    phrases = self._sample_text_phrases(K)
                    caption = " ".join([f"{p} ." for p in phrases])
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
                    slot_records = [
                        support_record
                        if support_record is not None
                        else {"phrase": phrase_text, "head_phrase": canonical_text, "text_is_negative": False}
                    ]
                    phrases = [phrase_text]
                    (
                        caption,
                        phrase_to_token_mask,
                        canonical_to_token_mask,
                        attr_pos_to_token_mask,
                        attr_neg_to_token_mask,
                        relation_to_token_mask,
                        phrase_semantic_token_mask,
                        is_tn,
                        attr_neg_weight_mask,
                        invalid_text_mask_records,
                    ) = self._build_slot_text_masks(
                        phrases, [canonical_text], [alias_candidates], slot_records=slot_records
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
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])
            target["caption"] = caption
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
            if phrase_semantic_token_mask is not None:
                target["phrase_semantic_token_mask"] = phrase_semantic_token_mask
            if is_tn is not None:
                target["is_tn"] = is_tn
            if attr_neg_weight_mask is not None:
                target["attr_neg_weight_mask"] = attr_neg_weight_mask
            if negative_to_token_mask is not None:
                target["negative_to_token_mask"] = negative_to_token_mask
            if int(self.cfg.support_num_patches_max) > 1:
                target["support_classes"] = support_classes_t
                target["support_class"] = support_classes_t[:1]  # legacy logging key
                if self.cfg.support_patch_use_embedding:
                    target["patch_global"] = patches_or_emb
                else:
                    target["patches"] = patches_or_emb
            else:
                target["support_class"] = support_class
                if self.cfg.support_patch_use_embedding:
                    target["patch_global"] = patch
                else:
                    target["patch"] = patch
            target["is_negative_episode"] = torch.as_tensor([1 if is_negative else 0], dtype=torch.int64)

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        raise RuntimeError(
            f"Failed to sample a valid episode after {max_resample} attempts (likely too many not_exhaustive-only images)."
        )


def build_patch_episode(image_set: str, args, datasetinfo: Dict[str, Any]):
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
        datasetinfo.get("default_tn_category_weight", getattr(args, "default_tn_category_weight", 0.0))
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
        datasetinfo.get("skip_relation_like_tn_in_v1", getattr(args, "skip_relation_like_tn_in_v1", True))
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
    return PatchEpisodeJsonlDataset(
        root=root,
        anno=anno,
        transforms=tfm,
        box_format=box_format,
        neg_episode_prob=neg_episode_prob,
        support_min_count=support_min_count,
        support_patch_size=support_patch_size,
        support_num_patches_min=support_num_patches_min,
        support_num_patches_max=support_num_patches_max,
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
    )
