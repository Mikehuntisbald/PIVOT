#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _norm_phrase(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


HeadRecord = Dict[str, Optional[str]]
ClassNameMaps = Dict[int, Dict[str, Optional[str]]]


def _load_canonical_name_maps(data_root: Path) -> ClassNameMaps:
    path = data_root / "canonical_classes_with_aliases.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: ClassNameMaps = {}
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict) or "id" not in row:
            continue
        try:
            cid = int(row["id"])
        except Exception:
            continue
        raw = row.get("raw_name", None)
        norm = row.get("norm_name", None) or row.get("base_name", None) or raw
        out[cid] = {
            "raw_name": str(raw) if raw is not None else None,
            "norm_name": str(norm) if norm is not None else None,
        }
    return out


class _HeadClassifierResolver:
    def __init__(
        self,
        ckpt_path: Path,
        *,
        device: str = "cpu",
        batch_size: int = 128,
        max_length: Optional[int] = None,
        min_conf: float = 0.0,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoTokenizer, BertModel

        from models.GroundingDINO.bertwarper import BertModelWarper

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Head classifier checkpoint not found: {ckpt_path}. "
                "Pass --disable-refcoco-head-classifier to keep annotation class_id."
            )

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.min_conf = float(min_conf)
        self.cache: Dict[str, Tuple[Optional[int], float]] = {}

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.bert_model_name = str(ckpt.get("bert_model_name", "bert-base-uncased"))
        self.num_classes = int(ckpt.get("num_classes", 2048))
        cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
        self.max_length = int(max_length or cfg.get("max_len", 24))

        try:
            bert_cfg = AutoConfig.from_pretrained(self.bert_model_name, local_files_only=True)
        except Exception:
            bert_cfg = AutoConfig.from_pretrained(self.bert_model_name)
        bert = BertModel(bert_cfg)

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
                pooled = self.dropout(outputs.pooler_output)
                return self.classifier(pooled)

        self.model = _Model(bert, self.num_classes)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading head classifier: {unexpected[:10]}")
        bad_missing = [k for k in missing if "position_ids" not in k]
        if bad_missing:
            raise RuntimeError(f"Missing keys when loading head classifier: {bad_missing[:10]}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name, use_fast=True, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name, use_fast=True)
        self.model.eval().to(self.device)

    @staticmethod
    def _canon_head(head: Optional[str]) -> str:
        text = str(head or "").strip()
        if text and text[-1] not in ".?!":
            text += "."
        return text

    def predict_many(self, heads: List[Optional[str]]) -> List[Tuple[Optional[int], float]]:
        torch = self.torch
        keys = [self._canon_head(h) for h in heads]
        missing_keys = [k for k in dict.fromkeys(keys) if k and k not in self.cache]
        with torch.inference_mode():
            for i in range(0, len(missing_keys), self.batch_size):
                chunk = missing_keys[i : i + self.batch_size]
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
                for key, c, p in zip(chunk, conf.detach().cpu().tolist(), pred.detach().cpu().tolist()):
                    if float(c) >= self.min_conf:
                        self.cache[key] = (int(p), float(c))
                    else:
                        self.cache[key] = (None, float(c))

        out: List[Tuple[Optional[int], float]] = []
        for key in keys:
            if not key:
                out.append((None, 0.0))
            else:
                out.append(self.cache.get(key, (None, 0.0)))
        return out

    def predict_one(self, head: Optional[str]) -> Tuple[Optional[int], float]:
        return self.predict_many([head])[0]


def _load_head_phrase_maps(data_root: Path) -> Tuple[Dict[Tuple, HeadRecord], Dict[Tuple, HeadRecord]]:
    exact: Dict[Tuple, HeadRecord] = {}
    loose: Dict[Tuple, HeadRecord] = {}
    pair_dirs = [
        data_root / "data_proc" / "refcoco_text_pairs",
        data_root / "refcoco_text_pairs",
    ]
    pair_names = {
        "refcoco_unc": "refcoco_unc_pairs.jsonl",
        "refcoco+_unc": "refcoco+_unc_pairs.jsonl",
        "refcocog_google": "refcocog_google_pairs.jsonl",
    }
    for pair_dir in pair_dirs:
        for source_name, filename in pair_names.items():
            path = pair_dir / filename
            if not path.exists():
                continue
            for row in _iter_jsonl(path):
                head_phrase = row.get("head_phrase")
                head = row.get("head")
                if not isinstance(head_phrase, str) or not head_phrase.strip():
                    continue
                if not isinstance(head, str) or not head.strip():
                    head = None
                record = {"head_phrase": head_phrase.strip(), "head": (head.strip() if head else None)}
                ref_id = row.get("ref_id")
                ann_id = row.get("ann_id")
                image_id = row.get("image_id")
                raw_phrase = _norm_phrase(row.get("raw_phrase"))
                if ref_id is None or ann_id is None or image_id is None or not raw_phrase:
                    continue
                exact.setdefault((source_name, int(ref_id), int(ann_id), int(image_id), raw_phrase), record)
                loose.setdefault((source_name, int(ref_id), raw_phrase), record)
    return exact, loose


def _lookup_head_record(
    row: Dict,
    phrase: str,
    exact_map: Dict[Tuple, HeadRecord],
    loose_map: Dict[Tuple, HeadRecord],
) -> Optional[HeadRecord]:
    source_name = row.get("pair_source") or row.get("source")
    ref_id = row.get("ref_id")
    ann_id = row.get("ann_id")
    image_id = row.get("image_id")
    norm_phrase = _norm_phrase(phrase)
    if not isinstance(source_name, str) or ref_id is None or not norm_phrase:
        return None

    if ann_id is not None and image_id is not None:
        hit = exact_map.get((source_name, int(ref_id), int(ann_id), int(image_id), norm_phrase))
        if hit:
            return hit
    return loose_map.get((source_name, int(ref_id), norm_phrase))


def _head_record_with_fallback(
    row: Dict,
    phrase: str,
    exact_map: Dict[Tuple, HeadRecord],
    loose_map: Dict[Tuple, HeadRecord],
) -> HeadRecord:
    hit = _lookup_head_record(row, phrase, exact_map, loose_map)
    if hit:
        return hit
    head_phrase = None
    head = None
    for key in ("try_tn_head_phrase", "try_tn_head"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            head_phrase = value.strip()
            break
    value = row.get("try_tn_head")
    if isinstance(value, str) and value.strip():
        head = value.strip()
    return {"head_phrase": head_phrase, "head": head}


def _resolve_coco_roots(
    data_root: Path,
    coco_train_root: Optional[Path] = None,
    coco_val_root: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    train_root = coco_train_root if coco_train_root is not None else (data_root / "COCO" / "coco2014" / "train2014")
    val_root = coco_val_root if coco_val_root is not None else (data_root / "COCO" / "coco2014" / "val2014")
    if not train_root.exists():
        raise SystemExit(f"Missing COCO train2014 root: {train_root}")
    return train_root, (val_root if val_root.exists() else None)


def _canonical_coco_name(file_name: Optional[str], image_id: Optional[int]) -> Tuple[Optional[str], str]:
    raw_name = Path(str(file_name)).name if file_name else ""
    split = "train2014"
    if raw_name.startswith("COCO_val2014_"):
        split = "val2014"

    if raw_name:
        if raw_name.startswith("COCO_train2014_") or raw_name.startswith("COCO_val2014_"):
            stem = raw_name[:-4] if raw_name.endswith(".jpg") else raw_name
            parts = stem.split("_")
            if len(parts) >= 3:
                base = "_".join(parts[:3])
                return base + ".jpg", split
            return raw_name, split
        return raw_name, split

    if image_id is None:
        return None, split
    return f"COCO_{split}_{int(image_id):012d}.jpg", split


def _resolve_coco_image(file_name: Optional[str], image_id: Optional[int], train_root: Path, val_root: Optional[Path]) -> Optional[str]:
    if file_name and ("/" in str(file_name)):
        file_path = Path(str(file_name))
        if file_path.is_file():
            return str(file_path)

    name, split = _canonical_coco_name(file_name, image_id)
    if not name:
        return None
    if split == "val2014":
        if val_root is None:
            return None
        path = val_root / name
    else:
        path = train_root / name
    return str(path)


def _pick_bbox(row: Dict) -> Optional[List[float]]:
    for key in ("bbox", "gt_bbox", "sam_bbox"):
        value = row.get(key)
        if not isinstance(value, list) or len(value) != 4:
            continue
        try:
            box = [float(x) for x in value]
        except Exception:
            continue
        if box[2] <= 0 or box[3] <= 0:
            continue
        return box
    return None


def _canonical_name(row: Dict) -> str:
    for key in ("class_norm_name", "class_raw_name", "category_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "object"


def _classifier_head(row: Dict, head_record: HeadRecord) -> Optional[str]:
    head = head_record.get("head") or row.get("try_tn_head") or head_record.get("head_phrase")
    return head if isinstance(head, str) and head.strip() else None


def _row_with_head_class_override(
    row: Dict,
    *,
    head_record: HeadRecord,
    resolver: Optional[_HeadClassifierResolver],
    class_name_maps: ClassNameMaps,
    stats: Dict[str, int],
) -> Dict:
    if resolver is None:
        return row

    head = _classifier_head(row, head_record)
    stats["head_classifier_seen"] = stats.get("head_classifier_seen", 0) + 1
    pred_cid, conf = resolver.predict_one(head)
    if pred_cid is None:
        stats["head_classifier_no_pred"] = stats.get("head_classifier_no_pred", 0) + 1
        return row

    out = dict(row)
    stats["head_classifier_override"] = stats.get("head_classifier_override", 0) + 1
    try:
        old_cid = int(row.get("class_id"))
    except Exception:
        old_cid = None
    if old_cid is not None and old_cid != int(pred_cid):
        stats["head_classifier_changed_class"] = stats.get("head_classifier_changed_class", 0) + 1

    for key in ("category_name", "class_id", "class_raw_name", "class_norm_name", "label_match_type"):
        if key in row:
            out[f"refcoco_annotation_{key}"] = copy.deepcopy(row.get(key))

    names = class_name_maps.get(int(pred_cid), {})
    out["class_id"] = int(pred_cid)
    out["class_raw_name"] = names.get("raw_name") or row.get("class_raw_name") or row.get("category_name")
    out["class_norm_name"] = names.get("norm_name") or out["class_raw_name"]
    out["label_match_type"] = "head_classifier"
    out["class_id_source"] = "head_classifier"
    out["head_classifier_head"] = head
    out["head_classifier_class_id"] = int(pred_cid)
    out["head_classifier_conf"] = float(conf)
    out["head_classifier_bert_model_name"] = resolver.bert_model_name
    return out


def _maybe_add_tn_metadata(instance: Dict, row: Dict) -> None:
    optional_fields = (
        "try_tn_rule",
        "tn_edits",
        "replace_from",
        "replace_to",
        "replace_category",
        "replace_span",
        "try_tn_method",
        "try_tn_head",
        "try_tn_head_phrase",
        "try_tn_retry_count",
        "try_tn_failure_reason",
        "vlm_verdict",
        "vlm_reason",
        "vlm_raw_answer",
        "visual_filter_status",
        "visual_filter_reason",
        "candidate_cache_status",
        "candidate_cache_version",
        "target_bbox_used",
        "target_bbox_source",
        "proposal_num",
    )
    for key in optional_fields:
        value = row.get(key)
        if value is None:
            continue
        instance[key] = copy.deepcopy(value)


def _source_name_from_row(row: Dict, fallback: str, suffix: str) -> str:
    dataset = row.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        return f"{dataset.strip()}_{suffix}"
    pair_source = row.get("pair_source")
    if isinstance(pair_source, str) and pair_source.strip():
        return f"{pair_source.strip()}_{suffix}"
    return fallback


def _default_existing_paths(paths: Iterable[Path]) -> List[Path]:
    return [path for path in paths if path.exists()]


def _build_meta(
    row: Dict,
    *,
    phrase: str,
    filename: str,
    text_is_negative: bool,
    source_name: str,
    head_phrase: Optional[str] = None,
    head: Optional[str] = None,
) -> Dict:
    instance = {
        "bbox": _pick_bbox(row),
        "class_id": int(row["class_id"]),
        "raw_phrase": phrase,
        "head_phrase": head_phrase,
        "head": head,
        "canonical_name": _canonical_name(row),
        "text_is_negative": bool(text_is_negative),
        "positive_phrase": row.get("sent"),
        "pair_source": row.get("pair_source"),
    }
    for key in (
        "category_name",
        "class_raw_name",
        "class_norm_name",
        "label_match_type",
        "class_id_source",
        "refcoco_annotation_category_name",
        "refcoco_annotation_class_id",
        "refcoco_annotation_class_raw_name",
        "refcoco_annotation_class_norm_name",
        "refcoco_annotation_label_match_type",
        "head_classifier_head",
        "head_classifier_class_id",
        "head_classifier_conf",
        "head_classifier_bert_model_name",
    ):
        if key in row:
            instance[key] = copy.deepcopy(row[key])
    if text_is_negative:
        _maybe_add_tn_metadata(instance, row)
    return {
        "filename": filename,
        "source": source_name,
        "image_id": row.get("image_id"),
        "ann_id": row.get("ann_id"),
        "ref_id": row.get("ref_id"),
        "sent_id": row.get("sent_id"),
        "instances": [instance],
    }


def _write_positive_jsonl(
    out_path: Path,
    src_path: Path,
    source_name: str,
    train_root: Path,
    val_root: Optional[Path],
    exact_head_phrase_map: Dict[Tuple, HeadRecord],
    loose_head_phrase_map: Dict[Tuple, HeadRecord],
    head_classifier: Optional[_HeadClassifierResolver],
    class_name_maps: ClassNameMaps,
    stats: Dict[str, int],
) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for row in _iter_jsonl(src_path):
            phrase = row.get("sent")
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            if "class_id" not in row:
                continue
            bbox = _pick_bbox(row)
            if bbox is None:
                continue
            filename = _resolve_coco_image(row.get("file_name"), row.get("image_id"), train_root, val_root)
            if filename is None:
                continue
            head_record = _head_record_with_fallback(row, phrase, exact_head_phrase_map, loose_head_phrase_map)
            row = _row_with_head_class_override(
                row,
                head_record=head_record,
                resolver=head_classifier,
                class_name_maps=class_name_maps,
                stats=stats,
            )
            meta = _build_meta(
                row,
                phrase=phrase.strip(),
                filename=filename,
                text_is_negative=False,
                source_name=source_name,
                head_phrase=head_record.get("head_phrase"),
                head=head_record.get("head"),
            )
            if meta["instances"][0]["bbox"] is None:
                continue
            out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_tn_jsonl(
    out_path: Path,
    src_paths: List[Path],
    train_root: Path,
    val_root: Optional[Path],
    exact_head_phrase_map: Dict[Tuple, HeadRecord],
    loose_head_phrase_map: Dict[Tuple, HeadRecord],
    head_classifier: Optional[_HeadClassifierResolver],
    class_name_maps: ClassNameMaps,
    stats: Dict[str, int],
) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for src_path in src_paths:
            fallback_source_name = src_path.parent.stem if src_path.stem in {"accepted", "rejected", "unknown", "skipped"} else src_path.stem
            for row in _iter_jsonl(src_path):
                visual_filter_status = row.get("visual_filter_status")
                if visual_filter_status is not None:
                    if visual_filter_status != "accept":
                        continue
                    phrase = row.get("try_tn")
                    source_name = _source_name_from_row(row, fallback_source_name, "tn_vlm_filter")
                else:
                    vlm_verdict = row.get("vlm_verdict")
                    if isinstance(vlm_verdict, str) and vlm_verdict.strip() and vlm_verdict != "absent":
                        continue
                    phrase = row.get("vlm_tn") or row.get("try_tn")
                    source_name = _source_name_from_row(row, fallback_source_name, "tn")
                if not isinstance(phrase, str) or not phrase.strip():
                    continue
                if "class_id" not in row:
                    continue
                bbox = _pick_bbox(row)
                if bbox is None:
                    continue
                filename = _resolve_coco_image(row.get("file_name"), row.get("image_id"), train_root, val_root)
                if filename is None:
                    continue
                head_record = _head_record_with_fallback(
                    row,
                    row.get("sent", phrase),
                    exact_head_phrase_map,
                    loose_head_phrase_map,
                )
                row = _row_with_head_class_override(
                    row,
                    head_record=head_record,
                    resolver=head_classifier,
                    class_name_maps=class_name_maps,
                    stats=stats,
                )
                meta = _build_meta(
                    row,
                    phrase=phrase.strip(),
                    filename=filename,
                    text_is_negative=True,
                    source_name=source_name,
                    head_phrase=head_record.get("head_phrase"),
                    head=head_record.get("head"),
                )
                if meta["instances"][0]["bbox"] is None:
                    continue
                out_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                count += 1
    return count


def _collect_refcoco_classifier_heads(
    refcocoplus_src: Path,
    refcocog_src: Path,
    tn_srcs: List[Path],
    exact_head_phrase_map: Dict[Tuple, HeadRecord],
    loose_head_phrase_map: Dict[Tuple, HeadRecord],
) -> List[str]:
    heads: List[str] = []

    for src_path in (refcocoplus_src, refcocog_src):
        for row in _iter_jsonl(src_path):
            phrase = row.get("sent")
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            if "class_id" not in row or _pick_bbox(row) is None:
                continue
            head_record = _head_record_with_fallback(row, phrase, exact_head_phrase_map, loose_head_phrase_map)
            head = _classifier_head(row, head_record)
            if head:
                heads.append(head)

    for src_path in tn_srcs:
        fallback_source_name = src_path.parent.stem if src_path.stem in {"accepted", "rejected", "unknown", "skipped"} else src_path.stem
        for row in _iter_jsonl(src_path):
            visual_filter_status = row.get("visual_filter_status")
            if visual_filter_status is not None:
                if visual_filter_status != "accept":
                    continue
                phrase = row.get("try_tn")
            else:
                vlm_verdict = row.get("vlm_verdict")
                if isinstance(vlm_verdict, str) and vlm_verdict.strip() and vlm_verdict != "absent":
                    continue
                phrase = row.get("vlm_tn") or row.get("try_tn")
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            if "class_id" not in row or _pick_bbox(row) is None:
                continue
            head_record = _head_record_with_fallback(
                row,
                row.get("sent", phrase),
                exact_head_phrase_map,
                loose_head_phrase_map,
            )
            head = _classifier_head(row, head_record)
            if head:
                heads.append(head)
    return heads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/media/haoyi/T9/data")
    ap.add_argument("--out-dir", default="/media/haoyi/T9/data/patch_episode_prebuilt")
    ap.add_argument("--coco-train-root", default=None)
    ap.add_argument("--coco-val-root", default=None)
    ap.add_argument("--refcocoplus-src", default=None)
    ap.add_argument("--refcocog-src", default=None)
    ap.add_argument("--tn-srcs", nargs="*", default=None)
    ap.add_argument(
        "--refcoco-head-classifier-ckpt",
        default=str(Path(__file__).resolve().parents[1] / "exp_vg_multiclass_clean" / "best.pt"),
        help="Phrase/head classifier checkpoint used to override RefCOCO annotation class_id from head.",
    )
    ap.add_argument(
        "--refcoco-head-classifier-device",
        default="cpu",
        help="Device for the head classifier override pass, e.g. cpu or cuda.",
    )
    ap.add_argument("--refcoco-head-classifier-batch-size", type=int, default=128)
    ap.add_argument(
        "--refcoco-head-classifier-min-conf",
        type=float,
        default=0.05,
        help="Keep annotation class_id when the head classifier confidence is below this threshold.",
    )
    ap.add_argument(
        "--disable-refcoco-head-classifier",
        action="store_true",
        help="Keep legacy RefCOCO annotation-derived class_id instead of overriding from head classifier.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root, val_root = _resolve_coco_roots(
        data_root,
        coco_train_root=(Path(args.coco_train_root) if args.coco_train_root else None),
        coco_val_root=(Path(args.coco_val_root) if args.coco_val_root else None),
    )
    exact_head_phrase_map, loose_head_phrase_map = _load_head_phrase_maps(data_root)
    class_name_maps = _load_canonical_name_maps(data_root)

    head_classifier = None
    override_stats: Dict[str, int] = {}
    if not bool(args.disable_refcoco_head_classifier):
        head_classifier = _HeadClassifierResolver(
            Path(args.refcoco_head_classifier_ckpt),
            device=str(args.refcoco_head_classifier_device),
            batch_size=int(args.refcoco_head_classifier_batch_size),
            min_conf=float(args.refcoco_head_classifier_min_conf),
        )
        print(
            "[INFO] RefCOCO head classifier override enabled: "
            f"ckpt={args.refcoco_head_classifier_ckpt} "
            f"device={args.refcoco_head_classifier_device} "
            f"min_conf={args.refcoco_head_classifier_min_conf}"
        )

    refcocoplus_src = (
        Path(args.refcocoplus_src)
        if args.refcocoplus_src
        else (data_root / "SAM3" / "out" / "refcocoplus_sam3_washed_try_tn_llm_head.jsonl")
    )
    refcocog_src = (
        Path(args.refcocog_src)
        if args.refcocog_src
        else (data_root / "SAM3" / "out" / "refcocog_sam3_washed_try_tn_llm_head.jsonl")
    )
    tn_srcs = (
        [Path(p) for p in args.tn_srcs]
        if args.tn_srcs
        else _default_existing_paths(
            [
                data_root / "SAM3" / "output" / "refcoco_sam3_washed_try_tn_llm_head_candidates_vlm_filter" / "accepted.jsonl",
                data_root / "SAM3" / "output" / "refcocoplus_sam3_washed_try_tn_llm_head_candidates_vlm_filter" / "accepted.jsonl",
                data_root / "SAM3" / "output" / "refcocog_sam3_washed_try_tn_llm_head_candidates_vlm_filter" / "accepted.jsonl",
            ]
        )
    )

    if not tn_srcs:
        raise SystemExit("No TN source files found. Pass --tn-srcs explicitly or generate *_vlm_filter/accepted.jsonl files.")

    for path in [refcocoplus_src, refcocog_src] + tn_srcs:
        if not path.exists():
            raise SystemExit(f"Missing source file: {path}")

    if head_classifier is not None:
        heads = _collect_refcoco_classifier_heads(
            refcocoplus_src,
            refcocog_src,
            tn_srcs,
            exact_head_phrase_map,
            loose_head_phrase_map,
        )
        unique_heads = {
            head_classifier._canon_head(head)
            for head in heads
            if head_classifier._canon_head(head)
        }
        print(
            "[INFO] Precomputing RefCOCO head classifier cache: "
            f"rows={len(heads)} unique_heads={len(unique_heads)}"
        )
        head_classifier.predict_many(heads)
        print(f"[INFO] Head classifier cache ready: cached={len(head_classifier.cache)}")

    out_plus = out_dir / "refcocoplus_stageb_phrase_v1.jsonl"
    out_gog = out_dir / "refcocog_stageb_phrase_v1.jsonl"
    out_tn = out_dir / "refexp_tn_stageb_v1.jsonl"

    n_plus = _write_positive_jsonl(
        out_plus,
        refcocoplus_src,
        "refcocoplus_phrase",
        train_root,
        val_root,
        exact_head_phrase_map,
        loose_head_phrase_map,
        head_classifier,
        class_name_maps,
        override_stats,
    )
    n_gog = _write_positive_jsonl(
        out_gog,
        refcocog_src,
        "refcocog_phrase",
        train_root,
        val_root,
        exact_head_phrase_map,
        loose_head_phrase_map,
        head_classifier,
        class_name_maps,
        override_stats,
    )
    n_tn = _write_tn_jsonl(
        out_tn,
        tn_srcs,
        train_root,
        val_root,
        exact_head_phrase_map,
        loose_head_phrase_map,
        head_classifier,
        class_name_maps,
        override_stats,
    )

    summary = {
        "coco_roots": {
            "train2014": str(train_root),
            "val2014": (str(val_root) if val_root is not None else None),
        },
        "outputs": {
            "refcocoplus": {"path": str(out_plus), "count": n_plus},
            "refcocog": {"path": str(out_gog), "count": n_gog},
            "tn": {"path": str(out_tn), "count": n_tn},
        },
        "inputs": {
            "refcocoplus": str(refcocoplus_src),
            "refcocog": str(refcocog_src),
            "tn": [str(path) for path in tn_srcs],
        },
        "refcoco_head_classifier_override": {
            "enabled": head_classifier is not None,
            "ckpt": str(args.refcoco_head_classifier_ckpt),
            "device": str(args.refcoco_head_classifier_device),
            "min_conf": float(args.refcoco_head_classifier_min_conf),
            "stats": override_stats,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
