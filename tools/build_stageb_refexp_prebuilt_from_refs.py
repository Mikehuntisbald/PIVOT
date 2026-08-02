#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")


def _norm_text(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "").replace("_", " ").replace(".", " ").strip().lower())


def _clean_phrase(value: Any) -> str:
    text = _WS_RE.sub(" ", str(value or "").replace("_", " ").replace(".", " ").strip())
    return text or "object"


def _load_canonical_name_maps(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    alias_values: List[Tuple[str, int]] = []
    for row in data:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        cid = int(row["id"])
        preferred = row.get("base_name") or row.get("norm_name") or row.get("raw_name")
        if isinstance(preferred, str) and preferred.strip():
            id_to_name.setdefault(cid, _clean_phrase(preferred))
        values = [row.get("raw_name"), row.get("norm_name"), row.get("base_name")]
        values.extend(row.get("synonyms") or [])
        for value in values:
            if isinstance(value, str) and value.strip():
                name_to_id.setdefault(_norm_text(value), cid)
        for alias in row.get("aliases") or []:
            if not isinstance(alias, dict):
                continue
            for key in ("name", "norm_name"):
                value = alias.get(key)
                if isinstance(value, str) and value.strip():
                    alias_values.append((value, cid))
    # Preserve exact canonical/synonym priority, then use aliases only to fill
    # names such as COCO's "keyboard", "microwave", and "remote".
    for value, cid in alias_values:
        name_to_id.setdefault(_norm_text(value), cid)
    return name_to_id, id_to_name


def _load_instances(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    anns = {int(row["id"]): row for row in data.get("annotations", [])}
    images = {int(row["id"]): row for row in data.get("images", [])}
    cats = {int(row["id"]): str(row.get("name", "")) for row in data.get("categories", [])}
    return anns, images, cats


def _image_path(data_root: Path, image: Dict[str, Any]) -> str:
    filename = str(image.get("file_name", ""))
    candidates = [
        data_root / "COCO" / "coco2014" / "train2014" / filename,
        data_root / "COCO" / "coco2014" / "val2014" / filename,
        data_root / "COCO" / "coco2017" / "train2017" / "train2017" / filename.replace("COCO_train2014_", ""),
        data_root / "COCO" / "coco2017" / "val2017" / filename.replace("COCO_val2014_", ""),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _iter_ref_sentences(refs_path: Path, instances_path: Path, split: str) -> Iterator[Dict[str, Any]]:
    refs = pickle.load(refs_path.open("rb"))
    anns, images, cats = _load_instances(instances_path)
    for ref in refs:
        if str(ref.get("split")) != str(split):
            continue
        ann = anns.get(int(ref["ann_id"]))
        image = images.get(int(ref["image_id"]))
        if ann is None or image is None:
            continue
        category_name = cats.get(int(ann.get("category_id", -1)), "")
        for sent in ref.get("sentences", []) or []:
            phrase = _clean_phrase(sent.get("sent") or sent.get("raw"))
            yield {
                "ref": ref,
                "ann": ann,
                "image": image,
                "category_name": category_name,
                "phrase": phrase,
                "sent_id": int(sent.get("sent_id", -1)),
            }


def _class_record(
    *,
    ann: Dict[str, Any],
    category_name: str,
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
) -> Dict[str, Any]:
    cid = name_to_id.get(_norm_text(category_name), int(ann.get("category_id", -1)))
    canon = id_to_name.get(int(cid), _clean_phrase(category_name))
    return {
        "class_id": int(cid),
        "head": canon,
        "head_phrase": canon,
        "canonical_name": canon,
        "class_id_source": "category_fallback",
    }


def _meta_from_row(
    *,
    data_root: Path,
    dataset: str,
    splitby: str,
    split: str,
    row: Dict[str, Any],
    class_rec: Dict[str, Any],
    phrase: str,
    text_is_negative: bool,
    positive_phrase: Optional[str] = None,
    replace_from: Optional[str] = None,
    replace_to: Optional[str] = None,
    replace_category: Optional[str] = None,
) -> Dict[str, Any]:
    ref = row["ref"]
    ann = row["ann"]
    image = row["image"]
    instance = {
        "bbox": ann["bbox"],
        "class_id": int(class_rec["class_id"]),
        "raw_phrase": phrase,
        "head_phrase": _clean_phrase(class_rec["head_phrase"]),
        "head": _clean_phrase(class_rec["head"]),
        "canonical_name": _clean_phrase(class_rec["canonical_name"]),
        "positive_phrase": positive_phrase or phrase,
        "text_is_negative": bool(text_is_negative),
        "pair_source": f"{dataset}_{splitby}",
        "category_name": row["category_name"],
        "class_id_source": class_rec["class_id_source"],
        "refcoco_category_id": int(ann.get("category_id", -1)),
    }
    if text_is_negative:
        instance.update(
            {
                "replace_from": replace_from,
                "replace_to": replace_to,
                "replace_category": replace_category or "attribute",
                "try_tn": phrase,
                "try_tn_head": _clean_phrase(class_rec["head"]),
                "try_tn_head_phrase": _clean_phrase(class_rec["head_phrase"]),
                "try_tn_method": "synthetic_rule",
                "try_tn_rule": "single_token_attribute_swap",
            }
        )
    return {
        "filename": _image_path(data_root, image),
        "source": f"{dataset}_{splitby}_{split}",
        "image_id": int(ref["image_id"]),
        "ann_id": int(ref["ann_id"]),
        "ref_id": int(ref["ref_id"]),
        "sent_id": int(row["sent_id"]),
        "split": split,
        "instances": [instance],
    }


_SWAPS: Dict[str, Tuple[str, str]] = {
    "black": ("white", "color"),
    "white": ("black", "color"),
    "red": ("blue", "color"),
    "blue": ("red", "color"),
    "green": ("yellow", "color"),
    "yellow": ("green", "color"),
    "brown": ("gray", "color"),
    "grey": ("brown", "color"),
    "gray": ("brown", "color"),
    "orange": ("purple", "color"),
    "purple": ("orange", "color"),
    "pink": ("green", "color"),
    "large": ("small", "size"),
    "big": ("small", "size"),
    "small": ("large", "size"),
    "little": ("large", "size"),
    "tall": ("short", "size"),
    "short": ("tall", "size"),
    "left": ("right", "spatial"),
    "right": ("left", "spatial"),
    "front": ("back", "spatial"),
    "back": ("front", "spatial"),
    "top": ("bottom", "spatial"),
    "bottom": ("top", "spatial"),
    "upper": ("lower", "spatial"),
    "lower": ("upper", "spatial"),
}


def _make_tn_phrase(phrase: str) -> Optional[Tuple[str, str, str, str]]:
    tokens = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9']+", phrase)
    for i, tok in enumerate(tokens):
        if not re.match(r"^[A-Za-z][A-Za-z']*$", tok):
            continue
        key = tok.lower()
        if key not in _SWAPS:
            continue
        replacement, category = _SWAPS[key]
        if tok[:1].isupper():
            replacement = replacement.capitalize()
        new_tokens = list(tokens)
        new_tokens[i] = replacement
        tn_phrase = "".join(new_tokens)
        if _norm_text(tn_phrase) == _norm_text(phrase):
            continue
        return phrase, tn_phrase, tok, category
    return None


def _write_positive_split(
    *,
    data_root: Path,
    out_path: Path,
    dataset: str,
    splitby: str,
    split: str,
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
) -> int:
    ref_root = data_root / "COCO" / dataset
    refs_path = ref_root / f"refs({splitby}).p"
    instances_path = ref_root / "instances.json"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in _iter_ref_sentences(refs_path, instances_path, split):
            class_rec = _class_record(
                ann=row["ann"],
                category_name=row["category_name"],
                name_to_id=name_to_id,
                id_to_name=id_to_name,
            )
            meta = _meta_from_row(
                data_root=data_root,
                dataset=dataset,
                splitby=splitby,
                split=split,
                row=row,
                class_rec=class_rec,
                phrase=row["phrase"],
                text_is_negative=False,
            )
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_tn_splits(
    *,
    data_root: Path,
    out_path: Path,
    specs: Iterable[Tuple[str, str, str]],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    max_rows: int,
) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for dataset, splitby, split in specs:
            ref_root = data_root / "COCO" / dataset
            refs_path = ref_root / f"refs({splitby}).p"
            instances_path = ref_root / "instances.json"
            for row in _iter_ref_sentences(refs_path, instances_path, split):
                tn = _make_tn_phrase(row["phrase"])
                if tn is None:
                    continue
                positive_phrase, tn_phrase, replaced_token, category = tn
                class_rec = _class_record(
                    ann=row["ann"],
                    category_name=row["category_name"],
                    name_to_id=name_to_id,
                    id_to_name=id_to_name,
                )
                meta = _meta_from_row(
                    data_root=data_root,
                    dataset=dataset,
                    splitby=splitby,
                    split=split,
                    row=row,
                    class_rec=class_rec,
                    phrase=tn_phrase,
                    text_is_negative=True,
                    positive_phrase=positive_phrase,
                    replace_from=positive_phrase,
                    replace_to=tn_phrase,
                    replace_category=category,
                )
                replacement_token = _SWAPS[replaced_token.lower()][0]
                if replaced_token[:1].isupper():
                    replacement_token = replacement_token.capitalize()
                meta["instances"][0]["replace_from"] = replaced_token
                meta["instances"][0]["replace_to"] = replacement_token
                meta["instances"][0]["replace_token"] = replaced_token
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                count += 1
                if max_rows > 0 and count >= max_rows:
                    return count
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/user/datasets/pivot_data")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-tn-rows", type=int, default=60000)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "patch_episode_prebuilt"
    out_dir.mkdir(parents=True, exist_ok=True)
    name_to_id, id_to_name = _load_canonical_name_maps(data_root / "canonical_classes_with_aliases.json")

    plus_out = out_dir / "refcocoplus_stageb_phrase_v1.jsonl"
    gog_out = out_dir / "refcocog_stageb_phrase_v1.jsonl"
    coco_out = out_dir / "refcoco_stageb_phrase_v1.jsonl"
    tn_out = out_dir / "refexp_tn_stageb_v1.jsonl"

    n_coco = _write_positive_split(
        data_root=data_root,
        out_path=coco_out,
        dataset="refcoco",
        splitby="unc",
        split=str(args.split),
        name_to_id=name_to_id,
        id_to_name=id_to_name,
    )
    n_plus = _write_positive_split(
        data_root=data_root,
        out_path=plus_out,
        dataset="refcoco+",
        splitby="unc",
        split=str(args.split),
        name_to_id=name_to_id,
        id_to_name=id_to_name,
    )
    n_gog = _write_positive_split(
        data_root=data_root,
        out_path=gog_out,
        dataset="refcocog",
        splitby="umd",
        split=str(args.split),
        name_to_id=name_to_id,
        id_to_name=id_to_name,
    )
    n_tn = _write_tn_splits(
        data_root=data_root,
        out_path=tn_out,
        specs=[
            ("refcoco", "unc", str(args.split)),
            ("refcoco+", "unc", str(args.split)),
            ("refcocog", "umd", str(args.split)),
        ],
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        max_rows=int(args.max_tn_rows),
    )

    print(f"wrote {n_coco} rows -> {coco_out}")
    print(f"wrote {n_plus} rows -> {plus_out}")
    print(f"wrote {n_gog} rows -> {gog_out}")
    print(f"wrote {n_tn} rows -> {tn_out}")


if __name__ == "__main__":
    main()
