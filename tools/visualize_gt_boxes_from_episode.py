import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_canonical_id_to_name(path: Optional[str]) -> Dict[int, str]:
    if not path:
        return {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}
    out: Dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        cid = item.get("id", None)
        if cid is None:
            continue
        try:
            cid_i = int(cid)
        except Exception:
            continue
        name = item.get("raw_name", None) or item.get("norm_name", None) or str(cid_i)
        out[cid_i] = str(name)
    return out


def _color_for_id(cid: int) -> Tuple[int, int, int]:
    # Deterministic pseudo-random color by class id.
    r = (cid * 37 + 23) % 255
    g = (cid * 91 + 101) % 255
    b = (cid * 53 + 7) % 255
    return int(r), int(g), int(b)


def _clamp_xyxy(box: List[float], w: int, h: int) -> List[int]:
    x0, y0, x1, y1 = box
    x0 = int(max(0, min(w - 1, round(x0))))
    y0 = int(max(0, min(h - 1, round(y0))))
    x1 = int(max(x0 + 1, min(w, round(x1))))
    y1 = int(max(y0 + 1, min(h, round(y1))))
    return [x0, y0, x1, y1]


def _pick_datasetinfo(args) -> Dict[str, Any]:
    if args.datasets_json:
        with Path(args.datasets_json).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if args.split not in meta:
            raise KeyError(f"datasets_json missing split={args.split}")
        entries = meta[args.split]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"datasets_json[{args.split}] must be a non-empty list")

        if int(args.dataset_index) >= 0:
            datasetinfo = entries[int(args.dataset_index)]
        else:
            datasetinfo = None
            if args.source is not None:
                for e in entries:
                    if e.get("source", None) == args.source:
                        datasetinfo = e
                        break
            datasetinfo = datasetinfo or entries[0]
    else:
        if not args.anno or not args.root or not args.canonical_classes_json:
            raise ValueError("Without --datasets_json, you must provide --anno --root --canonical_classes_json.")
        datasetinfo = {
            "dataset_mode": "patch_episode",
            "root": args.root,
            "anno": args.anno,
            "canonical_classes_json": args.canonical_classes_json,
            "box_format": "xyxy",
            "vg_image_roots": [p.strip() for p in (args.vg_image_roots or "").split(",") if p.strip()] or None,
        }

    if datasetinfo.get("dataset_mode", None) != "patch_episode":
        raise ValueError(f"Expected dataset_mode=patch_episode, got {datasetinfo.get('dataset_mode')}")

    if args.anno_override:
        datasetinfo = dict(datasetinfo)
        datasetinfo["anno"] = args.anno_override
    if args.root_override:
        datasetinfo = dict(datasetinfo)
        datasetinfo["root"] = args.root_override
    if args.vg_image_roots_override:
        datasetinfo = dict(datasetinfo)
        datasetinfo["vg_image_roots"] = [p.strip() for p in args.vg_image_roots_override.split(",") if p.strip()]
    if args.canonical_classes_json:
        datasetinfo = dict(datasetinfo)
        datasetinfo["canonical_classes_json"] = args.canonical_classes_json
    return datasetinfo


def _is_flat_region_jsonl(anno_path: str) -> bool:
    p = Path(anno_path)
    if p.suffix.lower() != ".jsonl" or (not p.exists()):
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    return False
                # Expected keys for your new labeled VG jsonl: one region per line.
                return ("image_id" in obj) and ("x" in obj) and ("width" in obj) and ("label" in obj) and ("instances" not in obj)
    except Exception:
        return False
    return False


def _sample_image_ids_from_flat_jsonl(anno_path: str, k: int, seed: int) -> List[int]:
    """
    Reservoir-sample unique image_ids from a flat jsonl (one region per line).
    Two-pass approach will later collect all regions for these ids.
    """
    rnd = random.Random(int(seed))
    seen: set[int] = set()
    reservoir: List[int] = []
    t = 0
    with Path(anno_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Fast path: pull image_id via JSON because file is already jsonl.
                obj = json.loads(line)
                img_id = int(obj.get("image_id"))
            except Exception:
                continue
            if img_id in seen:
                continue
            seen.add(img_id)
            t += 1
            if len(reservoir) < k:
                reservoir.append(img_id)
            else:
                j = rnd.randrange(t)
                if j < k:
                    reservoir[j] = img_id
    reservoir.sort()
    return reservoir


def _collect_metas_from_flat_jsonl(anno_path: str, image_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Convert a flat region jsonl into per-image metas compatible with our visualization loop:
      {"filename": "123.jpg", "instances": [{"bbox":[x0,y0,x1,y1], "class_id": int}, ...]}
    """
    want = set(int(x) for x in image_ids)
    by_id: Dict[int, List[Dict[str, Any]]] = {int(i): [] for i in want}
    with Path(anno_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            try:
                img_id = int(obj.get("image_id"))
            except Exception:
                continue
            if img_id not in want:
                continue
            try:
                x = float(obj["x"])
                y = float(obj["y"])
                w = float(obj["width"])
                h = float(obj["height"])
            except Exception:
                continue
            cls = obj.get("class_id", obj.get("label", None))
            if cls is None:
                continue
            try:
                cid = int(cls)
            except Exception:
                continue
            by_id[img_id].append({"bbox": [x, y, x + w, y + h], "class_id": cid})

    metas: List[Dict[str, Any]] = []
    for img_id in image_ids:
        inst = by_id.get(int(img_id), [])
        if not inst:
            continue
        metas.append({"filename": f"{int(img_id)}.jpg", "instances": inst})
    return metas


def _open_image_from_roots(root: str, rel_path: str, alt_roots: Optional[List[str]] = None):
    from PIL import Image

    rel_p = Path(rel_path)
    abs_path = rel_p if rel_p.is_absolute() else (Path(root) / rel_p)
    if not abs_path.exists() and (not rel_p.is_absolute()) and alt_roots:
        for r in alt_roots:
            cand = Path(r) / rel_p
            if cand.exists():
                abs_path = cand
                break
    if not abs_path.exists():
        raise FileNotFoundError(str(abs_path))
    return Image.open(abs_path).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Randomly sample images and visualize GT boxes + canonical labels using the same parsing logic as training."
    )
    parser.add_argument("--datasets_json", default=None, help="e.g. config/datasets_patch_stage_a_raw_local.json")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--source", default=None, help="Pick a dataset entry by datasetinfo['source'] when using --datasets_json.")
    parser.add_argument("--dataset_index", type=int, default=-1, help="Pick dataset entry by index (overrides --source).")
    parser.add_argument("--anno_override", type=str, default=None, help="Override datasetinfo['anno'].")
    parser.add_argument("--root_override", type=str, default=None, help="Override datasetinfo['root'].")
    parser.add_argument(
        "--vg_image_roots_override",
        type=str,
        default=None,
        help="Override datasetinfo['vg_image_roots'] as comma-separated list.",
    )

    # Direct mode (without datasets_json)
    parser.add_argument("--anno", type=str, default=None, help="Annotation file (.jsonl/.json).")
    parser.add_argument("--root", type=str, default=None, help="Image root for relative filenames.")
    parser.add_argument(
        "--vg_image_roots", type=str, default=None, help="Comma-separated alt roots for VG images."
    )
    parser.add_argument(
        "--canonical_classes_json",
        type=str,
        default=None,
        help="Path to canonical_classes_with_aliases.json (used for label names).",
    )

    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_boxes", type=int, default=200, help="Limit number of drawn boxes per image (0=all).")
    parser.add_argument("--min_boxes", type=int, default=1, help="Reject images with < min_boxes instances.")
    parser.add_argument("--max_tries", type=int, default=5000)
    parser.add_argument("--font_size", type=int, default=14)
    parser.add_argument(
        "--use_phrase_classifier",
        action="store_true",
        help="For raw VG region_descriptions.json: enable classifier labeling (slow). Not needed for labeled jsonl.",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    datasetinfo = _pick_datasetinfo(args)

    from datasets.patch_episode import PatchEpisodeJsonlDataset

    canonical_classes_json = datasetinfo.get("canonical_classes_json", None)
    id2name = _load_canonical_id_to_name(canonical_classes_json)

    source = datasetinfo.get("source", None)
    vg_phrase_labeler = datasetinfo.get("vg_phrase_labeler", "prefix")
    if (source == "vg_region_descriptions") and (not args.use_phrase_classifier):
        vg_phrase_labeler = "prefix"

    anno_path = str(datasetinfo["anno"])
    is_flat = _is_flat_region_jsonl(anno_path)

    ds = None
    metas: List[Dict[str, Any]]
    if is_flat:
        # Your new file is one-region-per-line; group to per-image metas first.
        chosen = _sample_image_ids_from_flat_jsonl(anno_path, k=max(1, int(args.num_images) * 2), seed=int(args.seed))
        metas = _collect_metas_from_flat_jsonl(anno_path, chosen)
    else:
        ds = PatchEpisodeJsonlDataset(
            root=datasetinfo.get("root", "/"),
            anno=anno_path,
            transforms=None,
            box_format=datasetinfo.get("box_format", "xyxy"),
            canonical_classes_json=canonical_classes_json,
            source=source,
            lvis_image_root=datasetinfo.get("lvis_image_root", None),
            vg_image_roots=datasetinfo.get("vg_image_roots", None),
            vg_phrase_labeler=vg_phrase_labeler,
            phrase_classifier_ckpt=datasetinfo.get("phrase_classifier_ckpt", None),
            phrase_classifier_device=datasetinfo.get("phrase_classifier_device", "cpu"),
            phrase_classifier_max_length=int(datasetinfo.get("phrase_classifier_max_length", 24)),
            phrase_classifier_batch_size=int(datasetinfo.get("phrase_classifier_batch_size", 64)),
            phrase_classifier_min_conf=float(datasetinfo.get("phrase_classifier_min_conf", 0.0)),
            phrase_cache_size=int(datasetinfo.get("phrase_cache_size", 50000)),
            # No patch bank needed for GT visualization.
            support_patch_tsv=None,
            support_patch_use_embedding=False,
            patch_bank_cache=False,
            anno_cache=True,
        )
        metas = ds.metas

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=int(args.font_size))
    except Exception:
        pass

    saved = 0
    tries = 0
    used_rel: set[str] = set()
    while saved < int(args.num_images) and tries < int(args.max_tries):
        tries += 1
        idx = random.randrange(0, len(metas))
        meta_i = metas[idx]
        rel = meta_i.get("filename", meta_i.get("file_name", None))
        if rel is None:
            continue
        rel_key = str(rel)
        if rel_key in used_rel:
            continue
        try:
            if ds is not None:
                img = ds._open_image(rel)
            else:
                img = _open_image_from_roots(
                    root=str(datasetinfo.get("root", "/")),
                    rel_path=str(rel),
                    alt_roots=datasetinfo.get("vg_image_roots", None),
                )
        except Exception:
            continue
        w, h = img.size
        if ds is not None:
            boxes_xyxy, labels = ds._extract_instances(meta_i)
        else:
            instances = meta_i.get("instances", []) or []
            boxes = [it["bbox"] for it in instances if "bbox" in it]
            labels_list = [int(it.get("class_id", it.get("label"))) for it in instances if ("class_id" in it or "label" in it)]
            if len(boxes) != len(labels_list) or not boxes:
                continue
            import torch

            boxes_xyxy = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels = torch.as_tensor(labels_list, dtype=torch.int64).reshape(-1)
        if labels.numel() < int(args.min_boxes):
            continue

        n = int(labels.numel())
        keep_idx = list(range(n))
        max_boxes = int(args.max_boxes)
        if max_boxes > 0 and n > max_boxes:
            keep_idx = random.sample(keep_idx, k=max_boxes)
            keep_idx.sort()
            boxes_xyxy = boxes_xyxy[keep_idx]
            labels = labels[keep_idx]

        draw = ImageDraw.Draw(img)
        counts = Counter(labels.tolist())
        top_classes = counts.most_common(6)
        header = " | ".join([f"{id2name.get(c, c)}:{k}" for c, k in top_classes])
        draw.text((5, 5), header, fill=(255, 255, 0), font=font)

        for box, cid in zip(boxes_xyxy.tolist(), labels.tolist()):
            x0, y0, x1, y1 = _clamp_xyxy(box, w=w, h=h)
            color = _color_for_id(int(cid))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            name = id2name.get(int(cid), str(int(cid)))
            txt = f"{name}({int(cid)})"
            draw.text((x0 + 2, max(0, y0 - int(args.font_size))), txt, fill=color, font=font)

        stem = Path(str(rel)).stem
        out_path = out_dir / f"{saved:03d}_idx{idx}_{stem}.jpg"
        img.save(out_path, quality=90)

        sidecar = {
            "dataset_index": idx,
            "filename": str(rel),
            "num_boxes": int(labels.numel()),
            "box_format": "xyxy_abs",
            "labels": [int(x) for x in labels.tolist()],
            "boxes": [[float(v) for v in b] for b in boxes_xyxy.tolist()],
        }
        (out_dir / f"{saved:03d}_idx{idx}_{stem}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        used_rel.add(rel_key)
        saved += 1

    print(
        f"[DONE] saved={saved} tries={tries} out_dir={out_dir} source={datasetinfo.get('source')} "
        f"anno={datasetinfo.get('anno')} root={datasetinfo.get('root')} flat_region_jsonl={is_flat}"
    )
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
