# python tools/offline_label_vg_regions.py \
#   --in_json /media/haoyi/T9/data/vaw_dataset/data/region_descriptions.json \
#   --out_jsonl /media/haoyi/T9/data/vaw_dataset/data/region_descriptions_labeled.jsonl \
#   --ckpt /media/haoyi/T9/Open-GroundingDino/exp_vg_multiclass_clean/best.pt \
#   --device cuda:0 --batch_size 4096 --min_conf 0.05 --write_conf

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ensure_repo_on_path() -> None:
    # Avoid conflicts with the external `datasets` (HuggingFace) package by forcing
    # this repo root to the front of sys.path.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _iter_region_descriptions_items(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Iterate items from a VG/VAW `region_descriptions.json`.

    Tries to use `ijson` for streaming if available; otherwise falls back to `json.load`.
    """
    try:
        import ijson  # type: ignore

        with path.open("rb") as f:
            for item in ijson.items(f, "item"):
                if isinstance(item, dict):
                    yield item
        return
    except Exception:
        pass

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON root list, got {type(data)}")
    for item in data:
        if isinstance(item, dict):
            yield item


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> List[float]:
    return [float(x), float(y), float(x + w), float(y + h)]


def _batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline label VG/VAW phrases to canonical class_id.")
    parser.add_argument(
        "--in_json",
        required=True,
        help="Path to region_descriptions.json (list of {id, regions:[{phrase,x,y,width,height},...]})",
    )
    parser.add_argument(
        "--out_jsonl",
        required=True,
        help="Output JSONL path. Each line is {filename, instances:[{bbox, class_id, conf?}, ...]}",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Phrase classifier checkpoint (e.g. exp_vg_multiclass_clean/best.pt)",
    )
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    parser.add_argument("--max_length", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--min_conf", type=float, default=0.2)
    parser.add_argument("--write_conf", action="store_true", help="Write classifier confidence to output instances.")
    parser.add_argument("--max_images", type=int, default=0, help="For debugging: limit number of images (0=all).")
    parser.add_argument(
        "--flush_every",
        type=int,
        default=50,
        help="Flush output file every N images (0=never).",
    )
    args = parser.parse_args()

    _ensure_repo_on_path()

    in_path = Path(args.in_json)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse the exact classifier implementation used by the training dataset.
    try:
        from datasets.patch_episode import _PhraseClassifierLabeler  # pylint: disable=import-error
    except Exception as e:
        print(f"[ERR] Failed to import phrase classifier helper from datasets.patch_episode: {e}", file=sys.stderr)
        return 2

    clf = _PhraseClassifierLabeler(
        ckpt_path=args.ckpt,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        min_conf=args.min_conf,
    )

    phrase_cache: Dict[str, Tuple[Optional[int], float]] = {}

    total_imgs = 0
    total_regions = 0
    kept_regions = 0
    skipped_empty_phrase = 0
    skipped_low_conf = 0

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out_f:
        for item in _iter_region_descriptions_items(in_path):
            img_id = item.get("id", None)
            if img_id is None:
                continue
            try:
                img_id_int = int(img_id)
            except Exception:
                continue

            regions = item.get("regions", []) or []
            if not isinstance(regions, list) or not regions:
                continue

            phrases: List[str] = []
            meta: List[Tuple[int, List[float], str]] = []
            for r_i, r in enumerate(regions):
                if not isinstance(r, dict):
                    continue
                phrase = r.get("phrase", None)
                if not isinstance(phrase, str) or not phrase.strip():
                    skipped_empty_phrase += 1
                    continue
                try:
                    x = float(r["x"])
                    y = float(r["y"])
                    w = float(r["width"])
                    h = float(r["height"])
                except Exception:
                    continue
                bbox = _xywh_to_xyxy(x, y, w, h)
                phrases.append(phrase)
                meta.append((r_i, bbox, phrase))

            if not phrases:
                continue

            # Predict with caching.
            unique_to_predict: List[str] = []
            for p in phrases:
                if p not in phrase_cache:
                    unique_to_predict.append(p)
            if unique_to_predict:
                for chunk in _batched(unique_to_predict, batch_size=max(1, int(args.batch_size))):
                    preds = clf.predict_top1(chunk)
                    for p, (cid, conf) in zip(chunk, preds):
                        phrase_cache[p] = (cid, conf)

            instances: List[Dict[str, Any]] = []
            for _idx, bbox, phrase in meta:
                cid, conf = phrase_cache.get(phrase, (None, 0.0))
                total_regions += 1
                if cid is None:
                    skipped_low_conf += 1
                    continue
                inst: Dict[str, Any] = {"bbox": bbox, "class_id": int(cid)}
                if args.write_conf:
                    inst["conf"] = float(conf)
                instances.append(inst)
                kept_regions += 1

            if not instances:
                continue

            rec = {
                # Keep filename relative; PatchEpisodeJsonlDataset can resolve via root + vg_image_roots.
                "filename": f"{img_id_int}.jpg",
                "instances": instances,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_imgs += 1

            if args.max_images and total_imgs >= int(args.max_images):
                break
            if total_imgs % 100 == 0:
                dt = time.time() - t0
                print(
                    f"[INFO] images={total_imgs} regions_total={total_regions} kept={kept_regions} "
                    f"skip_empty={skipped_empty_phrase} skip_lowconf={skipped_low_conf} "
                    f"cache={len(phrase_cache)} elapsed={dt:.1f}s"
                )
            if args.flush_every and int(args.flush_every) > 0 and total_imgs % int(args.flush_every) == 0:
                out_f.flush()

    dt = time.time() - t0
    print(
        f"[DONE] wrote={out_path} images={total_imgs} regions_total={total_regions} kept={kept_regions} "
        f"skip_empty={skipped_empty_phrase} skip_lowconf={skipped_low_conf} cache={len(phrase_cache)} "
        f"elapsed={dt/60:.1f}min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
