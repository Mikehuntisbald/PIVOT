# python3 -m models.GroundingDINO.encode_patch_quality_embeddings_lmdb_basekey \
#   --quality_tsv /media/haoyi/T9/data/patches_quality/patch_quality_log.tsv \
#   --lmdb_path   /media/haoyi/T9/data/patch_emb.lmdb \
#   --map_size_gb 200 \
#   --base_root   /media/haoyi/T9/data \
#   --gdino_cfg  tools/GroundingDINO_SwinT_OGC.py \
#   --gdino_ckpt weights/groundingdino_swint_ogc.pth \
#   --save_fp16 --amp --fast_lmdb --trust_tsv_paths \
#   --commit_every 200000 #--skip_existing

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 禁用代理（避免环境变量污染）
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as T

from .patch_encoder import PatchEncoder

try:
    import lmdb
except ImportError:
    raise RuntimeError("缺少 lmdb：请先 pip install lmdb")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_gdino_backbone(cfg_path: str, ckpt_path: str, device: str):
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict

    args = SLConfig.fromfile(cfg_path)
    model = build_model(args)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(clean_state_dict(state), strict=False)

    model.eval().to(device)

    hidden_dim = 256
    if hasattr(model, "transformer") and hasattr(model.transformer, "d_model"):
        hidden_dim = int(model.transformer.d_model)

    return model.backbone, hidden_dim


def read_quality_tsv(tsv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def open_lmdb(lmdb_path: Path, map_size: int, readonly: bool, fast: bool):
    """
    fast=True:
      - sync=False, metasync=False, map_async=True
      - 速度更快，commit 卡顿显著减少
      - 风险：机器硬断电/崩溃时，可能丢最后一小段（你可以接受就开）
    """
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        map_size=map_size,
        subdir=False if lmdb_path.suffix == ".lmdb" else True,
        readonly=readonly,
        lock=not readonly,
        readahead=readonly,
        meminit=False,
        max_dbs=4,
    )
    if (not readonly) and fast:
        kwargs.update(dict(
            sync=False,
            metasync=False,
            map_async=True,
        ))
    env = lmdb.open(str(lmdb_path), **kwargs)
    return env


def make_rel_key_from_base(abs_path: str, base_root: Path) -> str:
    """
    key = abs_path 相对 base_root 的路径（posix 格式）
    e.g.
      base_root=/media/haoyi/T9/data
      abs_path=/media/haoyi/T9/data/patches_quality/clean/xx.jpg
      -> patches_quality/clean/xx.jpg
    """
    p = Path(abs_path).expanduser()
    try:
        p_abs = p.absolute()
    except Exception:
        p_abs = p

    try:
        rel = p_abs.relative_to(base_root)
        return rel.as_posix()
    except Exception:
        return f"ABS/{p_abs.as_posix()}"


class PatchFromTSVDataset(Dataset):
    """
    返回 (row_dict, img_tensor, key_str)。
    读图失败返回 None，由 collate 过滤。
    """
    def __init__(self, rows: List[Dict[str, Any]], tfm: T.Compose, base_root: Path):
        self.rows = rows
        self.tfm = tfm
        self.base_root = base_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img_path = Path(r["path"])
        try:
            img = Image.open(img_path).convert("RGB")
            img_t = self.tfm(img)
            key_str = make_rel_key_from_base(r["path"], self.base_root)
            return r, img_t, key_str
        except Exception:
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None
    rows, imgs, keys = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return list(rows), imgs, list(keys)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--quality_tsv", type=str, required=True)
    ap.add_argument("--lmdb_path", type=str, required=True)
    ap.add_argument("--map_size_gb", type=int, default=200)

    ap.add_argument("--base_root", type=str, required=True,
                    help="key 使用相对该目录的路径，比如 /media/haoyi/T9/data")

    ap.add_argument("--gdino_cfg", type=str, required=True)
    ap.add_argument("--gdino_ckpt", type=str, required=True)

    ap.add_argument("--bucket", type=str, default="", help="只跑 clean/borderline/bad；空表示全跑")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--save_fp16", action="store_true", help="value 保存 float16 bytes（省空间/更快）")
    ap.add_argument("--skip_existing", action="store_true",
                    help="跳过已存在 key。实现方式：putmulti overwrite=False（无 per-item get）")

    ap.add_argument("--commit_every", type=int, default=20000,
                    help="每写入多少条（尝试写入的条数）commit 一次；调大可减少批间卡顿")

    ap.add_argument("--fast_lmdb", action="store_true",
                    help="更快的 LMDB 写入（sync/metasync 关闭 + map_async）。断电可能丢最后一小段。")

    ap.add_argument("--trust_tsv_paths", action="store_true",
                    help="信任 TSV 里 path 都存在：默认只按扩展名过滤，不做 exists/is_file（推荐开）")

    ap.add_argument("--meta_key", type=str, default="__META__")
    args = ap.parse_args()

    tsv_path = Path(args.quality_tsv)
    lmdb_path = Path(args.lmdb_path)
    base_root = Path(args.base_root).expanduser()
    try:
        base_root = base_root.absolute()
    except Exception:
        pass

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # 1) 读 TSV
    rows = read_quality_tsv(tsv_path)
    if args.bucket:
        rows = [r for r in rows if (r.get("bucket", "") == args.bucket)]

    # 2) 轻过滤：只按扩展名（不触发 stat）
    before = len(rows)
    rows = [r for r in rows if Path(r["path"]).suffix.lower() in IMG_EXTS]
    print(f"[INFO] rows={len(rows)} (ext-only), before={before}")

    # 可选慢检查（一般别开）
    if not args.trust_tsv_paths:
        ok = []
        miss = 0
        for r in tqdm(rows, desc="verify paths (SLOW)", unit="row",
                      file=sys.stdout, dynamic_ncols=True):
            p = Path(r["path"])
            if p.exists() and p.is_file():
                ok.append(r)
            else:
                miss += 1
        rows = ok
        print(f"[INFO] verify done. rows={len(rows)}, missing={miss}")

    if len(rows) == 0:
        print("[WARN] rows=0, nothing to do.")
        return

    # 3) 构建 PatchEncoder
    print(f"[INFO] Build GDINO backbone: cfg={args.gdino_cfg}, ckpt={args.gdino_ckpt}")
    backbone, hidden_dim = build_gdino_backbone(args.gdino_cfg, args.gdino_ckpt, device=device)
    patch_encoder = PatchEncoder(backbone=backbone, hidden_dim=hidden_dim, gate_with_text=False).to(device)
    patch_encoder.eval()

    # 4) transform
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    ds = PatchFromTSVDataset(rows, tfm, base_root=base_root)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 5) 打开 LMDB
    map_size = int(args.map_size_gb) * 1024**3
    env = open_lmdb(lmdb_path, map_size=map_size, readonly=False, fast=args.fast_lmdb)
    db_emb = env.open_db(b"emb", create=True)
    db_meta = env.open_db(b"meta", create=True)

    dtype_str = "float16" if args.save_fp16 else "float32"
    meta = {
        "dtype": dtype_str,
        "hidden_dim": int(hidden_dim),
        "key_format": "relative_to_base_root",
        "base_root": str(base_root),
        "gdino_cfg": str(args.gdino_cfg),
        "gdino_ckpt": str(args.gdino_ckpt),
        "fast_lmdb": bool(args.fast_lmdb),
        "note": "LMDB db=emb: key=relpath(base_root), value=embedding raw bytes",
    }
    with env.begin(write=True, db=db_meta) as txn_meta:
        txn_meta.put(args.meta_key.encode("utf-8"),
                     json.dumps(meta, ensure_ascii=False).encode("utf-8"),
                     overwrite=True)

    # 6) 写入：putmulti 批量写 + 分段 commit
    txn = env.begin(write=True, db=db_emb)
    cur = txn.cursor()

    written = 0
    skipped_exist = 0
    bad_batches = 0
    abs_fallback = 0

    since_commit = 0  # 统计“尝试写入”的数量（len(items)），用于决定 commit

    pbar = tqdm(total=len(rows), desc="encode+lmdb(basekey)", unit="img",
                file=sys.stdout, dynamic_ncols=True)

    try:
        with torch.inference_mode():
            for batch_rows, imgs, keys in dl:
                if imgs is None:
                    bad_batches += 1
                    continue

                if device != "cpu":
                    imgs = imgs.to(device, non_blocking=True)

                if args.amp and device != "cpu":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = patch_encoder(imgs, text_dict=None)
                else:
                    out = patch_encoder(imgs, text_dict=None)

                emb = out.get("patch_global", None)
                if emb is None:
                    pbar.update(len(batch_rows))
                    continue

                # 先搬到 CPU（这里可能同步一次 GPU）
                if args.save_fp16:
                    emb_cpu = emb.detach().to(torch.float16).cpu().contiguous()
                else:
                    emb_cpu = emb.detach().to(torch.float32).cpu().contiguous()

                emb_np = emb_cpu.numpy()  # (B,D)
                B, D = emb_np.shape

                items: List[Tuple[bytes, bytes]] = []
                for i in range(B):
                    key_str = keys[i]
                    if key_str.startswith("ABS/"):
                        abs_fallback += 1
                    k = key_str.encode("utf-8")
                    v = emb_np[i].tobytes(order="C")
                    items.append((k, v))

                # ✅ putmulti：overwrite=False 可以“自动跳过已存在 key”，无需 per-item get
                overwrite = not args.skip_existing

                if args.skip_existing:
                    entries_before = txn.stat().get("entries", 0)
                    cur.putmulti(items, dupdata=False, overwrite=False)
                    entries_after = txn.stat().get("entries", 0)

                    added = max(0, entries_after - entries_before)
                    written += added
                    skipped_exist += (len(items) - added)
                else:
                    cur.putmulti(items, dupdata=False, overwrite=True)
                    written += len(items)


                since_commit += len(items)

                # commit
                if since_commit >= args.commit_every:
                    txn.commit()
                    txn = env.begin(write=True, db=db_emb)
                    cur = txn.cursor()
                    since_commit = 0

                pbar.update(B)
                pbar.set_postfix(written=written, skip_exist=skipped_exist,
                                 bad_batches=bad_batches, abs_key=abs_fallback)

        txn.commit()

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user. Committing current transaction...")
        try:
            txn.commit()
        except Exception:
            pass
        raise
    finally:
        pbar.close()
        # fast_lmdb 开着时，sync 一下更稳（不会太慢，次数少）
        try:
            env.sync()
        except Exception:
            pass
        env.close()

    print(f"[DONE] LMDB: {lmdb_path}")
    print(f"[DONE] written={written}, skipped_exist={skipped_exist}, bad_batches={bad_batches}, abs_key={abs_fallback}")
    print(f"[DONE] meta stored at db=meta key={args.meta_key!r}")


if __name__ == "__main__":
    main()
