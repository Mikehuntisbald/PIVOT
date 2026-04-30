# python3 -m models.GroundingDINO.encode_patch_quality_embeddings \
#   --quality_tsv /media/haoyi/T9/data/patches_quality/patch_quality_log.tsv \
#   --out_root /media/haoyi/T9/data/patches_quality_emb \
#   --gdino_cfg tools/GroundingDINO_SwinT_OGC.py \
#   --gdino_ckpt weights/groundingdino_swint_ogc.pth \
#   --batch_size 512 --num_workers 8 --write_workers 8 \ 
#   --amp --save_fp16 --skip_existing

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 禁用代理（避免 httpx/openai 之类意外读到 socks 代理）
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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_gdino_backbone(cfg_path: str, ckpt_path: str, device: str):
    """
    从 GroundingDINO 的 cfg+ckpt 构建 model，并返回 (backbone, hidden_dim)
    """
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


def make_out_rel(row: Dict[str, Any], out_suffix: str) -> Path:
    """
    输出结构：
      {bucket}/{source_root}/{class}/{filename}{out_suffix}

    source_root 从原 path 推断：
      .../vg_patches/<class>/xxx.jpg  -> source_root = vg_patches
      .../lvis_patches/<class>/yyy.jpg-> source_root = lvis_patches

    注意：这里完全不触发 stat，只做路径字符串解析。
    """
    p = Path(row["path"])
    bucket = row.get("bucket", "") or "unknown_bucket"
    cls = row.get("class", "") or "unknown_class"

    # 期望 p 的父目录是 class，父父目录是 vg_patches / lvis_patches
    # e.g. .../vg_patches/telephone/a.jpg
    source_root = p.parent.parent.name if len(p.parents) >= 2 else "unknown_source"

    return Path(bucket) / source_root / cls / (p.stem + out_suffix)


class PatchFromTSVDataset(Dataset):
    """
    从 TSV rows 读取 patch 图片并做 transform。
    这里做容错：打不开/损坏的图返回 None，后续 collate 过滤。
    """
    def __init__(self, rows: List[Dict[str, Any]], tfm: T.Compose):
        self.rows = rows
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img_path = Path(r["path"])
        try:
            img = Image.open(img_path).convert("RGB")
            img_t = self.tfm(img)  # (3,224,224)
            return r, img_t
        except Exception:
            return None


def collate_fn(batch):
    """
    过滤掉 Dataset 返回的 None。
    如果一个 batch 全坏，返回 (None, None)，主循环跳过。
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
    rows, imgs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return list(rows), imgs


def save_one_embedding_npy(emb_cpu: torch.Tensor, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), emb_cpu.numpy())


def save_one_embedding_pt(emb_cpu: torch.Tensor, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_cpu, str(out_path))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--quality_tsv", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--gdino_cfg", type=str, required=True)
    ap.add_argument("--gdino_ckpt", type=str, required=True)

    ap.add_argument("--bucket", type=str, default="", help="只跑 clean/borderline/bad；空表示全跑")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--out_fmt", type=str, default="npy", choices=["npy", "pt"])
    ap.add_argument("--save_fp16", action="store_true")

    ap.add_argument("--write_workers", type=int, default=8)
    ap.add_argument("--max_pending_writes", type=int, default=5000)
    ap.add_argument("--skip_existing", action="store_true")

    # ✅ 默认不做 exists/is_file 预检查（避免你卡死的 stat 风暴）
    # 只有你显式加 --verify_paths 才会逐条 stat。
    ap.add_argument("--verify_paths", action="store_true",
                    help="逐条检查 path.exists()/is_file()（非常慢，默认关闭）")

    ap.add_argument("--emb_index_tsv", type=str, default="emb_index_from_quality.tsv")
    args = ap.parse_args()

    tsv_path = Path(args.quality_tsv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # 1) 读 TSV 当输入清单
    rows = read_quality_tsv(tsv_path)
    if args.bucket:
        rows = [r for r in rows if (r.get("bucket", "") == args.bucket)]

    # 2) 过滤（默认：只按扩展名，不触发 stat）
    if args.verify_paths:
        ok_rows = []
        miss = 0
        pbar = tqdm(rows, desc="verify paths (SLOW)", unit="row",
                    file=sys.stdout, dynamic_ncols=True)
        for r in pbar:
            p = Path(r["path"])
            # 这一步会触发 os.stat，非常慢（你刚刚卡的就是这里）
            if p.exists() and p.is_file() and p.suffix.lower() in IMG_EXTS:
                ok_rows.append(r)
            else:
                miss += 1
        rows = ok_rows
        print(f"[INFO] verify_paths=1, rows={len(rows)}, skipped_missing={miss}")
    else:
        before = len(rows)
        rows = [r for r in rows if Path(r["path"]).suffix.lower() in IMG_EXTS]
        print(f"[INFO] verify_paths=0, rows={len(rows)} (ext-only), before={before}")

    if len(rows) == 0:
        print("[WARN] rows=0, nothing to do.")
        return

    # 3) backbone + patch encoder
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

    out_suffix = ".npy" if args.out_fmt == "npy" else ".pt"
    save_func = save_one_embedding_npy if args.out_fmt == "npy" else save_one_embedding_pt

    ds = PatchFromTSVDataset(rows, tfm)
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

    # 5) 异步写盘线程池
    writer = ThreadPoolExecutor(max_workers=args.write_workers)
    pending = []

    def flush_pending(force=False):
        """
        控制 pending futures 数量，避免内存/队列爆炸。
        """
        nonlocal pending
        if not pending:
            return
        if (not force) and (len(pending) < args.max_pending_writes):
            return

        # 等待一部分完成（避免全等）
        target = max(1, len(pending) // 2)
        done = []
        for fut in as_completed(pending[:target]):
            fut.result()
            done.append(fut)
        pending = [f for f in pending if f not in done]

    # 6) 索引缓存
    emb_index_rows = []
    emb_index_path = out_root / args.emb_index_tsv

    # 7) 主进度条：编码+提交写盘
    pbar = tqdm(total=len(rows), desc="encode", unit="img",
                file=sys.stdout, dynamic_ncols=True)

    with torch.inference_mode():
        for batch_rows, imgs in dl:
            if imgs is None:
                # 这个 batch 全坏，直接跳过
                continue

            if device != "cpu":
                imgs = imgs.to(device, non_blocking=True)

            if args.amp and device != "cpu":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = patch_encoder(imgs, text_dict=None)
            else:
                out = patch_encoder(imgs, text_dict=None)

            emb = out.get("patch_global", None)  # (B,D)
            if emb is None:
                pbar.update(len(batch_rows))
                continue

            if args.save_fp16:
                emb_cpu = emb.detach().to(torch.float16).cpu()
                dtype_str = "float16"
            else:
                emb_cpu = emb.detach().to(torch.float32).cpu()
                dtype_str = "float32"

            B, D = emb_cpu.shape

            # 先更新进度条（这样 IO 卡住也能看到在推进）
            pbar.update(B)
            pbar.set_postfix(pending_writes=len(pending))

            for i in range(B):
                r = batch_rows[i]
                out_rel = make_out_rel(r, out_suffix)
                emb_path = out_root / out_rel

                emb_index_rows.append([
                    r["path"],
                    r.get("class", ""),
                    r.get("occlusion", ""),
                    r.get("blur", ""),
                    r.get("class_confidence", ""),
                    r.get("bucket", ""),
                    str(out_rel),
                    str(D),
                    dtype_str,
                ])

                if args.skip_existing and emb_path.exists():
                    continue

                fut = writer.submit(save_func, emb_cpu[i], emb_path)
                pending.append(fut)

            flush_pending(force=False)

    pbar.close()

    # 8) 等待剩余写盘（第二个进度条）
    if pending:
        wait_bar = tqdm(total=len(pending), desc="wait writes", unit="file",
                        file=sys.stdout, dynamic_ncols=True)
        for fut in as_completed(pending):
            fut.result()
            wait_bar.update(1)
        wait_bar.close()

    writer.shutdown(wait=True)

    # 9) 写 embedding index
    with open(emb_index_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "class", "occlusion", "blur", "class_confidence", "bucket",
                    "emb_rel_path", "dim", "dtype"])
        w.writerows(emb_index_rows)

    print(f"[DONE] embeddings: {out_root}")
    print(f"[DONE] emb index:  {emb_index_path}")


if __name__ == "__main__":
    main()
