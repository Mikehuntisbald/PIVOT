"""
Reset a fraction of DETR query embeddings in a training checkpoint, then save a new checkpoint.

This is useful when you want to "refresh" some query slots without restarting training from scratch.
Typically, you reset `transformer.tgt_embed.weight` (num_queries x d_model).

Example:
  python tools/reset_queries_in_checkpoint.py \\
    --in_ckpt outputs/stageA_patch/checkpoint.pth \\
    --out_ckpt outputs/stageA_patch/checkpoint_reset50.pth \\
    --frac 0.5 --seed 0 --strategy random

Then resume training from the new checkpoint:
  python main.py -c config/cfg_patch_stage_a.py --datasets ... --output_dir outputs/stageA_patch --resume outputs/stageA_patch/checkpoint_reset50.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        try:
            from torch import serialization as _serialization  # type: ignore

            _serialization.add_safe_globals([argparse.Namespace])  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            return torch.load(path, map_location=map_location)
        except Exception:
            return torch.load(path, map_location=map_location, weights_only=False)


def _find_key(sd: Dict[str, torch.Tensor], key: str) -> str | None:
    if key in sd:
        return key
    pref = "module." + key
    if pref in sd:
        return pref
    # best-effort suffix match
    for k in sd.keys():
        if k.endswith(key):
            return k
    return None


def _reset_rows_(w: torch.Tensor, idx: torch.Tensor, *, std: float):
    # Match Transformer init: nn.init.normal_(weight) (std defaults to 1.0)
    with torch.no_grad():
        w[idx].normal_(mean=0.0, std=float(std))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--frac", type=float, default=0.5, help="fraction of queries to reset (0..1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", type=str, default="random", choices=["random", "last", "first"])
    ap.add_argument("--std", type=float, default=1.0, help="std for re-init normal_(0,std)")
    ap.add_argument(
        "--reset_refpoints",
        action="store_true",
        help="also reset transformer.refpoint_embed.weight (only relevant for two_stage_type='no')",
    )
    args = ap.parse_args()

    frac = float(args.frac)
    if not (0.0 <= frac <= 1.0):
        raise ValueError("--frac must be in [0,1].")

    ckpt = _torch_load_compat(args.in_ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Input checkpoint must be a training checkpoint dict with key 'model'.")
    sd = ckpt["model"]
    if not isinstance(sd, dict):
        raise ValueError("checkpoint['model'] must be a state_dict dict.")

    key_tgt = _find_key(sd, "transformer.tgt_embed.weight")
    if key_tgt is None:
        raise KeyError("Could not find transformer.tgt_embed.weight in checkpoint model state_dict.")

    w = sd[key_tgt]
    if not torch.is_tensor(w) or w.dim() != 2:
        raise ValueError(f"{key_tgt} must be a 2D tensor, got {type(w)} shape={getattr(w,'shape',None)}")

    n = int(w.shape[0])
    d = int(w.shape[1])
    n_reset = int(round(frac * n))
    n_reset = max(0, min(n, n_reset))
    if n_reset == 0:
        print(f"[INFO] frac={frac} -> n_reset=0; nothing to do. Saving a copy anyway.")
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(args.seed))
        if args.strategy == "random":
            idx = torch.randperm(n, generator=g)[:n_reset]
        elif args.strategy == "last":
            idx = torch.arange(n - n_reset, n)
        else:  # first
            idx = torch.arange(0, n_reset)
        _reset_rows_(w, idx, std=float(args.std))
        sd[key_tgt] = w
        print(f"[OK] reset {n_reset}/{n} rows of {key_tgt} (d_model={d}), strategy={args.strategy}, std={args.std}")

    if bool(args.reset_refpoints):
        key_ref = _find_key(sd, "transformer.refpoint_embed.weight")
        if key_ref is None:
            print("[WARN] --reset_refpoints requested but transformer.refpoint_embed.weight not found; skipping.")
        else:
            wr = sd[key_ref]
            if torch.is_tensor(wr) and wr.dim() == 2 and wr.shape[0] == n:
                # refpoints are in unsigmoid space in this repo; keep them small near 0 by default.
                with torch.no_grad():
                    wr.normal_(mean=0.0, std=0.1)
                sd[key_ref] = wr
                print(f"[OK] reset {key_ref} with normal_(0,0.1)")
            else:
                print(f"[WARN] {key_ref} has unexpected shape; skipping.")

    ckpt["model"] = sd
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out_path))
    print(f"[DONE] wrote {out_path}")


if __name__ == "__main__":
    main()

