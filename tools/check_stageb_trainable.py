#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from util.get_param_dicts import match_name_keywords  # noqa: E402
from util.slconfig import DictAction, SLConfig  # noqa: E402


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _as_keyword_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def apply_stageb_trainability(model: torch.nn.Module, cfg) -> None:
    freeze_keywords = _as_keyword_list(getattr(cfg, "freeze_keywords", None))
    if freeze_keywords:
        for name, parameter in model.named_parameters():
            if match_name_keywords(name, freeze_keywords):
                parameter.requires_grad_(False)

    unfreeze_n = int(getattr(cfg, "unfreeze_decoder_last_n_layers", 0) or 0)
    if unfreeze_n <= 0 and bool(getattr(cfg, "unfreeze_decoder_last_layer", False)):
        unfreeze_n = 1
    if unfreeze_n > 0:
        decoder = model.transformer.decoder
        layers = list(getattr(decoder, "layers", []))
        n = min(unfreeze_n, len(layers))
        for layer in layers[-n:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)

    only_train_keywords = _as_keyword_list(getattr(cfg, "only_train_keywords", None))
    if only_train_keywords:
        only_train_exclude_keywords = _as_keyword_list(getattr(cfg, "only_train_exclude_keywords", None))
        for _, parameter in model.named_parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if match_name_keywords(name, only_train_keywords) and not match_name_keywords(name, only_train_exclude_keywords):
                parameter.requires_grad_(True)


def trainable_summary(model: torch.nn.Module) -> Tuple[int, int, Dict[str, int], Dict[str, int]]:
    total = 0
    trainable = 0
    trainable_names: Dict[str, int] = {}
    grouped: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names[name] = count
            root = name.split(".", 1)[0]
            grouped[root] = grouped.get(root, 0) + count
    return total, trainable, trainable_names, grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict_text_head_only", action="store_true")
    parser.add_argument("--options", nargs="+", action=DictAction)
    args = parser.parse_args()

    cfg = SLConfig.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg.device = args.device

    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    model, _criterion, _postprocessors = build_func(cfg)

    ckpt = _torch_load_compat(args.checkpoint, map_location="cpu")
    state = clean_state_dict(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}", file=sys.stderr)

    apply_stageb_trainability(model, cfg)
    total, trainable, trainable_names, grouped = trainable_summary(model)

    print(f"total params: {total}")
    print(f"trainable params: {trainable}")
    print("trainable modules:")
    print(json.dumps(grouped, indent=2, sort_keys=True))
    print("trainable parameter names:")
    print(json.dumps(trainable_names, indent=2, sort_keys=True))

    if args.strict_text_head_only:
        allowed = ("feat_map", "class_embed")
        bad = [name for name in trainable_names if not any(key in name for key in allowed)]
        if bad:
            print("[ERR] Unexpected trainable parameters outside feat_map/class_embed:", file=sys.stderr)
            print(json.dumps(bad, indent=2), file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
