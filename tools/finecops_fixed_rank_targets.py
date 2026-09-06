"""Matched existence/emission targets with an immutable native localization route."""
import copy
import torch
import torch.nn.functional as F
from tools.b32a1_heads import B32A1AbstentionHead, ISOLATED

TARGETS = ("exists", "emit")
LOGIT_L2 = 1e-3


def make_heads(seed, device):
    torch.manual_seed(seed)
    model = B32A1AbstentionHead(mode=ISOLATED).to(device)
    for p in model.task_parameters("rank"):
        p.requires_grad_(False)
    return {"exists": model, "emit": copy.deepcopy(model)}


def target_loss(positive, negative, correct, target):
    if target not in TARGETS or positive.ndim != 1 or negative.ndim != 1:
        raise ValueError("invalid target or logits")
    if correct.dtype != torch.bool or correct.shape != positive.shape:
        raise ValueError("correctness must be aligned bool")
    if not positive.numel() or not negative.numel():
        raise ValueError("empty source")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("nonfinite logits")
    label = torch.ones_like(positive) if target == "exists" else correct.to(positive.dtype)
    # Source weights stay fixed: no target-dependent class rebalancing.
    loss = .5 * (F.binary_cross_entropy_with_logits(positive, label) + F.softplus(negative).mean())
    return loss + .5 * LOGIT_L2 * (positive.square().mean() + negative.square().mean())


def maxima_and_parity(model, features, native, mask):
    out = model(features, native, mask)
    expected = native.masked_fill(~mask, torch.finfo(native.dtype).min)
    if not torch.equal(out["rank_score"], expected) or torch.count_nonzero(out["rank_residual"]):
        raise ValueError("native ranking parity failed")
    return out["confidence_score"].masked_fill(~mask, -torch.inf).max(1).values
