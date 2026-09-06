"""Fixed confidence objective and activity diagnostics for the val-only control."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


LOGIT_L2 = 1e-3
OBJECTIVE = {
    "name": "balanced_global_max_bce_plus_logit_l2",
    "logit_l2": LOGIT_L2,
    "positive_label": 1,
    "negative_label": 0,
    "reduction": "half positive mean plus half negative mean",
    "penalized_logits": "sample global maxima on valid queries only",
    "queue_active": False,
    "target": "referent_exists",
}


def confidence_objective(positive: Tensor, negative: Tensor):
    for value in (positive, negative):
        if value.ndim != 1 or not value.numel() or not bool(torch.isfinite(value).all()):
            raise ValueError("confidence maxima must be finite nonempty vectors")
    bce = 0.5 * (F.softplus(-positive).mean() + F.softplus(negative).mean())
    penalty = 0.5 * LOGIT_L2 * (positive.square().mean() + negative.square().mean())
    loss = bce + penalty
    if not bool(torch.isfinite(loss)):
        raise ValueError("BCE/L2 objective became nonfinite")
    return loss, {"loss": float(loss.detach()), "bce": float(bce.detach()),
                  "logit_l2_penalty": float(penalty.detach()),
                  "positive_mean": float(positive.detach().mean()),
                  "negative_mean": float(negative.detach().mean())}


def select_health_rows(rows, per_level=128, negatives=128):
    def key(row):
        return hashlib.sha256(("bce-l2-health-20260905:" + row["sample_id"]).encode()).hexdigest()
    selected = []
    for level in (1, 2, 3):
        population = [r for r in rows if r["kind"] == "positive" and r["level"] == level]
        selected.extend(sorted(population, key=key)[:per_level])
    selected.extend(sorted((r for r in rows if r["kind"] != "positive"), key=key)[:negatives])
    return selected


def activity_health(module, rows, *, device, rank_loss_fn) -> dict[str, Any]:
    """Measure fixed val activity; never select an endpoint by accuracy."""
    selected = select_health_rows(rows)
    if not selected:
        raise ValueError("empty health surface")
    owners = module.factorized
    tower = owners.shared_trunk if owners.shared_trunk is not None else owners.rank_trunk
    hidden_norms = []
    def hook(_module, _args, output):
        hidden_norms.append(output.detach().norm(dim=-1).flatten().cpu())
    handle = tower.register_forward_hook(hook)
    residuals, spans, logits = [], [], []
    try:
        with torch.no_grad():
            for start in range(0, len(selected), 32):
                batch = selected[start:start + 32]
                features = torch.stack([r["query_features"] for r in batch]).to(device)
                native = torch.stack([r["native_score"] for r in batch]).to(device)
                mask = torch.stack([r["candidate_mask"] for r in batch]).to(device)
                output = module(features, native, mask)
                residual = output["rank_residual"]
                residuals.append(residual[mask].detach().cpu())
                spans.append((residual.masked_fill(~mask, -torch.inf).max(1).values
                              - residual.masked_fill(~mask, torch.inf).min(1).values).cpu())
                logits.append(output["confidence_score"].masked_fill(~mask, -torch.inf).max(1).values.cpu())
    finally:
        handle.remove()
    positives = [r for r in selected if r["kind"] == "positive"][:16]
    with torch.enable_grad():
        loss, _ = rank_loss_fn(module, positives, device=device)
        grads = torch.autograd.grad(loss, module.task_parameters("rank"), allow_unused=False)
        grad_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
    residual = torch.cat(residuals)
    span = torch.cat(spans)
    logit = torch.cat(logits)
    hidden = torch.cat(hidden_norms)
    limit = module.rank_residual_limit
    saturated = float((residual.abs() >= limit).float().mean())
    span_mean = float(span.mean())
    conf_abs = float(logit.abs().max())
    hidden_mean = float(hidden.mean())
    result = {
        "sample_count": len(selected),
        "sample_ids_sha256": hashlib.sha256("\n".join(r["sample_id"] for r in selected).encode()).hexdigest(),
        "candidate_count": residual.numel(),
        "exact_saturation_fraction": saturated,
        "residual_span_mean": span_mean,
        "rank_loss_gradient_l2": grad_norm,
        "rank_tower_hidden_norm_mean": hidden_mean,
        "confidence_max_abs": conf_abs,
        "numerical_health": conf_abs < 100 and hidden_mean < 10000 and math.isfinite(grad_norm),
        "rank_active": saturated < .99 and span_mean > 1e-6 and grad_norm > 1e-10,
        "rule": "conf_abs<100,hidden_mean<10000; saturation<.99,span>1e-6,rank_grad>1e-10",
        "accuracy_used_for_gate": False,
    }
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError("nonfinite activity diagnostic")
    return result
