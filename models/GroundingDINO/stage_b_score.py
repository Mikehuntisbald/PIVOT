from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def gdino_focal_match_cost_from_logits(
    logits: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = logits.sigmoid()
    neg_cost = (1 - float(alpha)) * (prob ** float(gamma)) * (-(1 - prob + 1e-8).log())
    pos_cost = float(alpha) * ((1 - prob) ** float(gamma)) * (-(prob + 1e-8).log())
    return pos_cost - neg_cost


def aggregate_stage_b_tokens(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    token_weight: Optional[torch.Tensor] = None,
    text_agg: str = "mean",
    softmin_tau: float = 0.7,
    mean_softmin_alpha: float = 0.5,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"pred_logits_text must be (B,Q,T), got {tuple(logits.shape)}")
    if mask.dim() != 3:
        raise ValueError(f"token mask must be (B,K,T), got {tuple(mask.shape)}")
    if logits.shape[0] != mask.shape[0] or logits.shape[-1] != mask.shape[-1]:
        raise ValueError(
            f"Token mask/logit shape mismatch: logits={tuple(logits.shape)} mask={tuple(mask.shape)}"
        )

    text_agg = str(text_agg).lower().strip()
    mask = mask.to(device=logits.device, dtype=torch.bool)
    weight = None
    if token_weight is not None:
        if token_weight.shape != mask.shape:
            raise ValueError(
                "token_weight must have the same shape as mask, "
                f"got weight={tuple(token_weight.shape)} mask={tuple(mask.shape)}"
            )
        weight = token_weight.to(device=logits.device, dtype=logits.dtype).masked_fill(~mask, 0.0)
    token_scores = logits.sigmoid()
    z = token_scores[:, :, None, :]  # (B,Q,1,T)
    m = mask[:, None, :, :]  # (B,1,K,T)
    w = weight[:, None, :, :] if weight is not None else m.to(logits.dtype)
    valid = (w > 0).any(dim=-1)  # (B,1,K)

    if text_agg == "mean":
        denom = w.sum(dim=-1).clamp(min=1e-6)
        score = (z.masked_fill(~m, 0.0) * w).sum(dim=-1) / denom
    elif text_agg == "max":
        effective = m & (w > 0)
        score = z.masked_fill(~effective, torch.finfo(logits.dtype).min).max(dim=-1).values
    elif text_agg == "softmin":
        tau = max(float(softmin_tau), 1e-6)
        weighted_logits = w.clamp(min=1e-12).log() - z / tau
        score = -tau * torch.logsumexp(
            weighted_logits.masked_fill(w <= 0, torch.finfo(logits.dtype).min),
            dim=-1,
        )
    elif text_agg in {"mean_norm_softmin", "mean_normalized_softmin"}:
        tau = max(float(softmin_tau), 1e-6)
        alpha = min(1.0, max(0.0, float(mean_softmin_alpha)))
        denom = w.sum(dim=-1).clamp(min=1e-6)
        mean_score = (z.masked_fill(~m, 0.0) * w).sum(dim=-1) / denom
        weighted_logits = w.clamp(min=1e-12).log() - z / tau
        softmin_score = -tau * torch.logsumexp(
            weighted_logits.masked_fill(w <= 0, torch.finfo(logits.dtype).min),
            dim=-1,
        )
        normalized_softmin_score = softmin_score + tau * denom.log()
        score = alpha * mean_score + (1.0 - alpha) * normalized_softmin_score
    else:
        raise ValueError(f"Unsupported Stage B text aggregator: {text_agg}")

    return score.masked_fill(~valid, 0.0)  # (B,Q,K)


def aggregate_stage_b_token_match_cost(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    token_weight: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"pred_logits_text must be (B,Q,T), got {tuple(logits.shape)}")
    if mask.dim() != 3:
        raise ValueError(f"token mask must be (B,K,T), got {tuple(mask.shape)}")
    if logits.shape[0] != mask.shape[0] or logits.shape[-1] != mask.shape[-1]:
        raise ValueError(
            f"Token mask/logit shape mismatch: logits={tuple(logits.shape)} mask={tuple(mask.shape)}"
        )

    mask = mask.to(device=logits.device, dtype=torch.bool)
    weight = None
    if token_weight is not None:
        if token_weight.shape != mask.shape:
            raise ValueError(
                "token_weight must have the same shape as mask, "
                f"got weight={tuple(token_weight.shape)} mask={tuple(mask.shape)}"
            )
        weight = token_weight.to(device=logits.device, dtype=logits.dtype).masked_fill(~mask, 0.0)
    token_cost = gdino_focal_match_cost_from_logits(logits, alpha=alpha, gamma=gamma)
    z = token_cost[:, :, None, :]  # (B,Q,1,T)
    m = mask[:, None, :, :]  # (B,1,K,T)
    w = weight[:, None, :, :] if weight is not None else m.to(logits.dtype)
    valid = (w > 0).any(dim=-1)  # (B,1,K)
    denom = w.sum(dim=-1).clamp(min=1e-6)
    cost = (z.masked_fill(~m, 0.0) * w).sum(dim=-1) / denom
    return cost.masked_fill(~valid, 0.0)  # (B,Q,K)


def stage_b_weighted_phrase_mask(
    phrase_to_token_mask: torch.Tensor,
    canonical_to_token_mask: Optional[torch.Tensor],
    *,
    canonical_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    phrase_to_token_mask = phrase_to_token_mask.to(dtype=torch.bool)
    if canonical_to_token_mask is None:
        canonical_to_token_mask = torch.zeros_like(phrase_to_token_mask)
    else:
        canonical_to_token_mask = canonical_to_token_mask.to(
            device=phrase_to_token_mask.device,
            dtype=torch.bool,
        )
    canonical_to_token_mask = canonical_to_token_mask & phrase_to_token_mask
    token_weight = torch.ones(
        phrase_to_token_mask.shape,
        device=phrase_to_token_mask.device,
        dtype=torch.float32,
    )
    token_weight = token_weight.masked_fill(canonical_to_token_mask, float(canonical_weight))
    token_weight = token_weight.masked_fill(~phrase_to_token_mask, 0.0)
    return phrase_to_token_mask, token_weight


def compute_stage_b_slot_logits(
    outputs: Dict[str, torch.Tensor],
    *,
    beta: float = 1.0,
    canonical_weight: float = 1.0,
    text_agg: str = "mean",
    softmin_tau: float = 0.7,
    mean_softmin_alpha: float = 0.5,
    detach_patch: bool = False,
    normalize_fused_score: bool = True,
) -> torch.Tensor:
    pred_logits_patch = outputs.get("pred_logits_patch", None)
    pred_logits_text = outputs.get("pred_logits_text", None)
    phrase_to_token_mask = outputs.get("phrase_to_token_mask", None)
    if pred_logits_patch is None:
        raise KeyError("Stage B slot scoring requires outputs['pred_logits_patch'].")
    if pred_logits_text is None:
        raise KeyError("Stage B slot scoring requires outputs['pred_logits_text'].")
    if phrase_to_token_mask is None:
        raise KeyError("Stage B slot scoring requires outputs['phrase_to_token_mask'].")

    if pred_logits_patch.dim() == 2:
        pred_logits_patch = pred_logits_patch.unsqueeze(-1)
    elif pred_logits_patch.dim() != 3:
        raise ValueError(
            f"pred_logits_patch must be (B,Q) or (B,Q,K), got {tuple(pred_logits_patch.shape)}"
        )
    if detach_patch:
        pred_logits_patch = pred_logits_patch.detach()

    B, _Q, K = pred_logits_patch.shape
    T = pred_logits_text.shape[-1]
    phrase_to_token_mask = phrase_to_token_mask.to(device=pred_logits_text.device, dtype=torch.bool)
    if phrase_to_token_mask.shape[0] != B or phrase_to_token_mask.shape[-1] != T:
        raise ValueError(
            "phrase_to_token_mask must be shaped (B,K,T) and share B/T with pred_logits_text, "
            f"got phrase={tuple(phrase_to_token_mask.shape)} text={tuple(pred_logits_text.shape)}"
        )
    if phrase_to_token_mask.shape[1] < K:
        raise ValueError(
            f"phrase_to_token_mask has fewer slots than patch logits: {phrase_to_token_mask.shape[1]} < {K}"
        )
    phrase_to_token_mask = phrase_to_token_mask[:, :K, :]

    canonical_to_token_mask = outputs.get("canonical_to_token_mask", None)
    if canonical_to_token_mask is None:
        canonical_to_token_mask = torch.zeros_like(phrase_to_token_mask)
    else:
        canonical_to_token_mask = canonical_to_token_mask.to(device=pred_logits_text.device, dtype=torch.bool)
        if canonical_to_token_mask.shape[0] != B or canonical_to_token_mask.shape[-1] != T:
            raise ValueError(
                "canonical_to_token_mask must be shaped (B,K,T) and share B/T with pred_logits_text, "
                f"got canonical={tuple(canonical_to_token_mask.shape)} text={tuple(pred_logits_text.shape)}"
            )
        if canonical_to_token_mask.shape[1] < K:
            raise ValueError(
                f"canonical_to_token_mask has fewer slots than patch logits: {canonical_to_token_mask.shape[1]} < {K}"
            )
        canonical_to_token_mask = canonical_to_token_mask[:, :K, :] & phrase_to_token_mask

    phrase_score_mask, phrase_token_weight = stage_b_weighted_phrase_mask(
        phrase_to_token_mask,
        canonical_to_token_mask,
        canonical_weight=canonical_weight,
    )
    text_score = aggregate_stage_b_tokens(
        pred_logits_text,
        phrase_score_mask,
        token_weight=phrase_token_weight,
        text_agg=text_agg,
        softmin_tau=softmin_tau,
        mean_softmin_alpha=mean_softmin_alpha,
    )

    patch_score = pred_logits_patch.to(pred_logits_text.device).sigmoid()
    slot_logits = patch_score + float(beta) * text_score
    if normalize_fused_score:
        slot_logits = slot_logits / max(1.0 + float(beta), 1e-6)
    patch_mask = outputs.get("patch_mask", outputs.get("patch_phrase_mask", None))
    if patch_mask is not None:
        patch_mask = patch_mask.to(device=slot_logits.device, dtype=torch.bool)
        if patch_mask.shape[0] == B and patch_mask.shape[1] >= K:
            slot_logits = slot_logits.masked_fill(~patch_mask[:, None, :K], -100.0)
    return slot_logits


def compute_stage_b_slot_match_cost(
    outputs: Dict[str, torch.Tensor],
    *,
    beta: float = 1.0,
    canonical_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    detach_patch: bool = False,
) -> torch.Tensor:
    pred_logits_patch = outputs.get("pred_logits_patch", None)
    pred_logits_text = outputs.get("pred_logits_text", None)
    phrase_to_token_mask = outputs.get("phrase_to_token_mask", None)
    if pred_logits_patch is None:
        raise KeyError("Stage B matching requires outputs['pred_logits_patch'].")
    if pred_logits_text is None:
        raise KeyError("Stage B matching requires outputs['pred_logits_text'].")
    if phrase_to_token_mask is None:
        raise KeyError("Stage B matching requires outputs['phrase_to_token_mask'].")

    if pred_logits_patch.dim() == 2:
        pred_logits_patch = pred_logits_patch.unsqueeze(-1)
    elif pred_logits_patch.dim() != 3:
        raise ValueError(
            f"pred_logits_patch must be (B,Q) or (B,Q,K), got {tuple(pred_logits_patch.shape)}"
        )
    if detach_patch:
        pred_logits_patch = pred_logits_patch.detach()

    B, _Q, K = pred_logits_patch.shape
    T = pred_logits_text.shape[-1]
    phrase_to_token_mask = phrase_to_token_mask.to(device=pred_logits_text.device, dtype=torch.bool)
    if phrase_to_token_mask.shape[0] != B or phrase_to_token_mask.shape[-1] != T:
        raise ValueError(
            "phrase_to_token_mask must be shaped (B,K,T) and share B/T with pred_logits_text, "
            f"got phrase={tuple(phrase_to_token_mask.shape)} text={tuple(pred_logits_text.shape)}"
        )
    if phrase_to_token_mask.shape[1] < K:
        raise ValueError(
            f"phrase_to_token_mask has fewer slots than patch logits: {phrase_to_token_mask.shape[1]} < {K}"
        )
    phrase_to_token_mask = phrase_to_token_mask[:, :K, :]

    canonical_to_token_mask = outputs.get("canonical_to_token_mask", None)
    if canonical_to_token_mask is None:
        canonical_to_token_mask = torch.zeros_like(phrase_to_token_mask)
    else:
        canonical_to_token_mask = canonical_to_token_mask.to(device=pred_logits_text.device, dtype=torch.bool)
        if canonical_to_token_mask.shape[0] != B or canonical_to_token_mask.shape[-1] != T:
            raise ValueError(
                "canonical_to_token_mask must be shaped (B,K,T) and share B/T with pred_logits_text, "
                f"got canonical={tuple(canonical_to_token_mask.shape)} text={tuple(pred_logits_text.shape)}"
            )
        if canonical_to_token_mask.shape[1] < K:
            raise ValueError(
                f"canonical_to_token_mask has fewer slots than patch logits: {canonical_to_token_mask.shape[1]} < {K}"
            )
        canonical_to_token_mask = canonical_to_token_mask[:, :K, :] & phrase_to_token_mask

    phrase_cost_mask, phrase_token_weight = stage_b_weighted_phrase_mask(
        phrase_to_token_mask,
        canonical_to_token_mask,
        canonical_weight=canonical_weight,
    )
    text_cost = aggregate_stage_b_token_match_cost(
        pred_logits_text,
        phrase_cost_mask,
        token_weight=phrase_token_weight,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    patch_cost = gdino_focal_match_cost_from_logits(
        pred_logits_patch.to(pred_logits_text.device),
        alpha=focal_alpha,
        gamma=focal_gamma,
    )
    match_cost = patch_cost + float(beta) * text_cost
    patch_mask = outputs.get("patch_mask", outputs.get("patch_phrase_mask", None))
    if patch_mask is not None:
        patch_mask = patch_mask.to(device=match_cost.device, dtype=torch.bool)
        if patch_mask.shape[0] == B and patch_mask.shape[1] >= K:
            match_cost = match_cost.masked_fill(~patch_mask[:, None, :K], 1e6)
    return match_cost
