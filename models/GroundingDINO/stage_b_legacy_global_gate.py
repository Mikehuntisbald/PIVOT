"""Image-expression confidence gate for frozen legacy Stage-B models."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


class LegacyStageBGlobalGate(nn.Module):
    """Predict one confidence offset per image-expression pair.

    The gate only consumes detached legacy outputs. Its scalar offset is
    broadcast across queries, so it cannot change query ordering or boxes.
    """

    score_feature_dim = 6

    def __init__(
        self,
        hidden_dim: int,
        gate_hidden_dim: int = 128,
        score_pool_temperature: float = 0.1,
        score_topk: int = 10,
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or int(gate_hidden_dim) <= 0:
            raise ValueError("hidden dimensions must be positive")
        if float(score_pool_temperature) <= 0.0:
            raise ValueError("score_pool_temperature must be positive")
        if int(score_topk) <= 0:
            raise ValueError("score_topk must be positive")

        self.hidden_dim = int(hidden_dim)
        self.score_pool_temperature = float(score_pool_temperature)
        self.score_topk = int(score_topk)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim + self.score_feature_dim, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _pool_inputs(
        self,
        query_hs: torch.Tensor,
        legacy_slot_score: torch.Tensor,
    ) -> torch.Tensor:
        if query_hs.dim() != 3:
            raise ValueError(f"query_hs must be (B,Q,D), got {tuple(query_hs.shape)}")
        if legacy_slot_score.dim() == 2:
            legacy_slot_score = legacy_slot_score.unsqueeze(-1)
        if legacy_slot_score.dim() != 3:
            raise ValueError(
                "legacy_slot_score must be (B,Q) or (B,Q,K), got "
                f"{tuple(legacy_slot_score.shape)}"
            )
        if query_hs.shape[:2] != legacy_slot_score.shape[:2]:
            raise ValueError("query_hs and legacy_slot_score must share (B,Q)")
        if int(query_hs.shape[-1]) != self.hidden_dim:
            raise ValueError(
                f"query_hs hidden dimension must be {self.hidden_dim}, got {query_hs.shape[-1]}"
            )

        # Detach at the module boundary so no caller can accidentally train the
        # localization/text backbone through the confidence objective.
        hs = query_hs.detach()
        score = legacy_slot_score.detach().to(device=hs.device, dtype=hs.dtype)
        query_count = int(score.shape[1])
        if query_count <= 0:
            raise ValueError("legacy_slot_score must contain at least one query")

        weights = torch.softmax(score / self.score_pool_temperature, dim=1)
        pooled_hs = torch.einsum("bqk,bqd->bkd", weights, self.query_norm(hs))

        top_count = min(self.score_topk, query_count)
        top_values = torch.topk(score, k=top_count, dim=1, largest=True, sorted=True).values
        score_max = top_values[:, 0]
        score_top_mean = top_values.mean(dim=1)
        score_mean = score.mean(dim=1)
        score_std = score.std(dim=1, unbiased=False)
        if query_count > 1:
            score_margin = top_values[:, 0] - torch.topk(
                score, k=2, dim=1, largest=True, sorted=True
            ).values[:, 1]
            entropy_scale = math.log(float(query_count))
            score_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
            score_entropy = score_entropy / max(entropy_scale, 1e-8)
        else:
            score_margin = torch.zeros_like(score_max)
            score_entropy = torch.zeros_like(score_max)
        score_features = torch.stack(
            (
                score_max,
                score_top_mean,
                score_mean,
                score_std,
                score_margin,
                score_entropy,
            ),
            dim=-1,
        )
        return torch.cat((pooled_hs, score_features), dim=-1)

    def forward(
        self,
        query_hs: torch.Tensor,
        legacy_slot_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if legacy_slot_score.dim() == 2:
            legacy_slot_score = legacy_slot_score.unsqueeze(-1)
        gate_inputs = self._pool_inputs(query_hs, legacy_slot_score)
        gate_bias = self.gate(gate_inputs).squeeze(-1)
        confidence = legacy_slot_score.detach() + gate_bias[:, None, :]
        return {
            "gate_bias": gate_bias,
            "confidence": confidence,
            "gate_inputs": gate_inputs,
        }


class LegacyStageBGlobalGateCriterion(nn.Module):
    """Train the gate against the deployed global maximum confidence."""

    def __init__(
        self,
        *,
        absolute_weight: float = 1.0,
        pair_weight: float = 1.0,
        tail_weight: float = 1.0,
        pair_margin: float = 0.3,
        tail_margin: float = 0.3,
        loss_temperature: float = 0.1,
        tail_fraction: float = 0.05,
        tail_objective: str = "cvar",
        require_proposalset_proxy_verified: bool = True,
    ) -> None:
        super().__init__()
        if float(loss_temperature) <= 0.0:
            raise ValueError("loss_temperature must be positive")
        if not 0.0 < float(tail_fraction) <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        tail_objective = str(tail_objective).lower().strip()
        if tail_objective not in {"cvar", "fpr95"}:
            raise ValueError("tail_objective must be 'cvar' or 'fpr95'")
        self.weight_dict = {
            "loss_legacy_gate_absolute": float(absolute_weight),
            "loss_legacy_gate_pair": float(pair_weight),
            "loss_legacy_gate_tail": float(tail_weight),
        }
        self.pair_margin = float(pair_margin)
        self.tail_margin = float(tail_margin)
        self.loss_temperature = float(loss_temperature)
        self.tail_fraction = float(tail_fraction)
        self.tail_objective = tail_objective
        self.require_proposalset_proxy_verified = bool(
            require_proposalset_proxy_verified
        )

    @staticmethod
    def _proposalset_proxy_verified_flag(target: Dict) -> bool:
        value = target.get("proposalset_proxy_verified", None)
        if torch.is_tensor(value):
            return bool(
                value.dtype == torch.bool
                and value.numel() == 1
                and value.detach().view(-1)[0].item() is True
            )
        return value is True

    @staticmethod
    def _global_max(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        confidence = outputs.get("stage_b_legacy_global_confidence", None)
        if not torch.is_tensor(confidence):
            raise KeyError(
                "LegacyStageBGlobalGateCriterion requires "
                "outputs['stage_b_legacy_global_confidence']"
            )
        if confidence.dim() == 2:
            confidence = confidence.unsqueeze(-1)
        if confidence.dim() != 3:
            raise ValueError("stage_b_legacy_global_confidence must be (B,Q,K)")
        return confidence.flatten(1).max(dim=1).values

    def _smooth_margin(self, gap: torch.Tensor, margin: float) -> torch.Tensor:
        temperature = self.loss_temperature
        return F.softplus((float(margin) - gap) / temperature).mean() * temperature

    @staticmethod
    def _exact_tpr_threshold(
        positive_score: torch.Tensor,
        target_tpr: float = 0.95,
    ) -> torch.Tensor:
        """Match the evaluator's exact ``score >= threshold`` order statistic."""
        score = positive_score.float().reshape(-1)
        if score.numel() == 0 or not bool(torch.isfinite(score).all().item()):
            raise ValueError("positive global scores must be non-empty and finite")
        if not 0.0 < float(target_tpr) <= 1.0:
            raise ValueError("target_tpr must be in (0, 1]")
        accepted = max(1, int(math.ceil(float(target_tpr) * int(score.numel()))))
        # kthvalue is one-indexed; ties are accepted by the evaluator's >= rule.
        kth = int(score.numel()) - accepted + 1
        return torch.kthvalue(score, kth).values

    def forward(self, outputs: Dict, targets: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        negative_outputs = outputs.get("stage_b_legacy_global_tn_outputs", None)
        if not isinstance(negative_outputs, dict):
            raise KeyError(
                "Legacy gate training requires paired negative outputs under "
                "'stage_b_legacy_global_tn_outputs'"
            )
        positive_score = self._global_max(outputs)
        negative_score = self._global_max(negative_outputs)
        if positive_score.shape != negative_score.shape:
            raise ValueError("positive and TN global score batches must align")
        if targets is not None and len(targets) != int(positive_score.shape[0]):
            raise ValueError("targets must align with the paired gate batch")
        if self.require_proposalset_proxy_verified:
            if targets is None or not all(
                self._proposalset_proxy_verified_flag(target) for target in targets
            ):
                raise RuntimeError(
                    "Legacy global gate rejected a non-proxy TN pair; every target "
                    "must carry proposalset_proxy_verified=True"
                )

        positive_absolute = F.binary_cross_entropy_with_logits(
            positive_score, torch.ones_like(positive_score)
        )
        negative_absolute = F.binary_cross_entropy_with_logits(
            negative_score, torch.zeros_like(negative_score)
        )
        absolute_loss = 0.5 * (positive_absolute + negative_absolute)
        pair_loss = self._smooth_margin(positive_score - negative_score, self.pair_margin)

        fpr95_threshold = self._exact_tpr_threshold(positive_score, target_tpr=0.95)
        if self.tail_objective == "fpr95":
            # FPR's denominator contains every TN. Keeping this vector intact
            # gives every negative row a gradient, including rows below the
            # current operating threshold. The existing positive absolute BCE
            # anchors the otherwise translation-invariant threshold objective.
            tail_loss = self._smooth_margin(
                fpr95_threshold - negative_score.float(), self.tail_margin
            )
            tail_positive = fpr95_threshold
            tail_negative = negative_score.float().mean()
        else:
            batch_size = int(positive_score.numel())
            tail_count = max(1, int(math.ceil(self.tail_fraction * batch_size)))
            tail_positive = torch.topk(
                positive_score, k=tail_count, largest=False, sorted=False
            ).values.mean()
            tail_negative = torch.topk(
                negative_score, k=tail_count, largest=True, sorted=False
            ).values.mean()
            tail_loss = self._smooth_margin(
                (tail_positive - tail_negative).view(1), self.tail_margin
            )

        with torch.no_grad():
            batch_exact_tpr = (
                positive_score.float() >= fpr95_threshold.detach()
            ).float().mean()
            batch_exact_fpr = (
                negative_score.float() >= fpr95_threshold.detach()
            ).float().mean()

        return {
            "loss_legacy_gate_absolute": absolute_loss,
            "loss_legacy_gate_pair": pair_loss,
            "loss_legacy_gate_tail": tail_loss,
            "legacy_gate_positive_global_max": positive_score.detach().mean(),
            "legacy_gate_tn_global_max": negative_score.detach().mean(),
            "legacy_gate_pair_win_rate": (
                positive_score.detach() > negative_score.detach()
            ).float().mean(),
            "legacy_gate_tail_positive": tail_positive.detach(),
            "legacy_gate_tail_negative": tail_negative.detach(),
            "legacy_gate_fpr95_threshold": fpr95_threshold.detach(),
            "legacy_gate_batch_exact_tpr": batch_exact_tpr,
            "legacy_gate_batch_exact_fpr95": batch_exact_fpr,
        }
