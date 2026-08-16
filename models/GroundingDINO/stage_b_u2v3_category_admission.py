"""U2-v3 category-admission-only objective.

The mathematical objective reuses the audited deployment-aligned D10 gate
geometry, but U2-v3 has an independent public contract and provenance.  Its
only differentiable owner is the Stage-A patch projection surface; B58, R100,
C100, and the U0 compatibility shell remain frozen.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .stage_b_u0_gate_aligned_d10 import StageBU0GateAlignedD10Criterion


STAGE_B_U2V3_CATEGORY_ADMISSION_CONTRACT_VERSION = 1
STAGE_B_U2V3_CATEGORY_ADMISSION_LOSS = (
    "loss_stage_b_u2v3_category_admission"
)


class StageBU2V3CategoryAdmissionCriterion(nn.Module):
    """Expose D10's hard-gate-aligned loss under a clean U2-v3 contract."""

    def __init__(self, *, weight: float = 1.0, **geometry: Any) -> None:
        super().__init__()
        self.core = StageBU0GateAlignedD10Criterion(weight=1.0, **geometry)
        self.weight_dict = {STAGE_B_U2V3_CATEGORY_ADMISSION_LOSS: float(weight)}
        if not torch.isfinite(torch.as_tensor(float(weight))) or float(weight) <= 0.0:
            raise ValueError("U2-v3 category-admission weight must be finite and positive")
        self.register_buffer(
            "criterion_contract_version",
            torch.as_tensor(
                STAGE_B_U2V3_CATEGORY_ADMISSION_CONTRACT_VERSION,
                dtype=torch.int64,
            ),
            persistent=True,
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        core = self.core(outputs, targets)
        core_loss_key = next(iter(self.core.weight_dict))
        result = {
            STAGE_B_U2V3_CATEGORY_ADMISSION_LOSS: core[core_loss_key],
        }
        prefix = "stage_b_u0_gate_aligned_d10_"
        for key, value in core.items():
            if key.startswith(prefix):
                result["stage_b_u2v3_" + key.removeprefix(prefix)] = value
        return result


__all__ = [
    "STAGE_B_U2V3_CATEGORY_ADMISSION_CONTRACT_VERSION",
    "STAGE_B_U2V3_CATEGORY_ADMISSION_LOSS",
    "StageBU2V3CategoryAdmissionCriterion",
]
