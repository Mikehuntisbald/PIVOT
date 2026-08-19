from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_original_ogc_score_contract_is_predeclared_and_distinct() -> None:
    from tools.arrow_original_gdino_common import (
        PRIMARY_SCORE,
        SENSITIVITY_SCORE,
    )
    from tools.eval_arrow_original_gdino_finecops import _query_scores

    # B=1, Q=2, T=4. Only token columns 1 and 2 belong to the expression.
    logits = torch.tensor(
        [[[-10.0, 0.0, 2.0, 20.0], [-10.0, 1.0, 1.0, 20.0]]]
    )
    phrase = torch.zeros((1, 2, 4), dtype=torch.bool)
    phrase[0, 0, 1] = True
    phrase[0, 1, 2] = True
    outputs = {
        "pred_logits": logits,
        "pred_boxes": torch.zeros((1, 2, 4)),
        "phrase_to_token_mask": phrase,
    }
    scores = _query_scores(outputs)
    assert set(scores) == {PRIMARY_SCORE, SENSITIVITY_SCORE}
    expected_mean = logits.sigmoid()[:, :, 1:3].mean(dim=2)
    expected_max = logits.sigmoid()[:, :, 1:3].max(dim=2).values
    assert torch.equal(scores[PRIMARY_SCORE], expected_mean)
    assert torch.equal(scores[SENSITIVITY_SCORE], expected_max)
    # The primary matched mean and upstream-native sensitivity may select
    # different queries; their roles must never be chosen after evaluation.
    assert scores[PRIMARY_SCORE].argmax(dim=1).item() == 1
    assert scores[SENSITIVITY_SCORE].argmax(dim=1).item() == 0


def test_original_ogc_score_contract_rejects_empty_mask() -> None:
    from tools.eval_arrow_original_gdino_finecops import _query_scores

    outputs = {
        "pred_logits": torch.zeros((1, 2, 4)),
        "pred_boxes": torch.zeros((1, 2, 4)),
        "phrase_to_token_mask": torch.zeros((1, 1, 4), dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="empty expression mask"):
        _query_scores(outputs)


def test_original_ogc_config_is_expression_only() -> None:
    from util.slconfig import SLConfig

    root = Path(__file__).resolve().parents[1]
    cfg = SLConfig.fromfile(
        str(root / "config/ablations/cfg_arrow_original_gdino_swint_ogc.py")
    )
    assert cfg.stage_b_arrow_original_ogc_eval is True
    assert cfg.backbone == "swin_T_224_1k"
    assert cfg.dn_labelbook_size == 2000
    assert cfg.num_queries == 900
    assert cfg.data_aug_hflip_prob == 0.0
    for key in (
        "patch_only",
        "stage_b",
        "stage_b_gdino_score_adapter",
        "stage_b_u0_patch_rank",
        "stage_b_data_driven_score",
        "stage_b_native_patch_category",
        "enable_patch_branch",
    ):
        assert getattr(cfg, key) is False


def test_original_ogc_checkpoint_identity_is_exact() -> None:
    from tools.arrow_finecops_common import file_record
    from tools.arrow_original_gdino_common import (
        CHECKPOINT_SHA256,
        CHECKPOINT_SIZE,
    )

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "weights/groundingdino_swint_ogc.pth"
    if not checkpoint.is_file():
        pytest.skip("original OGC checkpoint is an external release artifact")
    record = file_record(checkpoint)
    assert record["sha256"] == CHECKPOINT_SHA256
    assert record["size_bytes"] == CHECKPOINT_SIZE
