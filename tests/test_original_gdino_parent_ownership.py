from __future__ import annotations

from pathlib import Path

import torch


def test_parent_matrix_reuses_capacity_matched_mature_heads() -> None:
    from tools.mmgdino_e5_ownership import MMGDinoE5ResponsibilityOwners
    from tools.original_gdino_parent_ownership import FORMAL_SEEDS, OWNERS

    assert OWNERS == ("shared_wide", "isolated_128")
    assert FORMAL_SEEDS == (17, 42, 73)
    shared = MMGDinoE5ResponsibilityOwners(
        ownership=OWNERS[0]
    ).architecture_report()
    isolated = MMGDinoE5ResponsibilityOwners(
        ownership=OWNERS[1]
    ).architecture_report()
    assert (shared.trainable_parameters, shared.macs_per_query_both_outputs) == (
        100362,
        99424,
    )
    assert (
        isolated.trainable_parameters,
        isolated.macs_per_query_both_outputs,
    ) == (100358, 98816)


def test_parent_config_is_pure_gdino_and_expression_only() -> None:
    from util.slconfig import SLConfig

    root = Path(__file__).resolve().parents[1]
    cfg = SLConfig.fromfile(
        str(root / "config/ablations/cfg_original_gdino_parent_ownership_eval.py")
    )
    assert cfg.stage_b_original_gdino_parent_ownership_eval is True
    assert cfg.num_queries == 900
    assert cfg.dn_labelbook_size == 91
    assert cfg.data_aug_hflip_prob == 0.0
    for name in (
        "stage_b",
        "patch_only",
        "enable_patch_branch",
        "stage_b_gdino_score_adapter",
        "stage_b_u0_patch_rank",
        "stage_b_data_driven_score",
        "stage_b_native_patch_category",
    ):
        assert getattr(cfg, name) is False


def test_expression_mean_uses_only_generated_phrase_tokens() -> None:
    from tools.extract_original_gdino_ownership_cache import (
        original_expression_mean,
    )

    logits = torch.tensor(
        [[[-10.0, 0.0, 2.0, 20.0], [-10.0, 1.0, 1.0, 20.0]]]
    )
    phrase = torch.zeros((1, 2, 4), dtype=torch.bool)
    phrase[0, 0, 1] = True
    phrase[0, 1, 2] = True
    observed = original_expression_mean(logits, phrase)
    expected = logits.sigmoid()[:, :, 1:3].mean(dim=2)
    assert torch.equal(observed, expected)


def test_commands_bind_direct_parent_and_all_surfaces() -> None:
    from tools.original_gdino_parent_ownership import (
        CHECKPOINT_SHA256,
        REF_INPUTS,
        TRUNK_ID,
        eval_cache_path,
        training_cache_path,
    )
    from tools.run_original_gdino_parent_ownership import (
        _eval_extract_command,
        _training_extract_command,
    )

    train = _training_extract_command(
        TRUNK_ID,
        42,
        output=training_cache_path(TRUNK_ID, 42),
        receipt=training_cache_path(TRUNK_ID, 42).with_suffix(".receipt.json"),
    )
    joined = " ".join(train)
    assert CHECKPOINT_SHA256 in train
    assert "ogc_original_finetune_stage_a/checkpoint0001.pth" in joined
    assert "rank_seed42.jsonl" in joined
    assert "d3_seed42.jsonl" in joined
    assert set(REF_INPUTS) == {
        "refcoco_testA",
        "refcoco_testB",
        "refcocop_testA",
        "refcocop_testB",
        "refcocog_test",
        "strict2031",
    }
    evaluate = _eval_extract_command(TRUNK_ID, "strict2031")
    assert "--mode" in evaluate and "tn" in evaluate
    assert str(eval_cache_path(TRUNK_ID, "strict2031")) in evaluate


def test_lineage_contract_is_the_immediate_b58_parent() -> None:
    from tools.original_gdino_parent_ownership import (
        B58_CHECKPOINT,
        CHECKPOINT,
        PARENT_TO_B58_CHANGED_TENSORS,
        PARENT_TO_B58_UNCHANGED_TENSORS,
        PARENT_UNUSED_PATCH_TENSORS,
        PURE_TRUNK_TENSORS,
    )

    assert CHECKPOINT.name == B58_CHECKPOINT.name == "checkpoint0001.pth"
    assert "ogc_original_finetune_stage_a" in str(CHECKPOINT)
    assert "gdino_ft_stageb_from_gdino_ft_e1" in str(B58_CHECKPOINT)
    assert PURE_TRUNK_TENSORS == 938
    assert PARENT_UNUSED_PATCH_TENSORS == 200
    assert PARENT_TO_B58_CHANGED_TENSORS == 727
    assert PARENT_TO_B58_UNCHANGED_TENSORS == 211


def test_original_aggregate_context_restores_mature_globals() -> None:
    import tools.aggregate_mmgdino_pretrain_ownership as mature
    from tools.aggregate_original_gdino_parent_ownership import _context
    from tools.original_gdino_parent_ownership import TRUNK_ID

    before = mature.TRUNK_ID
    with _context():
        assert mature.TRUNK_ID == TRUNK_ID
    assert mature.TRUNK_ID == before
