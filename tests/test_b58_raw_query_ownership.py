from __future__ import annotations

from pathlib import Path

import torch


def test_b58_raw_query_matrix_matches_parent_100k_heads() -> None:
    from tools.b58_raw_query_ownership import FORMAL_SEEDS, OWNERS
    from tools.mmgdino_e5_ownership import MMGDinoE5ResponsibilityOwners

    assert OWNERS == ("shared_wide", "isolated_128")
    assert FORMAL_SEEDS == (17, 42, 73)
    shared = MMGDinoE5ResponsibilityOwners(ownership=OWNERS[0]).architecture_report()
    isolated = MMGDinoE5ResponsibilityOwners(ownership=OWNERS[1]).architecture_report()
    assert (shared.trainable_parameters, shared.macs_per_query_both_outputs) == (
        100362, 99424,
    )
    assert (
        isolated.trainable_parameters,
        isolated.macs_per_query_both_outputs,
    ) == (100358, 98816)


def test_b58_config_disables_all_historical_stageb_branches() -> None:
    from util.slconfig import SLConfig

    root = Path(__file__).resolve().parents[1]
    cfg = SLConfig.fromfile(
        str(root / "config/ablations/cfg_b58_raw_query_ownership_eval.py")
    )
    assert cfg.stage_b_b58_raw_query_ownership_eval is True
    assert cfg.num_queries == 900
    assert cfg.data_aug_hflip_prob == 0.0
    for name in (
        "stage_b", "patch_only", "enable_patch_branch",
        "stage_b_gdino_score_adapter", "stage_b_u0_patch_rank",
        "stage_b_data_driven_score", "stage_b_native_patch_category",
    ):
        assert getattr(cfg, name) is False


def test_b58_uses_identical_expression_score_contract() -> None:
    from tools.extract_original_gdino_ownership_cache import (
        original_expression_mean,
        preprocess_original_caption,
    )

    logits = torch.tensor([[[0.0, 2.0, 30.0], [1.0, 1.0, 30.0]]])
    phrase = torch.zeros((1, 2, 3), dtype=torch.bool)
    phrase[0, 0, 0] = True
    phrase[0, 1, 1] = True
    assert torch.equal(
        original_expression_mean(logits, phrase),
        logits.sigmoid()[:, :, :2].mean(dim=2),
    )
    assert preprocess_original_caption(" Old Lady ") == "old lady."


def test_commands_bind_b58_and_exact_parent_schedules() -> None:
    from tools.b58_raw_query_ownership import (
        CHECKPOINT_SHA256,
        REF_INPUTS,
        TRUNK_ID,
        eval_cache_path,
        training_cache_path,
    )
    from tools.run_b58_raw_query_ownership import (
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
    assert "gdino_ft_stageb_from_gdino_ft_e1" in joined
    assert "extract_b58_raw_query_ownership_cache.py" in joined
    assert "rank_seed42.jsonl" in joined
    assert "d3_seed42.jsonl" in joined
    assert set(REF_INPUTS) == {
        "refcoco_testA", "refcoco_testB", "refcocop_testA",
        "refcocop_testB", "refcocog_test", "strict2031",
    }
    evaluate = _eval_extract_command(TRUNK_ID, "strict2031")
    assert "tn" in evaluate
    assert str(eval_cache_path(TRUNK_ID, "strict2031")) in evaluate


def test_b58_runtime_adapter_restores_sealed_parent_engine() -> None:
    import tools.extract_original_gdino_ownership_cache as engine
    from tools.extract_b58_raw_query_ownership_cache import (
        B58FrozenRuntime,
        _runtime_context,
    )

    before = engine.OriginalGDINOFrozenRuntime
    with _runtime_context():
        assert engine.OriginalGDINOFrozenRuntime is B58FrozenRuntime
    assert engine.OriginalGDINOFrozenRuntime is before


def test_b58_aggregate_context_restores_mature_globals() -> None:
    import tools.aggregate_mmgdino_pretrain_ownership as mature
    from tools.aggregate_b58_raw_query_ownership import _context
    from tools.b58_raw_query_ownership import TRUNK_ID

    before = mature.TRUNK_ID
    with _context():
        assert mature.TRUNK_ID == TRUNK_ID
    assert mature.TRUNK_ID == before


def test_lineage_declares_only_trunk_weights_change() -> None:
    from tools.b58_raw_query_ownership import (
        CHECKPOINT,
        PARENT_CHECKPOINT,
        PARENT_TO_B58_CHANGED_TENSORS,
        PARENT_TO_B58_UNCHANGED_TENSORS,
        PURE_TRUNK_TENSORS,
    )

    assert "ogc_original_finetune_stage_a" in str(PARENT_CHECKPOINT)
    assert "gdino_ft_stageb_from_gdino_ft_e1" in str(CHECKPOINT)
    assert PURE_TRUNK_TENSORS == 938
    assert PARENT_TO_B58_CHANGED_TENSORS == 727
    assert PARENT_TO_B58_UNCHANGED_TENSORS == 211
