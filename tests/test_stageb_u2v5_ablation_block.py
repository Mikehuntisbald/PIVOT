from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from engine import _build_stage_b_gdino_adapter_pair_captions
from models.GroundingDINO.stage_b_gdino_score_adapter import (
    StageBGDINOScoreAdapter,
    stage_b_gdino_confidence_objective_code,
    stage_b_gdino_tn_scope_code,
)
from tools.stageb_u2v5_ablation_contract import admission_trainable_keys
from tools.aggregate_stageb_u2v5_bootstrap import _bootstrap_ref, _threshold
from tools.stageb_u2v5_ablation_registry import (
    FORMAL_ROWS,
    ROOT,
    SEEDS,
    get_row,
    parse_run_id,
    validate_registry,
)


def _target(scope: str, table_id: str, audit: str, *, benchmark: bool = False):
    return {
        "tn_scope": scope,
        "table_b_id": table_id,
        "table_b_audit_sha256": audit,
        "proposalset_proxy_verified": torch.tensor(False),
        "global_tn_verified": torch.tensor(False),
        "benchmark_dataft_alltn": torch.tensor(benchmark),
        "verifier_num_patch_slots": torch.tensor(1),
        "verifier_pair_stride": torch.tensor(2),
        "cap_list": ["a red person .", "a blue person ."],
        "is_tn": torch.tensor([False, True]),
    }


def test_registry_is_exactly_14_rows_times_three_seeds():
    validate_registry()
    assert len(FORMAL_ROWS) == 14
    assert len(FORMAL_ROWS) * len(SEEDS) == 42
    assert {row.row_id for row in FORMAL_ROWS} == {
        "A1", "A2", "A3", "A4", "C1", "C2", "C4",
        "D1", "D2", "D2m", "D3m", "O0", "O1", "O2",
    }


def test_formal_runner_locks_expandable_allocator():
    source = (ROOT / "tools/run_stageb_u2v5_ablation_matrix.py").read_text(
        encoding="utf-8"
    )
    assert 'env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"' in source


def test_run_id_and_nonformal_rows_fail_closed():
    row, seed = parse_run_id("A1:17")
    assert row.row_id == "A1" and seed == 17
    with pytest.raises(ValueError, match="not trainable"):
        parse_run_id("A5:17")
    with pytest.raises(ValueError, match="seed"):
        parse_run_id("A1:1")


def test_admission_role_allowlists_are_exact():
    assert len(admission_trainable_keys(("surface8",))) == 8
    assert len(admission_trainable_keys(("auxiliary8",))) == 8
    assert len(admission_trainable_keys(("surface8", "auxiliary8"))) == 16
    with pytest.raises(RuntimeError, match="invalid admission roles"):
        admission_trainable_keys(("confidence12",))


def test_all_leaf_configs_bind_registry_and_erase_c100():
    for row in FORMAL_ROWS:
        values = runpy.run_path(str(ROOT / row.config))
        assert values["stage_b_u2v5_ablation"] is True
        assert values["stage_b_u2v5_ablation_row_id"] == row.row_id
        assert values.get("stage_b_u2v2_c100_checkpoint") is None
        assert values.get("stage_b_u2v2_c100_sha256") is None
    a4 = runpy.run_path(
        str(ROOT / "config/ablations/cfg_stageb_u2v5_ablation_a4_no_category_complete.py")
    )
    assert a4["stage_b_u2_category_complete_supervision"] is False
    assert a4["stage_b_u2_category_loss_weight"] == 0.0
    c1 = runpy.run_path(
        str(ROOT / "config/ablations/cfg_stageb_u2v5_ablation_c1_current_batch.py")
    )
    assert c1["stage_b_gdino_confidence_objective"] == "queue_q05_st"
    assert c1["stage_b_gdino_queue_size"] == 0


def test_paired_confidence_datasets_have_one_source_only():
    for row_id in ("D1", "D2", "D2m", "D3m"):
        row = get_row(row_id)
        payload = json.loads((ROOT / row.dataset).read_text(encoding="utf-8"))
        assert len(payload["train"]) == 1
        assert payload["train"][0]["source"] == "sam3_tn_pair"
        assert payload["train"][0]["mix_weight"] == 1.0
        if row_id.endswith("m"):
            assert payload["train"][0]["paper_runtime_contract"] == (
                "v24_parent_matched_class_aligned_v2_fail_closed"
            )


@pytest.mark.parametrize(
    "scope,table_id,benchmark",
    [
        ("unverified_all_negative", "D1", True),
        ("traceable_counterfactual_edit", "D2", False),
        ("traceable_counterfactual_edit", "D2m", False),
        ("proposal_covered_verified", "D3m", False),
    ],
)
def test_clean_scope_pair_extraction_is_audit_bound(scope, table_id, benchmark):
    audit = "a" * 64
    result = _build_stage_b_gdino_adapter_pair_captions(
        [_target(scope, table_id, audit, benchmark=benchmark)],
        scope,
        expected_table_b_id=table_id,
        expected_audit_sha256=audit,
    )
    assert result[0] == ["a red person ."]
    changed = _target(scope, table_id, "b" * 64, benchmark=benchmark)
    with pytest.raises(RuntimeError, match="clean-ablation"):
        _build_stage_b_gdino_adapter_pair_captions(
            [changed], scope, expected_table_b_id=table_id,
            expected_audit_sha256=audit,
        )


def test_new_scope_and_objective_codes_are_distinct():
    assert stage_b_gdino_tn_scope_code("unverified_all_negative") == 4
    assert stage_b_gdino_tn_scope_code("traceable_counterfactual_edit") == 5
    assert stage_b_gdino_confidence_objective_code(
        "detached_recent_q05_scope_labeled"
    ) == 5


def test_unverified_scope_never_requires_alltn_upgrade():
    from models.GroundingDINO.stage_b_gdino_score_adapter import (
        StageBGDINOScoreAdapterCriterion,
    )

    criterion = StageBGDINOScoreAdapterCriterion(
        tn_scope="unverified_all_negative",
        train_mode="confidence_only",
        confidence_objective="detached_recent_q05_scope_labeled",
        queue_size=1,
        queue_min_count=1,
    )
    assert criterion.tn_scope == "unverified_all_negative"


def test_bootstrap_is_deterministic_and_clusters_all_seed_draws():
    reference = {
        "a": {"sample_id": "a", "image_id": 1, "ann_id": 1, "ref_id": 1, "sent_id": 1, "correct50": False},
        "b": {"sample_id": "b", "image_id": 1, "ann_id": 2, "ref_id": 2, "sent_id": 2, "correct50": True},
        "c": {"sample_id": "c", "image_id": 2, "ann_id": 3, "ref_id": 3, "sent_id": 3, "correct50": False},
    }
    candidate = {}
    for seed in SEEDS:
        candidate[seed] = {
            key: {**value, "correct50": True}
            for key, value in reference.items()
        }
    first = _bootstrap_ref(
        candidate, reference, iterations=20,
        rng=np.random.default_rng(7),
    )
    second = _bootstrap_ref(
        candidate, reference, iterations=20,
        rng=np.random.default_rng(7),
    )
    assert first == second
    assert first["unique_images"] == 2
    assert first["gain"] == pytest.approx(2 / 3)


def test_fpr_threshold_recomputes_positive_q05():
    assert _threshold(np.asarray([0.1, 0.2, 0.3, 0.4]), 0.5) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "ownership,expect_rank_grad,expect_rank_output_grad",
    [
        ("shared_score", True, True),
        ("shared_trunk_two_heads", True, False),
        ("isolated_heads", False, False),
    ],
)
def test_ownership_confidence_gradient_graph(
    ownership, expect_rank_grad, expect_rank_output_grad
):
    module = StageBGDINOScoreAdapter(
        8, adapter_dim=4, gate_hidden_dim=4,
        u2v5_score_ownership=ownership,
    )
    with torch.no_grad():
        module.rank_output.weight.fill_(0.1)
        module.confidence_gate[-1].weight.fill_(0.1)
    output = module(torch.randn(2, 3, 8), torch.randn(2, 3))
    output["confidence_score"].sum().backward()
    rank_trunk_grad = module.rank_trunk[0].weight.grad
    rank_output_grad = module.rank_output.weight.grad
    assert (rank_trunk_grad is not None) is expect_rank_grad
    assert (rank_output_grad is not None) is expect_rank_output_grad
