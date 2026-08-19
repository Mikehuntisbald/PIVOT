from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.arrow_grefcoco_common import EXPECTED_SHA256, reject_train_surface
from tools.aggregate_arrow_grefcoco import _binary_metrics, _threshold_at_tpr, _weighted_auc
from tools.prepare_arrow_grefcoco import _classify_ref, _normalize_text


def test_official_annotation_hashes_when_present() -> None:
    from tools.arrow_grefcoco_common import digest_file

    root = Path("/home/haoyi/Downloads/grefs")
    if not root.is_dir():
        pytest.skip("local official gRefCOCO annotation is unavailable")
    assert {name: digest_file(root / name) for name in EXPECTED_SHA256} == EXPECTED_SHA256


def test_ref_kind_contract_fails_closed() -> None:
    assert _classify_ref({"ann_id": [-1], "category_id": [-1], "no_target": True}) == "negative"
    assert _classify_ref({"ann_id": [3], "category_id": [1], "no_target": False}) == "positive"
    assert _classify_ref({"ann_id": [3, 4], "category_id": [1, 1], "no_target": False}) == "multi"
    with pytest.raises(ValueError):
        _classify_ref({"ann_id": [3], "category_id": [1], "no_target": True})
    with pytest.raises(ValueError):
        _classify_ref({"ann_id": [], "category_id": [], "no_target": False})


def test_train_surface_is_forbidden() -> None:
    with pytest.raises(ValueError, match="train"):
        reject_train_surface("train")
    reject_train_surface("testA")


def test_expression_normalization_preserves_punctuation() -> None:
    assert _normalize_text(" A  Dog! ") == "a dog!"
    assert _normalize_text("A dog") != _normalize_text("A dog!")


def test_binary_metrics_ties_and_threshold_comparison() -> None:
    labels = np.asarray([1, 0])
    scores = np.asarray([0.5, 0.5])
    result = _binary_metrics(labels, scores)
    assert result["auroc"] == 0.5
    assert result["aupr"] == 0.5
    assert result["fpr95"] == 1.0
    assert result["actual_tpr"] == 1.0
    assert _threshold_at_tpr(np.asarray([3.0, 2.0, 1.0]), 0.95) == 1.0


def test_weighted_auc_matches_expanded_cluster_sample() -> None:
    labels = np.asarray([1, 0, 1])
    scores = np.asarray([0.9, 0.8, 0.1])
    weights = np.asarray([2.0, 1.0, 1.0])
    expanded_labels = np.repeat(labels, weights.astype(int))
    expanded_scores = np.repeat(scores, weights.astype(int))
    assert _weighted_auc(labels, scores, weights) == pytest.approx(
        _binary_metrics(expanded_labels, expanded_scores)["auroc"]
    )


def test_actual_manifest_counts_and_no_fake_negative_targets() -> None:
    root = Path("/media/haoyi/T9/data/gRefCOCO/v1/manifests")
    path = root / "grefcoco_testab_single_no_target.jsonl"
    if not path.is_file():
        pytest.skip("prepared gRefCOCO manifest is unavailable")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert len(rows) == 20684
    assert sum(row["kind"] == "positive" for row in rows) == 11563
    assert sum(row["kind"] == "negative" for row in rows) == 9121
    assert len({row["sample_id"] for row in rows}) == len(rows)
    for row in rows:
        if row["kind"] == "negative":
            assert row["ann_id"] is None
            assert row["bbox_xywh"] is None
            assert row["admission_defined"] is False


def test_actual_overlap_surfaces_are_frozen() -> None:
    path = Path("/media/haoyi/T9/data/gRefCOCO/v1/manifests/overlap_audit.json")
    if not path.is_file():
        pytest.skip("gRefCOCO overlap audit is unavailable")
    audit = json.loads(path.read_text())
    assert audit["stagea"]["pixel_byte_audit"] == {"byte_identical": 1500, "missing": 0, "mismatch": 0}
    assert audit["r100"]["images"] == 942
    assert audit["d3"]["union_images"] == 212
    assert audit["finecops_source_crosswalk_images"] == 14
    assert audit["refcoco_exact_positive_expression_overlap"] == 10479
    assert audit["surfaces"]["d3_disjoint"] == {
        "by_split_images": {"testA": 657, "testB": 631},
        "images": 1288,
        "negative": 7796,
        "positive": 9924,
    }


def test_gref_config_is_deterministic_confidence_only() -> None:
    pytest.importorskip("torch")
    from util.slconfig import SLConfig

    cfg = SLConfig.fromfile("config/ablations/cfg_arrow_grefcoco_confidence_only.py")
    assert cfg.stage_b_arrow_grefcoco_eval is True
    assert cfg.stage_b_arrow_grefcoco_confidence_only is True
    assert cfg.data_aug_scales == [800]
    assert cfg.data_aug_max_size == 1333
    assert cfg.data_aug_hflip_prob == 0.0


def test_gdino_caption_adds_only_terminal_phrase_delimiter() -> None:
    pytest.importorskip("torch")
    from tools.eval_arrow_grefcoco import _gdino_caption

    assert _gdino_caption("the person on the left") == "the person on the left ."
    assert _gdino_caption("the person on the left .") == "the person on the left ."
    assert _gdino_caption("which dog?") == "which dog? ."
    with pytest.raises(ValueError):
        _gdino_caption("  ")
