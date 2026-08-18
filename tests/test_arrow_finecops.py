from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.arrow_finecops_common import normalize_name, reject_train_or_val_path
from tools.prepare_arrow_finecops import _safe_member, _support_source_id


def test_normalize_name_matches_existing_taxonomy_semantics() -> None:
    assert normalize_name("  Wine-Glass._ ") == "wine glass"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/FineCops/Train-set/data.json",
        "/tmp/expression_all_train_set.json",
        "/tmp/expression_all_val_set.json",
    ],
)
def test_train_and_val_paths_fail_closed(path: str) -> None:
    with pytest.raises(ValueError, match="train/val"):
        reject_train_or_val_path(Path(path))


def test_archive_path_traversal_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsafe archive"):
        _safe_member("../../escape.jpg")
    assert _safe_member("images/123.jpg").name == "123.jpg"


def test_support_source_identity_parser() -> None:
    assert _support_source_id(
        Path("/data/vg_patches/telephone/telephone_2414605_123_0.jpg")
    ) == ("vg_patches", 2414605)
    assert _support_source_id(
        Path("/data/lvis_patches/wine_glass/wine_glass_399349_3_0.jpg")
    ) == ("lvis_patches", 399349)


def _record(
    *,
    annotation_id: int,
    parent: int,
    cluster: int,
    kind: str,
    level: int,
    probability: float,
    raw: float,
    iou: float,
    covered: bool = True,
) -> dict:
    payload = {
        "top1_query_index": 0,
        "top1_box_xyxy": [0.0, 0.0, 1.0, 1.0],
        "top1_iou": iou,
        "raw_confidence": raw,
        "official_probability": probability,
    }
    return {
        "schema": "arrow.finecops.eval_record/v1",
        "sample_id": f"finecops:test:{annotation_id}",
        "finecops_annotation_id": annotation_id,
        "parent_positive_id": parent,
        "cluster_gqa_image_id": cluster,
        "kind": kind,
        "level": level,
        "tuple_type": "1_hop",
        "negative_type": "attribute" if kind != "positive" else None,
        "negative_level": 1 if kind != "positive" else None,
        "active_category": "object",
        "support_covered": covered,
        "eligible_queries": 1,
        "eligible_mask_sha256": "0" * 64,
        "routes": {"b58": dict(payload), "r100_d3": dict(payload), "deployed": dict(payload)},
    }


def test_strict_recall_tie_is_failure() -> None:
    from tools.aggregate_arrow_finecops import _surface_metrics

    rows = [
        _record(
            annotation_id=1,
            parent=1,
            cluster=10,
            kind="positive",
            level=1,
            probability=0.5,
            raw=0.0,
            iou=0.8,
        ),
        _record(
            annotation_id=2,
            parent=1,
            cluster=10,
            kind="text",
            level=1,
            probability=0.5,
            raw=0.0,
            iou=0.0,
        ),
        _record(
            annotation_id=3,
            parent=1,
            cluster=10,
            kind="image",
            level=1,
            probability=0.5,
            raw=0.0,
            iou=0.0,
        ),
    ]
    result = _surface_metrics(
        rows, output_route="deployed", support_matched=False, fixed_threshold=0.0
    )
    assert result["rejection"]["text"]["recall1_strict_tie_fail"] == 0.0
    assert result["rejection"]["image"]["recall1_strict_tie_fail"] == 0.0


def test_binary_metrics_handle_ties_without_sklearn() -> None:
    from tools.aggregate_arrow_finecops import _binary_metrics

    result = _binary_metrics([0.5], [0.5])
    assert result["auroc"] == 0.5
    assert result["aupr"] == 0.5


def test_bootstrap_is_deterministic_and_carries_negative_pairs(monkeypatch) -> None:
    import tools.aggregate_arrow_finecops as aggregate

    monkeypatch.setattr(aggregate, "BOOTSTRAP_ITERATIONS", 32)
    runs = {route: {} for route in ("A", "B", "C")}
    for route in runs:
        for seed in (17, 42, 73):
            rows = []
            for index in range(30):
                parent = index + 1
                level = index % 3 + 1
                positive = _record(
                    annotation_id=parent,
                    parent=parent,
                    cluster=1000 + index,
                    kind="positive",
                    level=level,
                    probability=0.9,
                    raw=1.0,
                    iou=0.8 if route != "C" else 0.2,
                )
                rows.append(positive)
                for offset, kind in ((100, "text"), (200, "image")):
                    rows.append(
                        _record(
                            annotation_id=parent + offset,
                            parent=parent,
                            cluster=1000 + index,
                            kind=kind,
                            level=level,
                            probability=0.1,
                            raw=-1.0,
                            iou=0.0,
                        )
                    )
            runs[route][seed] = rows
    first = aggregate._bootstrap(runs)
    second = aggregate._bootstrap(runs)
    assert first == second
    assert set(first["contrasts"]) == {
        "positive_A_minus_B",
        "positive_B_minus_C",
        "text_A_minus_B",
        "text_B_minus_C",
        "image_A_minus_B",
        "image_B_minus_C",
    }


def test_masked_confidence_max_ignores_invalid_queries() -> None:
    torch = pytest.importorskip("torch")
    from tools.eval_arrow_finecops import _masked_max

    score = torch.tensor([[1.0, 9.0, 2.0]])
    mask = torch.tensor([[True, False, True]])
    assert _masked_max(score, mask).tolist() == [2.0]


def test_gt_iou_accepts_repository_tuple_api() -> None:
    torch = pytest.importorskip("torch")
    from tools.eval_arrow_finecops import _gt_iou

    prediction = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    target = {"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}
    assert _gt_iou(prediction, target, 0) == pytest.approx(1.0, abs=1e-5)


def test_b_route_uses_active_tuple_category_without_changing_caption() -> None:
    pytest.importorskip("torch")
    from tools.eval_arrow_finecops import _apply_route_admission_inputs

    targets = [{"caption": "the camera beside the wall .", "stage_a_caption": "person ."}]
    rows = [{"finecops_active_category": "camera"}]
    _apply_route_admission_inputs("B", targets, rows)
    assert targets == [
        {"caption": "the camera beside the wall .", "stage_a_caption": "camera ."}
    ]


def test_finecops_configs_keep_full_expression_and_gap3() -> None:
    pytest.importorskip("torch")
    from util.slconfig import SLConfig

    for route in ("a", "b", "c"):
        cfg = SLConfig.fromfile(
            f"config/ablations/cfg_arrow_finecops_{route}.py"
        )
        assert cfg.stage_b_arrow_finecops_eval is True
        assert cfg.stage_b_u0_category_gate_max_gap == 3.0
        assert getattr(cfg, "stage_b_u2v5_ref_query_caption_mode", "full") == "full"


def test_actual_exact_mapping_coverage_is_frozen() -> None:
    from tools.prepare_arrow_finecops import (
        DEFAULT_CANONICAL,
        DEFAULT_SUPPORT_CACHE,
        DEFAULT_VG_METADATA,
        _name_to_canonical,
        _select_supports,
        _vg_coco_crosswalk,
    )

    annotation = Path(
        "/media/haoyi/T9/data/FineCops-Ref/v1/raw/benchmark/"
        "test_expression_pos.json"
    )
    required = (annotation, DEFAULT_CANONICAL, DEFAULT_SUPPORT_CACHE, DEFAULT_VG_METADATA)
    if not all(path.is_file() for path in required):
        pytest.skip("local FineCops contract inputs are unavailable")
    positives = json.loads(annotation.read_text(encoding="utf-8"))
    mapping = _name_to_canonical(DEFAULT_CANONICAL)
    gqa_ids = {int(row["image_id"]) for row in positives.values()}
    coco_ids = _vg_coco_crosswalk(DEFAULT_VG_METADATA, gqa_ids)
    needed = {
        mapping[normalize_name(row["tuple"][0][0])]
        for row in positives.values()
        if normalize_name(row["tuple"][0][0]) in mapping
    }
    selected, _ = _select_supports(
        support_cache=DEFAULT_SUPPORT_CACHE,
        needed_class_ids=needed,
        test_gqa_ids=gqa_ids,
        test_coco_ids=coco_ids,
    )
    covered = sum(
        mapping.get(normalize_name(row["tuple"][0][0])) in selected
        for row in positives.values()
    )
    assert covered == 9182
