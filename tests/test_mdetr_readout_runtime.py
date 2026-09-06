import types
import json

import pytest
import torch

from tools.mdetr_frozen_runtime import (
    MDETRHookBatch, cxcywh_to_xyxy, official_native_index,
    official_resize_shape, require_ema,
)
from tools.extract_mdetr_readout_cache import build_row


def test_official_native_tuple_tie_not_argmax():
    scores = torch.tensor([.7, .7, .2])
    boxes = torch.tensor([[1., 1, 3, 3], [2., 1, 3, 3], [9., 1, 10, 3]])
    assert scores.argmax() == 0
    assert official_native_index(scores, boxes) == 1
    assert official_native_index(scores, boxes, torch.tensor([True, False, True])) == 0


def test_complete_tuple_tie_stable_first():
    assert official_native_index(torch.ones(3), torch.ones(3, 4)) == 0


@pytest.mark.parametrize("mask", [torch.zeros(3, dtype=torch.bool), torch.ones(3), torch.ones(2, dtype=torch.bool)])
def test_invalid_mask_fail_close(mask):
    with pytest.raises(ValueError):
        official_native_index(torch.ones(3), torch.ones(3, 4), mask)


def test_selection_finite():
    with pytest.raises(ValueError):
        official_native_index(torch.tensor([float("nan")]), torch.ones(1, 4))


@pytest.mark.parametrize("size,result", [((640, 480), (800, 1066)), ((480, 640), (1066, 800)), ((1920, 1080), (750, 1333)), ((800, 800), (800, 800))])
def test_official_resize_rounding(size, result):
    assert official_resize_shape(*size) == result


def test_ema_mandatory_even_when_raw_exists():
    with pytest.raises(ValueError, match="mandatory"):
        require_ema({"model": {"weight": torch.ones(1)}})
    state = {"weight": torch.ones(1)}
    assert require_ema({"model_ema": state}) is state
    with pytest.raises(ValueError):
        require_ema({"model_ema": {"weight": torch.tensor([float("nan")])}})


def test_cxcywh_formula_does_not_clip():
    box = torch.tensor([[.1, .2, .4, .6]])
    expected = torch.tensor([[-.1, -.1, .3, .5]])
    torch.testing.assert_close(cxcywh_to_xyxy(box), expected)


def request(kind="positive"):
    return types.SimpleNamespace(sample_id="finecops-train:1", annotation_id=1, image_path="/fixture.jpg",
                                 source_image_id="1", cluster_image_id="1", caption="the black cat", kind=kind,
                                 parent_positive_id=1, level=1, negative_type=None, negative_level=None,
                                 gt_boxes=torch.tensor([[.5, .5, .2, .2]]) if kind == "positive" else torch.empty(0, 4))


def hook():
    return MDETRHookBatch(torch.zeros(100, 256).half(), torch.ones(100).float(), torch.ones(100, 4).float()*.2,
                          torch.ones(100, dtype=torch.bool), 1, (480, 640), torch.ones(100, 4).float())


def test_cache_keeps_selected_id_and_dtype_and_no_target():
    for kind in ("positive", "text"):
        row = build_row(request(kind), hook(), "0"*64)
        assert row["native_selected_index"] == 1
        assert row["image_size"] == [480, 640]
        assert row["query_features"].shape == (100, 256)
        assert row["query_features"].dtype == torch.float16
        assert row["gt_boxes"].numel() == (4 if kind == "positive" else 0)


def test_negative_gt_and_image_negatives_rejected():
    value = request()
    value.kind = "text"
    with pytest.raises(ValueError):
        build_row(value, hook(), "0"*64)
    with pytest.raises(ValueError):
        build_row(request("image"), hook(), "0"*64)


def test_native_selector_agrees_with_training_helper_on_pixel_ties():
    from tools.confidence_readout import native_selected_index
    generator = torch.Generator().manual_seed(11)
    for count in (3, 100):
        boxes = torch.rand(count, 4, generator=generator)
        scores = torch.ones(count)
        mask = torch.ones(count, dtype=torch.bool)
        pixel = cxcywh_to_xyxy(boxes) * torch.tensor([640, 480, 640, 480])
        expected = official_native_index(scores, pixel, mask)
        row = {"boxes": boxes, "native_score": scores, "candidate_mask": mask,
               "image_size": [480, 640], "native_selected_index": expected}
        assert native_selected_index(row, "mdetr_r101_refcoco_ema") == expected


@pytest.mark.parametrize("split", ["test", "testA", "gref", "val"])
def test_unsealed_or_heldout_manifest_refused_before_parser_import(tmp_path, split):
    from tools.extract_mdetr_readout_cache import source_requests
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"split": split, "status": "complete", "records": 27926, "formal": True}))
    with pytest.raises(ValueError, match="train/val"):
        source_requests(manifest)


def test_completed_worker_and_manifest_receipts_are_idempotent_not_overwritten(tmp_path):
    from tools.extract_mdetr_readout_cache import write_json_idempotent
    for name in ("worker_00.json", "manifest.json"):
        path = tmp_path / name
        value = {"status": "complete", "records": 4, "binding": {"sha256": "a"*64}}
        write_json_idempotent(path, value)
        original, modified = path.read_bytes(), path.stat().st_mtime_ns
        write_json_idempotent(path, value)
        assert path.read_bytes() == original and path.stat().st_mtime_ns == modified
        with pytest.raises(ValueError, match="refuse overwrite"):
            write_json_idempotent(path, {**value, "records": 5})
        assert path.read_bytes() == original
