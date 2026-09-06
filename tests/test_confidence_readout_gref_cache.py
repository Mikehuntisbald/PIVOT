import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import extract_confidence_readout_gref_cache as cache


def tiny_source(monkeypatch):
    rows, images = [], {}
    for image in range(2):
        images[str(image)] = {"remote_path": f"/images/{image}.jpg", "sha256": str(image) * 64,
                              "width": 100, "height": 80,
                              "overlap": {"train_all": image == 0, "val_all": False}}
        for j, kind in enumerate(("positive", "no_target")):
            split = "testA" if image == 0 else "testB"
            rows.append({"sample_id": f"grefcoco:{split}:{image}:{j}", "split": split,
                         "image_id": image, "image_path": images[str(image)]["remote_path"],
                         "image_sha256": images[str(image)]["sha256"], "width": 100, "height": 80,
                         "expression": "the described person", "kind": kind, "level": None,
                         "bbox_xywh": [10., 20., 30., 40.] if kind == "positive" else None,
                         "finecops_train_val_source_disjoint": image == 1})
    monkeypatch.setattr(cache, "EXPECTED", {"records": 4, "images": 2, "positive": 2, "no_target": 2})
    monkeypatch.setattr(cache, "EXPECTED_DISJOINT", {"records": 2, "images": 1, "positive": 1, "no_target": 1})
    monkeypatch.setattr(cache, "EXPECTED_SPLITS", {(s, k): 1 for s in ("testA", "testB") for k in ("positive", "no_target")})
    return rows, images


def test_source_population_order_cardinality_and_disjoint_contract(monkeypatch):
    rows, images = tiny_source(monkeypatch)
    cache.validate_source_rows(rows, images)
    for field, value in (("bbox_xywh", [[1., 2., 3., 4.], [5., 6., 7., 8.]]),
                         ("split", "train"), ("finecops_train_val_source_disjoint", True),
                         ("image_sha256", "f" * 64)):
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError):
            cache.validate_source_rows(bad, images)
    with pytest.raises(ValueError, match="source order"):
        cache.validate_source_rows(list(reversed(rows)), images)
    bad = copy.deepcopy(rows)
    bad[1]["bbox_xywh"] = [1., 2., 3., 4.]
    with pytest.raises(ValueError, match="invented"):
        cache.validate_source_rows(bad, images)


def test_original_four_strided_B32_grouping_not_contiguous_cache_shards():
    rows = [{"sample_id": f"row{i}"} for i in range(20684)]
    groups = cache.evaluation_groups(rows)
    assert len(groups) == 648
    assert groups[0] == [f"row{i}" for i in range(0, 128, 4)]
    assert groups[1] == [f"row{i}" for i in range(128, 256, 4)]
    assert groups[162] == [f"row{i}" for i in range(1, 129, 4)]
    assert [len(g) for g in groups].count(19) == 4
    assert all(len(g) in (19, 32) for g in groups)
    assert set(x for g in groups for x in g) == {r["sample_id"] for r in rows}


def test_seal_requires_all_six_postflights_and_actual_bindings(tmp_path, monkeypatch):
    from tools import seal_confidence_readout_heads as sealer
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    sealer_path = tmp_path / "sealer.py"
    sealer_path.write_text("synthetic")
    postflights, calls = {}, []
    for localizer in cache.LOCALIZERS:
        postflights[localizer] = {}
        for seed in cache.SEEDS:
            path = tmp_path / f"{localizer}_{seed}.json"
            path.write_text("{}")
            postflights[localizer][seed] = cache.bind(path)
    def checked(path, protocol, localizer, seed):
        calls.append((localizer, seed))
        assert protocol == cache.bind(protocol_path)
        return {"arms": [None] * (2 if localizer == cache.MM else 4)}
    monkeypatch.setattr(sealer, "check_postflight", checked)
    gate = {"schema": "arrow.confidence_readout.all_heads_sealed/v1", "status": "complete",
            "trajectories": 18, "study_protocol": cache.bind(protocol_path),
            "metric_selection": False, "sealer": cache.bind(sealer_path), "postflights": postflights}
    path = tmp_path / "all_heads_sealed.json"
    path.write_text(json.dumps(gate))
    assert cache.require_all_heads(protocol_path, path) == cache.bind(path)
    assert len(calls) == 6
    gate["postflights"][cache.MD].pop("73")
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="all three seed"):
        cache.require_all_heads(protocol_path, path)
    gate["trajectories"] = 6
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="eighteen"):
        cache.require_all_heads(protocol_path, path)


def test_bound_json_rebase_and_append_only(tmp_path):
    path = tmp_path / "payload.json"
    cache.write_new(path, {"ok": True})
    entry = cache.bind(path)
    entry["path"] = "/old/host/location/payload.json"
    assert cache.verify(entry, path) == path.resolve()
    with pytest.raises(FileExistsError):
        cache.write_new(path, {"ok": False})
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA drift"):
        cache.verify(entry, path)


def tensor_hook(q):
    torch = pytest.importorskip("torch")
    return SimpleNamespace(query_features=torch.arange(q * 256).reshape(q, 256).to(torch.float16),
                           native_score=torch.full((q,), .2), boxes=torch.full((q, 4), .5),
                           candidate_mask=torch.ones(q, dtype=torch.bool))


def test_build_rows_dynamic_query_count_gt_and_source_fields(monkeypatch):
    torch = pytest.importorskip("torch")
    rows, _ = tiny_source(monkeypatch)
    h = tensor_hook(900)
    # FP16 arange at Q900 would overflow; use a bounded synthetic feature.
    h.query_features.zero_()
    h.native_score[7] = .8
    positive, negative = [cache.build_row(r, h, cache.MM) for r in rows[:2]]
    assert positive["native_selected_index"] == 7
    assert positive["split"] == "gref_testab" and positive["stratum"] == "testA"
    assert positive["parent_positive_id"] is None
    assert positive["gt_boxes"].tolist()[0] == pytest.approx([.25, .5, .3, .5])
    assert negative["gt_boxes"].shape == (0, 4)
    assert negative["kind"] == "no_target"
    assert positive["query_features"].shape == (900, 256)
    h.candidate_mask.zero_()
    with pytest.raises(ValueError, match="mask drift"):
        cache.build_row(rows[0], h, cache.MM)


def test_mdetr_official_tie_selection_and_pixel_geometry(monkeypatch):
    torch = pytest.importorskip("torch")
    rows, _ = tiny_source(monkeypatch)
    h = tensor_hook(100)
    h.image_size, h.native_selected_index = (80, 100), 4
    h.boxes[4, 0] = .7
    cx, cy, w, hh = h.boxes.unbind(-1)
    h.native_boxes_xyxy_abs = torch.stack((cx-.5*w, cy-.5*hh, cx+.5*w, cy+.5*hh), -1) * torch.tensor([100., 80., 100., 80.])
    result = cache.build_row(rows[0], h, cache.MD)
    assert result["native_selected_index"] == 4
    assert result["query_features"].shape == (100, 256)
    h.native_selected_index = 0
    with pytest.raises(ValueError, match="official"):
        cache.build_row(rows[0], h, cache.MD)


def test_mm_legacy_every_query_boxes_mask_selected_and_iou_parity(monkeypatch):
    torch = pytest.importorskip("torch")
    from tools.responsibility_isolation_cache import normalized_cxcywh_iou
    rows, _ = tiny_source(monkeypatch)
    h = tensor_hook(900)
    h.query_features.zero_()
    row = cache.build_row(rows[0], h, cache.MM)
    index = row["native_selected_index"]
    iou = float(normalized_cxcywh_iou(row["boxes"][index:index+1], row["gt_boxes"])[0, 0])
    legacy = {"native_top1_query": index, "native_score": float(row["native_score"][index]),
              "native_box": row["boxes"][index].tolist(), "native_iou": iou, "correct": iou >= .5,
              "boxes_sha256": cache.tensor_sha(row["boxes"]),
              "candidate_mask_sha256": cache.tensor_sha(row["candidate_mask"])}
    cache.check_mm_legacy(row, legacy)
    # Changing a NON-selected query's box must still fail all-query parity.
    row["boxes"][9, 0] = .6
    with pytest.raises(ValueError, match="boxes_sha256"):
        cache.check_mm_legacy(row, legacy)


def test_preflight_non_gref_and_exact_features_before_transfer(tmp_path):
    torch = pytest.importorskip("torch")
    h = tensor_hook(100)
    anchor = {k: getattr(h, k) for k in ("query_features", "native_score", "boxes", "candidate_mask")}
    anchor.update(sample_id="val:fixture", image_path="unused.jpg", caption="a person")
    shard = tmp_path / "val.pt"
    torch.save({"rows": [anchor]}, shard)
    runtime = SimpleNamespace(infer=lambda path, caption: h)
    result = cache.runtime_preflight(runtime, {"shards": [cache.bind(shard)]}, {"grefcoco:testA:1:2"})
    assert all(result["bitwise_parity"].values())
    with pytest.raises(ValueError, match="non-gRef"):
        cache.runtime_preflight(runtime, {"shards": [cache.bind(shard)]}, {"val:fixture"})
    h.query_features[0, 0] = -1.
    with pytest.raises(ValueError, match="parity failed"):
        cache.runtime_preflight(runtime, {"shards": [cache.bind(shard)]}, set())


def test_cached_shard_order_and_source_metadata_resume_validation(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    rows, _ = tiny_source(monkeypatch)
    h = tensor_hook(900)
    h.query_features.zero_()
    record = cache.build_row(rows[0], h, cache.MM)
    binding = {"localizer": cache.MM}
    value = {"schema": cache.SHARD_SCHEMA, "split": cache.SPLIT, "start": 0,
             "binding": binding, "rows": [record]}
    path = tmp_path / "shard.pt"
    torch.save(value, path)
    cache._existing_shard(path, binding, rows[:1], 0, None)
    value["rows"][0]["caption"] = "altered expression"
    torch.save(value, path)
    with pytest.raises(ValueError, match="source-derived field drift"):
        cache._existing_shard(path, binding, rows[:1], 0, None)


def test_no_runtime_import_or_optimizer_during_source_audit():
    source = Path(cache.__file__).read_text()
    assert "torch.optim" not in source and "optimizer.step" not in source
    assert "runtime.infer(Path(source[\"image_path\"]), source[\"expression\"])" in source
    assert "head_evaluation_performed\": False" in source
