import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.mdetr_frozen_runtime import MDETRHookBatch
from tools.extract_mdetr_readout_cache_v2 import build_row
from tools.recover_mdetr_negative_reference_cache import (
    request_for_study, rewrap_positive_payload, rows_fingerprint, audit_source,
)


def request(kind="positive"):
    return SimpleNamespace(sample_id="finecops-train:1",annotation_id=1,image_path=Path("/fixture.jpg"),
        source_image_id="1",cluster_image_id="1",caption="the black cat",kind=kind,parent_positive_id=1,
        level=1,negative_type="order" if kind=="text" else None,negative_level=1 if kind=="text" else None,
        gt_boxes=torch.tensor([[.5,.5,.2,.2]],dtype=torch.float32))


def hook():
    return MDETRHookBatch(torch.zeros(100,256).half(),torch.linspace(1,0,100),torch.ones(100,4)*.2,
        torch.ones(100,dtype=torch.bool),0,(480,640),torch.ones(100,4))


def test_real_parser_negative_reference_is_preserved_but_never_study_gt():
    original=request("text")
    before=original.gt_boxes.clone()
    row=build_row(original,hook(),"0"*64)
    assert row["gt_boxes"].shape==(0,4)
    assert row["annotation_reference_boxes_active"] is False
    assert row["annotation_reference_boxes_role"]=="source_parent_edit_reference_not_study_ground_truth"
    assert torch.equal(row["annotation_reference_boxes"],before)
    assert torch.equal(original.gt_boxes,before)
    assert row["native_selected_index"]==0
    from tools.confidence_readout import native_labels
    labels, _ = native_labels([row], "mdetr_r101_refcoco_ema")
    assert labels[row["sample_id"]] is None


def test_positive_keeps_exact_old_row_contract():
    from tools.extract_mdetr_readout_cache import build_row as old_build
    req=request()
    assert rows_fingerprint([build_row(req,hook(),"0"*64)])==rows_fingerprint([old_build(req,hook(),"0"*64)])


def test_source_semantics_not_inferred_from_empty_bbox():
    req=request("text");req.gt_boxes=torch.empty(0,4)
    with pytest.raises(ValueError,match="reference"):
        request_for_study(req)
    with pytest.raises(ValueError):request_for_study(request("image"))


def test_positive_cache_rewrap_preserves_every_tensor_and_metadata():
    req=request();row=build_row(req,hook(),"0"*64)
    payload={"schema":"arrow.confidence_readout.cache_shard/v1","start":0,"split":"train","binding":{"old":1},"rows":[row]}
    original=rows_fingerprint(payload["rows"])
    new=rewrap_positive_payload(payload,binding={"new":2},expected_requests=[req])
    assert rows_fingerprint(new["rows"])==original
    assert payload["binding"]=={"old":1} and new["binding"]=={"new":2}
    altered=copy.deepcopy(new);altered["rows"][0]["query_features"][0,0]=1
    assert rows_fingerprint(altered["rows"])!=original


def test_recovery_rejects_negative_or_gt_metadata_drift():
    req=request();row=build_row(req,hook(),"0"*64)
    payload={"schema":"arrow.confidence_readout.cache_shard/v1","rows":[row]}
    changed=request();changed.gt_boxes[0,0]=.9
    with pytest.raises(ValueError,match="GT"):rewrap_positive_payload(payload,binding={},expected_requests=[changed])
    row["kind"]="text"
    with pytest.raises(ValueError,match="positive-only"):rewrap_positive_payload(payload,binding={},expected_requests=[req])


def fixture_manifest(tmp_path,monkeypatch):
    import tools.recover_mdetr_negative_reference_cache as audit
    monkeypatch.setattr(audit,"COUNTS",{"train":(1,1),"val":(1,1)})
    data={"images":[{"id":1,"file_name":"1.jpg","width":100,"height":100},
                    {"id":2,"file_name":"1.jpg","width":100,"height":100}],
          "annotations":[{"id":1,"image_id":1,"bbox":[10,10,20,20]},
                         {"id":2,"image_id":2,"bbox":[10,10,20,20],"negative_type":"order","positive_id":1}]}
    annotation=tmp_path/"train.json";annotation.write_text(json.dumps(data))
    index=tmp_path/"index.jsonl";index.write_text('\n'.join(json.dumps({"sample_id":f"finecops-train:{i}","annotation_id":i,"kind":k}) for i,k in ((1,"positive"),(2,"text"))))
    parser=tmp_path/"parser.py";parser.write_text("# sealed parser fixture\n")
    def binding(path):return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest=tmp_path/"manifest.json"
    value={"split":"train","status":"complete","formal":True,"records":2,
           "annotation":binding(annotation),"index":binding(index),"extractor":binding(parser)}
    manifest.write_text(json.dumps(value))
    return manifest,data,annotation,value,binding


def test_all_negative_source_audit_reports_parent_reference(tmp_path,monkeypatch):
    manifest,*_=fixture_manifest(tmp_path,monkeypatch)
    out=audit_source(manifest)
    assert out["counts"]=={"positive":1,"negative_text":1,"reference_equals_parent":1}
    assert out["ambiguous_rows"]==0
    assert out["first_negative_in_source_order"]["gt_boxes_after_adapter_shape"]==[0,4]


def test_all_negative_audit_refuses_changed_reference_or_image(tmp_path,monkeypatch):
    manifest,data,annotation,value,binding=fixture_manifest(tmp_path,monkeypatch)
    data["annotations"][1]["bbox"][0]=11
    annotation.write_text(json.dumps(data));value["annotation"]=binding(annotation);manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="parent edit-reference"):
        audit_source(manifest)
