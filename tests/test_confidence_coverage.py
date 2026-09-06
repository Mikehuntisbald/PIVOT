"""Coverage stream, ownership, frozen-loader parity and paired risk tests."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from tools.confidence_coverage_metrics import CELLS,analyze_coverage


def fixture():
    runs={}
    for seed in ("17","42","73"):
        rows=[]
        for image in range(4):
            for state,correct,scores in (("C",True,[.8,.7,.81,.9]),("W",False,[.6,.3,.61,.2]),("N",None,[.4,.5,.41,.3])):
                rows.append({"sample_id":f"{image}:{state}","cluster_id":str(image),
                    "stratum":"testA" if image<2 else "testB","kind":"positive" if state!="N" else "no_target",
                    "correct":correct,"level":1 if state!="N" else None,"parent_positive_id":None,
                    "native_score":.6 if state!="N" else .3,"scores":dict(zip(CELLS,scores))})
        runs[seed]=rows
    return runs


def test_bootstrap_deterministic_and_interaction_identity():
    a=analyze_coverage(fixture(),iterations=12,conditionals=False)
    assert a==analyze_coverage(fixture(),iterations=12,conditionals=False)
    for m,v in a["effects"]["interaction"].items():
        if v["mean"] is not None:
            assert v["mean"]==pytest.approx(a["effects"]["D_emit"][m]["mean"]-a["effects"]["D_exists"][m]["mean"])
    assert a["bootstrap"]["unit"]=="image_cluster"
    assert a["bootstrap"]["strata"]=={"testA":2,"testB":2}
    assert a["bootstrap"]["q05_recomputed_each_draw"]
    assert a["max_state_identity_error"]<1e-12
    assert set(a["matched_cells"])==set(CELLS)


def test_missing_seed_identity_or_train_rejected():
    for change in ("seed","identity","train"):
        runs=fixture()
        if change=="seed":del runs["17"]
        elif change=="identity":runs["17"][0]["cluster_id"]="alien"
        else:runs["17"][0]["split"]="train"
        with pytest.raises(ValueError):analyze_coverage(runs,iterations=2,conditionals=False)


def test_negative_interaction_does_not_imply_emit_repair():
    runs=fixture()
    for rows in runs.values():
        for r in rows:
            r["scores"][CELLS[3]]=r["scores"][CELLS[1]]
            r["scores"][CELLS[2]]=-r["scores"][CELLS[0]]
    a=analyze_coverage(runs,iterations=5,conditionals=False)
    assert a["effects"]["D_emit"]["mixed_augrc"]["mean"]==0
    assert a["effects"]["interaction"]["mixed_augrc"]["mean"]<0


def test_equal_improvement_has_zero_interaction():
    runs=fixture()
    for rows in runs.values():
        for r in rows:
            r["scores"][CELLS[1]]=r["scores"][CELLS[0]]
            r["scores"][CELLS[2]]=r["scores"][CELLS[3]]
    a=analyze_coverage(runs,iterations=3,conditionals=False)
    assert a["effects"]["interaction"]["mixed_augrc"]["mean"]==0


def test_stream_balanced_total_budget_and_negative_schedule():
    pytest.importorskip("torch")
    from tools.confidence_coverage import positive_stream,epoch_batches
    from tools.train_finecops_bce_l2_heads import _epoch_schedule
    for size in (54015,83341):
        s=positive_stream(size,402255,17)
        assert np.array_equal(s,positive_stream(size,402255,17))
        count=np.bincount(s,minlength=size)
        assert count.min()==402255//size and count.max()==402255//size+1
        assert len(np.unique(s[:size]))==size
    for epoch in range(1,6):
        before,receipt=_epoch_schedule(seed=42,epoch=epoch,rank_count=83341,confidence_count=80451)
        batches,after=epoch_batches(42,epoch)
        assert receipt==after
        assert all(np.array_equal(a,b) for a,b in zip(batches,[i for t,i in before if t=="confidence"]))
        assert len(batches)==2515 and len(batches[-1])==3


def test_factory_identical_initialization_independent_owners():
    torch=pytest.importorskip("torch")
    from tools.confidence_coverage import make_readout_heads
    from tools.finecops_fixed_rank_targets import make_heads
    old=make_heads(17,"cpu");state=torch.get_rng_state().clone()
    heads=make_readout_heads(17,"cpu","mmgdino_positive")
    assert torch.equal(state,torch.get_rng_state())
    owners=[]
    for model in heads.values():
        assert all(torch.equal(v,old["exists"].state_dict()[k]) for k,v in model.state_dict().items())
        params=[p for p in model.parameters() if p.requires_grad]
        assert len(params)==8 and sum(p.numel() for p in params)==50179
        owners.append({id(p) for p in params})
    assert all(not a&b for i,a in enumerate(owners) for b in owners[i+1:])


def test_mmap_loader_matches_legacy_bytes(tmp_path):
    torch=pytest.importorskip("torch")
    from tools.confidence_coverage import load_readout_cache
    from tools.train_finecops_bce_l2_heads import (load_cache,CACHE_ROW_SCHEMA,CACHE_SHARD_SCHEMA,
        CACHE_MANIFEST_SCHEMA,file_sha256)
    row={"schema":CACHE_ROW_SCHEMA,"query_features":torch.randn(900,256).half(),
        "native_score":torch.rand(900),"boxes":torch.rand(900,4),"candidate_mask":torch.ones(900,dtype=torch.bool)}
    shard=tmp_path/"shard.pt";torch.save({"schema":CACHE_SHARD_SCHEMA,"split":"train","start":0,"rows":[row]},shard)
    manifest={"schema":CACHE_MANIFEST_SCHEMA,"status":"complete","formal":True,"split":"train","records":1,
        "shards":[{"path":str(shard),"sha256":file_sha256(shard),"rows":1}]}
    path=tmp_path/"manifest.json";path.write_text(json.dumps(manifest))
    a,_=load_cache(path,split="train");b,_=load_readout_cache(path,split="train",localizer="mmgdino_positive")
    assert all(torch.equal(a[0][k],b[0][k]) for k in ("query_features","native_score","boxes","candidate_mask"))
    manifest["shards"][0]["sha256"]="0"*64;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):load_readout_cache(path,split="train",localizer="mmgdino_positive")


def test_cuda_u1_four_owners_keep_native_route():
    torch=pytest.importorskip("torch")
    if not torch.cuda.is_available():pytest.skip("GPU smoke runs in the remote training environment")
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    from tools.confidence_coverage import make_readout_heads
    from tools.train_confidence_coverage_heads import train_step,parameter_audit
    from tools.train_finecops_bce_l2_heads import _deterministic_algorithms
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:3")
    heads=make_readout_heads(17,device,"mmgdino_positive")
    f=torch.randn(64,900,256,device=device,dtype=torch.float16)
    n=torch.rand(64,900,device=device);m=torch.ones(64,900,device=device,dtype=torch.bool);j=n.argmax(1)
    correct=torch.arange(32,device=device)%2==0
    with _deterministic_algorithms():
        for arm,model in heads.items():
            rank={k:p.detach().cpu().clone() for k,p in model.named_task_parameters("rank")}
            opt=torch.optim.AdamW(model.task_parameters("confidence"),lr=1e-4,weight_decay=0,foreach=False)
            loss,grad=train_step(model,opt,f,n,m,j,correct,target=arm.split("__")[1],
                                 readout="global_max",positive_count=32)
            assert np.isfinite(loss) and np.isfinite(grad)
            parameter_audit(model,opt,rank)
