"""All-head-sealed coverage scoring and record-only analysis; never open FineCops Test."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tools.confidence_coverage import ARMS, SCHEMA, SEEDS, make_readout_heads
from tools.confidence_readout import MMGDINO, GLOBAL, native_selected_index, readout_scores
from tools.train_confidence_coverage_heads import bind,verify,parameter_audit
from tools.train_finecops_bce_l2_heads import _stack_rows,_deterministic_algorithms,_tensor_state_sha256
from tools.evaluate_confidence_readout_cache import load_evaluation_cache,evaluation_groups,row_geometry,write_jsonl
from tools.seal_confidence_readout_heads import publish_json,terminal_for_head
from tools.confidence_coverage_metrics import analyze_coverage


def read(binding):
    verify(binding);return json.loads(Path(binding["path"]).read_text())


def validate_protocol(path):
    p=json.loads(path.read_text())
    if p.get("schema")!=SCHEMA or p.get("new_heads")!=12 or p.get("arms")!=list(ARMS):
        raise ValueError("coverage protocol identity drift")
    for name,b in p["code"].items():
        verify(b)
        if bind(ROOT/name)["sha256"]!=b["sha256"]:raise ValueError("runtime code drift")
    return p


def sealed_panel(path):
    validate_protocol(path);root=path.parent;postflights={};terminals={}
    for seed in SEEDS:
        b=bind(root/"heads/mmgdino_positive"/f"seed{seed}/postflight.json");p=read(b)
        if (p.get("schema")!="arrow.confidence_coverage.training_postflight/v1"
                or p.get("status")!="complete" or p.get("updates_per_head")!=12575
                or p.get("arms")!=list(ARMS) or p.get("seed")!=seed):
            raise ValueError("all twelve endpoints required before any new scoring")
        design=read(p["design"])
        if design["study_protocol"]!=bind(path):raise ValueError("head protocol mismatch")
        if (len(p.get("history",[]))!=5 or not p.get("no_optimizer_skips") or not p.get("no_amp")
                or any(h["amp_skips"]!=0 or h["nonfinite"]!=0 or h["updates"]!=i*2515
                       for i,h in enumerate(p["history"],1))):
            raise ValueError("training update/health audit incomplete")
        terminals[str(seed)]=terminal_for_head(root,MMGDINO,str(seed),Path(b["path"]))
        verify(p["checkpoint"]);postflights[str(seed)]=b
    receipt={"schema":"arrow.confidence_coverage.all_heads_sealed/v1","status":"complete",
        "protocol":bind(path),"trajectories":12,"postflights":postflights,"terminals":terminals}
    target=root/"all_heads_sealed.json"
    if target.exists():
        if json.loads(target.read_text())!=receipt:raise ValueError("endpoint seal drift")
    else:publish_json(target,receipt)
    return receipt


def score(path,surface,device):
    protocol=validate_protocol(path);panel=sealed_panel(path)
    root=path.parent/"evaluation"/surface
    if root.exists():raise FileExistsError("append-only coverage evaluation required")
    cache=protocol["evaluation_caches"][surface];verify(cache)
    references={}
    for seed in SEEDS:
        b=protocol["references"][surface][str(seed)];verify(b)
        rows=[json.loads(l) for l in Path(b["path"]).read_text().splitlines()]
        references[str(seed)]={r["sample_id"]:r for r in rows}
        if len(references[str(seed)])!=len(rows):raise ValueError("duplicate reference identity")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"]!=":4096:8":raise ValueError("determinism drift")
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(device);torch.cuda.set_device(device)
    models={}
    for seed in SEEDS:
        post=read(panel["postflights"][str(seed)])
        ck=torch.load(post["checkpoint"]["path"],map_location="cpu",weights_only=False)
        if ck["updates"]!=12575 or ck["epoch"]!=5 or ck["design"]!=post["design"]:
            raise ValueError("checkpoint endpoint drift")
        models[str(seed)]=make_readout_heads(seed,device,MMGDINO)
        for arm,model in models[str(seed)].items():
            model.load_state_dict(ck["models"][arm],strict=True)
            for task in ("rank","confidence"):
                key="frozen_rank_sha256" if task=="rank" else "confidence_sha256"
                if _tensor_state_sha256(dict(model.named_task_parameters(task)))!=post["ownership"][arm][key]:
                    raise ValueError("endpoint owner tensor SHA drift")
            model.eval().requires_grad_(False)
    rows,manifest=load_evaluation_cache(cache["path"],MMGDINO)
    expected={"finecops_val":{"positive":9426,"text":9029},"gref_full":{"positive":11563,"no_target":9121}}
    if dict(Counter(r["kind"] for r in rows))!=expected[surface]:raise ValueError("evaluation population drift")
    if manifest["model"]["checkpoint"]!=protocol["localizers"][MMGDINO]["checkpoint"]:
        raise ValueError("frozen localizer checkpoint drift")
    if any(set(ref)!=set(r["sample_id"] for r in rows) for ref in references.values()):
        raise ValueError("cache/reference sample-ID surface drift")
    official={}
    if surface=="finecops_val":
        annotation=read(manifest["annotation"])
        official={int(r["id"]):r for r in annotation["annotations"]}
    root.mkdir(parents=True)
    publish_json(root/"design.json",{"protocol":bind(path),"seal":bind(path.parent/"all_heads_sealed.json"),
        "cache":cache,"references":protocol["references"][surface],"surface":surface,
        "detector_forwards":0,"optimizer_updates":0,"threshold_fitting":False})
    output={s:[] for s in references}
    groups=evaluation_groups(rows,manifest,"evaluation")
    with _deterministic_algorithms(),torch.inference_mode():
        for i,group in enumerate(groups):
            f,n,m=_stack_rows(group,device)
            indices=[native_selected_index(r,MMGDINO) for r in group]
            selected=torch.tensor(indices,device=device,dtype=torch.int64)
            geometry=[row_geometry(r,j)[0] for r,j in zip(group,indices)]
            for seed,heads in models.items():
                values={arm:readout_scores(model,f,n,m,selected)[GLOBAL].cpu().tolist() for arm,model in heads.items()}
                for k,row in enumerate(group):
                    ref=references[seed][row["sample_id"]]
                    if any(ref[key]!=value for key,value in geometry[k].items()):
                        raise ValueError("Native boxes/scores/correctness/mask parity failed")
                    record={key:value for key,value in ref.items() if key not in ("scores","readout_diagnostics","schema")}
                    if surface=="finecops_val":
                        record["negative_edit_level"]=official[int(row["annotation_id"])].get("negative_level") if row["kind"]=="text" else None
                    record["schema"]="arrow.confidence_coverage.record/v1"
                    record["scores"]={arm:values[arm][k] for arm in ARMS}
                    record["scores"].update({"paired_l1__"+t:ref["scores"]["global_max__"+t] for t in ("exists","emit")})
                    output[seed].append(record)
            if i%100==0:print(f"[COVERAGE-EVAL] {surface} batches={i+1}/{len(groups)}",flush=True)
    bindings={s:write_jsonl(root/f"seed{s}.jsonl",rs) for s,rs in output.items()}
    publish_json(root/"postflight.json",{"schema":"arrow.confidence_coverage.evaluation_postflight/v1",
        "status":"complete","surface":surface,"protocol":bind(path),"records":bindings,
        "native_parity_with_v6":True,"detector_forwards":0,"optimizer_updates":0,"threshold_fitting":False})


def analyze(path,surface):
    p=validate_protocol(path);sealed_panel(path)
    origin="finecops_val" if surface=="finecops_val" else "gref_full"
    post=read(bind(path.parent/"evaluation"/origin/"postflight.json"))
    if post["status"]!="complete" or post["protocol"]!=bind(path):raise ValueError("unsealed evaluation")
    runs={}
    for seed,b in post["records"].items():
        verify(b);rows=[json.loads(l) for l in Path(b["path"]).read_text().splitlines()]
        if surface=="gref_source_disjoint":rows=[r for r in rows if r["finecops_train_val_source_disjoint"]]
        runs[seed]=rows
    counts={"finecops_val":(18455,3567),"gref_full":(20684,1500),"gref_source_disjoint":(17564,1277)}
    for rows in runs.values():
        if (len(rows),len({r["cluster_id"] for r in rows}))!=counts[surface]:raise ValueError("analysis population drift")
    target=path.parent/"analysis"/(surface+".json")
    if target.exists():raise FileExistsError("append-only bootstrap result")
    result=analyze_coverage(runs,iterations=p["bootstrap"]["iterations"],seed=p["bootstrap"]["seed"],
        progress=lambda i,n:print(f"[COVERAGE-BOOTSTRAP] {surface} {i}/{n}",flush=True))
    result["receipt"]={"protocol":bind(path),"evaluation":bind(path.parent/"evaluation"/origin/"postflight.json"),"surface":surface}
    target.parent.mkdir(exist_ok=True);publish_json(target,result)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("seal","evaluate","analyze"));parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--surface",choices=("finecops_val","gref_full","gref_source_disjoint"));parser.add_argument("--device",default="cuda:3")
    args=parser.parse_args()
    if args.command=="seal":print(json.dumps(sealed_panel(args.protocol)))
    elif args.command=="evaluate":score(args.protocol,args.surface,args.device)
    else:analyze(args.protocol,args.surface)
