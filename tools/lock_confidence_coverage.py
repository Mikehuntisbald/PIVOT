"""Seal the single-MM coverage intervention before any new head optimization."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.confidence_coverage import SCHEMA, ARMS, SEEDS
from tools.train_confidence_coverage_heads import RECIPE, REQUIRED_CODE, bind, verify
from tools.prepare_confidence_readout_analysis import CODE as METRICS_CODE
from tools.seal_confidence_readout_heads import publish_json

CODE=tuple(dict.fromkeys((*REQUIRED_CODE,*METRICS_CODE,
    "tools/confidence_coverage_metrics.py","tools/lock_confidence_coverage.py",
    "tools/run_confidence_coverage_stage.py","tools/evaluate_confidence_coverage.py",
    "tools/evaluate_confidence_readout_cache.py","tools/seal_confidence_readout_heads.py")))


def run(parent,output):
    if output.exists(): raise FileExistsError("append-only new coverage directory required")
    p=json.loads(parent.read_text()); loc=p["localizers"]["mmgdino_positive"]
    for b in (loc["train_cache"],loc["val_cache"],loc["checkpoint"]): verify(b)
    manifest=json.loads(Path(loc["train_cache"]["path"]).read_text())
    verify(manifest["index"]);verify(manifest["annotation"])
    rows=[json.loads(l) for l in Path(manifest["index"]["path"]).read_text().splitlines()]
    positive={r["annotation_id"]:r for r in rows if r["kind"]=="positive"}
    negative=[r for r in rows if r["kind"]=="text"]
    counts=Counter(r["level"] for r in positive.values())
    if counts!={1:54015,2:25282,3:4044} or len(negative)!=80451:
        raise ValueError("training population drift")
    parents={r["parent_positive_id"] for r in negative}
    if len(parents)!=43979 or any(positive[i]["level"]!=1 for i in parents):
        raise ValueError("paired-L1 parent audit drift")
    references={};caches={"finecops_val":loc["val_cache"]}
    for surface in ("finecops_val","gref_full"):
        inp=json.loads((parent.parent/"analysis_inputs"/(surface+".json")).read_text())
        references[surface]=inp["runs"]["mmgdino_positive"]
        for b in references[surface].values():verify(b)
    ed=json.loads((parent.parent/"evaluation/mmgdino_positive/gref_full/evaluation_lock.json").read_text())
    caches["gref_full"]=ed["cache"];verify(ed["cache"])
    protocol={"schema":SCHEMA,"parent":bind(parent),"seeds":list(SEEDS),"new_heads":12,
        "localizers":{"mmgdino_positive":{k:loc[k] for k in ("checkpoint","train_cache","val_cache")}},
        "training":RECIPE,"arms":list(ARMS),"readout":"global_max",
        "positive_sampling":{"pool_order":"sample_id lexical","cycle":"uniform permutation without replacement; continuous across five negative epochs",
            "rng":"PCG64 SeedSequence([20260912, training_seed]); independent from legacy schedule",
            "positive_presentations_per_head":402255,"negative_presentations_per_head":402255,
            "difficulty_counts":dict(counts),"paired_reference_unique_L1":43979,
            "loss_pairing":"none; source-balanced BCE retained; batch co-occurrence is not a semantic pair"},
        "code":{k:bind(ROOT/k) for k in CODE},"references":references,"evaluation_caches":caches,
        "evaluation":{"surfaces":["finecops_val","gref_full","gref_source_disjoint"],
            "all_12_heads_before_scoring":True,"finecops_test":False,"new_detector_traversals":0,
            "checkpoint_selection":False,"threshold_fitting":False,"negative_image":False,"multi_target":False},
        "bootstrap":{"iterations":5000,"seed":20260912,"rng":"PCG64","unit":"image_cluster",
            "gref_strata":["testA","testB"],"all_arms_and_seeds_same_draw":True,"q05_each_replicate":True},
        "primary":{"metric":"mixed_augrc","contrasts":["D_emit","D_exists","interaction","all_emit_minus_exists"],
            "mechanism":["C-W","C-N","W-N","difficulty C-N","residual full-coverage target gap"],
            "acceptance":"all seeds and surfaces regardless of winners; effect size and exploratory intervals",
            "scope":"positive supervision population intervention; not pure difficulty-label causality"}}
    output.mkdir(parents=True)
    publish_json(output/"protocol.json",protocol)
    print(json.dumps(bind(output/"protocol.json")),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.parent,args.output)
