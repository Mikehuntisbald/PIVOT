#!/usr/bin/env python3
"""Append-only Figure 1 revision showing the three paired training-seed effects."""
import argparse
import json
from pathlib import Path
import build_evidence_v7_assets as v7

PAPER=v7.PAPER


def seed_effects(data):
    md=data["finecops_val"]["localizers"][v7.MD]
    values={seed:{"inference_only":100*(r["global_max__emit__eval_selected"]["mixed_augrc"]-r[v7.GY]["mixed_augrc"]),
                  "matched_retraining":100*(r[v7.SY]["mixed_augrc"]-r[v7.GY]["mixed_augrc"])}
            for seed,r in md["per_seed"].items()}
    for name,key in (("inference_only",v7.CROSS),("matched_retraining","D_emit")):
        if abs(sum(r[name] for r in values.values())/3-100*md["effects"][key]["mixed_augrc"]["mean"])>1e-10:
            raise ValueError("paired seed effects do not reproduce sealed mean")
    return {"schema":"arrow.paper.readout_seed_display/v1","unit":"AUGRC points",
        "reference":"same-seed G-trained/G-deployed emission head","per_seed":values,
        "inference_only_sample_sd":100*md["effects"][v7.CROSS]["mixed_augrc"]["sample_sd"],
        "new_bootstrap":False,"image_ci_is_not_training_seed_uncertainty":True}


def figure(data,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "svg.fonttype":"none","svg.hashsalt":"evidence-v7-seeds"})
    fc=data["finecops_val"]["localizers"]
    fig,axes=plt.subplots(1,3,figsize=(9.1,3.6),gridspec_kw={"width_ratios":[1,1.35,1]})
    fig.subplots_adjust(left=.12,right=.985,bottom=.27,top=.70,wspace=.70)
    fig.text(.02,.97,"Supervising the right event is not enough",fontsize=16,weight="bold",va="top")
    for ax,title,labels,rows,xlabel in (
        (axes[0],"(a) Same labels, opposite risk",["MM-GDINO","MDETR"],
         [fc[l]["effects"][v7.TARGET]["mixed_augrc"] for l in (v7.MM,v7.MD)],"Emission − existence\nAUGRC change ×100"),
        (axes[2],"(c) Locate the C/N loss",["All positive\nlevels","L1 positives"],
         [fc[v7.MM]["effects"][v7.TARGET][k] for k in ("correct_vs_no_target_auroc","difficulty_cn_level1")],
         "MM-GDINO C/N AUROC\nChange ×100")):
        ax.axvline(0,color="#999999",ls="--",lw=.9)
        for y,r,color in zip([1,0],rows,["#236b91","#c8742f"]):
            lo,hi=[100*x for x in r["ci95"]];mean=100*r["mean"]
            ax.plot([lo,hi],[y,y],color=color,lw=2);ax.plot(mean,y,"o",color=color,ms=5)
            ax.annotate(f"{mean:+.3f}",(mean,y),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9,color=color)
        ax.set_yticks([1,0],labels);ax.set_ylim(-.5,1.55);ax.set_title(title,fontsize=9.5,pad=15,weight="bold")
        ax.set_xlabel(xlabel,fontsize=8.4);ax.margins(x=.22)
        ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
        ax.tick_params(axis="both",labelsize=8)
    ax=axes[1];values=seed_effects(data)["per_seed"]
    ax.axhline(0,color="#999999",ls="--",lw=.9)
    for seed,color,marker in zip(("17","42","73"),("#8456a2","#26867b","#c8742f"),("o","s","^")):
        ys=[values[seed][k] for k in ("inference_only","matched_retraining")]
        ax.plot([0,1],ys,color=color,marker=marker,lw=1.1,ms=5,label="Seed "+seed,alpha=.9)
    means=[sum(v[k] for v in values.values())/3 for k in ("inference_only","matched_retraining")]
    ax.scatter([-.10,1.10],means,marker="D",s=30,color="#20242a",label="Mean",zorder=5)
    ax.annotate("Seed 17: +4.726",(0,values["17"]["inference_only"]),xytext=(4,7),
                textcoords="offset points",fontsize=7.6,color="#8456a2")
    ax.set_xlim(-.25,1.25);ax.set_ylim(-.75,5.5)
    ax.set_xticks([0,1],["Inference\nG → S","Train +\ndeploy S"],fontsize=8)
    ax.set_ylabel("AUGRC change ×100",fontsize=8);ax.tick_params(axis="y",labelsize=8)
    ax.set_title("(b) Paired seed effects",fontsize=9.5,pad=15,weight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper right",bbox_to_anchor=(1.25,.84),fontsize=7,frameon=False,handlelength=1.3)
    fig.text(.02,.09,"FineCops val • Fixed Native boxes • (a,c): mean + image-cluster 95% CI; (b): paired seeds + mean",fontsize=8.2)
    fig.text(.02,.035,"The inference-only mean is driven by seed 17; image intervals do not measure training-seed variability.",fontsize=8.2,color="#4e555b")
    fig.savefig(out/"figure1_evidence.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(out/"figure1_evidence.svg",metadata={"Date":None});plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--check",action="store_true")
    args=p.parse_args();data,sources=v7.load_sources();facts=seed_effects(data)
    out=PAPER/"generated/evidence_v7_seed_r1"
    if args.check:
        r=json.loads((out/"receipt.json").read_text())
        assert r["generator"]==v7.bind(__file__) and r["sources"]==sources
        assert json.loads((out/"seed_effects.json").read_text())==facts
        for name,digest in r["outputs"].items():assert v7.old.sha(out/name)==digest
        print("paired seed display and unchanged sealed results verified");return
    if out.exists():raise FileExistsError("append-only figure directory exists")
    out.mkdir(parents=True);figure(data,out)
    (out/"seed_effects.json").write_text(json.dumps(facts,indent=2,sort_keys=True)+"\n")
    (out/"receipt.json").write_text(json.dumps({"schema":"arrow.paper.evidence_v7_seed_assets/v1",
        "generator":v7.bind(__file__),"sources":sources,"outputs":{p.name:v7.old.sha(p) for p in out.iterdir()},
        "new_model_scoring":False,"new_bootstrap":False},indent=2,sort_keys=True)+"\n")

if __name__=="__main__":main()
