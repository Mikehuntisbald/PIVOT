#!/usr/bin/env python3
"""Publication assets from completed coverage/readout studies, never new metrics."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import build_evidence_v7_assets as v7
import build_evidence_v7_seed_assets as seed_display

old=v7.old; PAPER=old.PAPER; ROOT=PAPER.parent
SOURCE=PAPER/"data/coverage_v1"
CELLS=("l1_uniform__exists","l1_uniform__emit","all_uniform__exists","all_uniform__emit")
SURFACES=("finecops_val","gref_source_disjoint","gref_full")
SUR=old.SUR
ARM={"native":"Native", "paired_l1__exists":"Paired-L1 E", "paired_l1__emit":"Paired-L1 Y",
     CELLS[0]:"L1-uniform E",CELLS[1]:"L1-uniform Y",CELLS[2]:"Full-uniform E",CELLS[3]:"Full-uniform Y"}
METRICS=("mixed_augrc","correctness_auroc","correct_vs_no_target_auroc",
         "wrong_positive_vs_no_target_auroc","existence_auroc","mixed_aurc","diagnostic_fpr95")
EFFECT={"D_emit":r"Full$-$L1, Y", "D_exists":r"Full$-$L1, E", "interaction":"Interaction",
        "l1_emit_minus_exists":r"L1: Y$-$E","all_emit_minus_exists":r"Full: Y$-$E"}


def bind(path):return {"path":str(Path(path).resolve()),"sha256":old.sha(path)}


def validate(data,surface):
    if data.get("schema")!="arrow.confidence_coverage.metrics/v1" or data.get("matched_cells")!=list(CELLS):
        raise ValueError("coverage matrix/schema drift")
    expected={"iterations":5000,"seed":20260912,"rng":"PCG64","unit":"image_cluster",
              "same_draw_all_arms_and_seeds":True,"q05_recomputed_each_draw":True,"fixed_threshold_fit":False}
    if any(data["bootstrap"].get(k)!=v for k,v in expected.items()):raise ValueError("bootstrap contract drift")
    images,positive,negative=old.COUNTS[surface];pop=data["population"]
    if (pop["images"],pop["C"]+pop["W"],pop["N"],pop["records"])!=(images,positive,negative,positive+negative):
        raise ValueError("complete evaluation population required")
    if set(data["per_seed"])!=set(old.SEEDS) or data["max_state_identity_error"]>2e-12:
        raise ValueError("missing seeds or failed state identity")
    for a in ARM:
        for m in METRICS:
            s=data["summary"][a][m]
            if s["undefined_replicates"] or s["ci95"] is None or not all(map(old.finite,[s["mean"],s["sample_sd"],*s["ci95"]])):
                raise ValueError("missing endpoint estimate")
            values=[data["per_seed"][seed][a][m] for seed in old.SEEDS]
            if abs(sum(values)/3-s["mean"])>2e-12:raise ValueError("seed/mean drift")
    for name in EFFECT:
        e=data["effects"][name]["mixed_augrc"]
        if e["ci95"] is None or e["undefined_replicates"]:raise ValueError("missing core contrast CI")
    e=data["effects"]
    if abs(e["interaction"]["mixed_augrc"]["mean"]-e["D_emit"]["mixed_augrc"]["mean"]+e["D_exists"]["mixed_augrc"]["mean"])>2e-12:
        raise ValueError("coverage interaction algebra drift")


def load_sources():
    previous,bindings=v7.load_sources()
    completion=json.loads((SOURCE/"completion.json").read_text())
    if completion.get("status")!="complete" or completion.get("new_heads")!=12:
        raise ValueError("all twelve heads and three analyses must finish")
    if completion["protocol"]["sha256"]!=old.sha(SOURCE/"protocol.json"):
        raise ValueError("protocol SHA drift")
    protocol=json.loads((SOURCE/"protocol.json").read_text())
    for name,b in protocol["code"].items():
        if old.sha(ROOT/name)!=b["sha256"]:raise ValueError("sealed runtime code drift: "+name)
    terminal=json.loads((SOURCE/"training_terminal.json").read_text())
    if old.sha(SOURCE/"training_terminal.json")!=completion["training_terminal"]["sha256"]:
        raise ValueError("training terminal drift")
    seal=json.loads((SOURCE/"all_heads_sealed.json").read_text())
    for seed in old.SEEDS:
        post=SOURCE/f"seed{seed}/postflight.json"
        if old.sha(post)!=seal["postflights"][seed]["sha256"] or terminal["terminal"][seed]["returncode"]!=0:
            raise ValueError("seed endpoint is not successfully sealed")
        p=json.loads(post.read_text())
        if p["updates_per_head"]!=12575 or p["arms"]!=list(CELLS):raise ValueError("training matrix drift")
        for h in p["history"]:
            if h["amp_skips"] or h["nonfinite"]:raise ValueError("failed training health")
    data={}
    for surface in SURFACES:
        item=completion["analyses"][surface];path=SOURCE/"analysis"/(surface+".json")
        if item["returncode"]!=0 or old.sha(path)!=item["result"]["sha256"]:
            raise ValueError("analysis completion/SHA drift")
        d=data[surface]=json.loads(path.read_text());validate(d,surface)
        for a,b in (("native","native"),("paired_l1__exists",v7.GE),("paired_l1__emit",v7.GY)):
            for m in METRICS:
                if abs(d["summary"][a][m]["mean"]-previous[surface]["localizers"][v7.MM]["summary"][b][m]["mean"])>2e-12:
                    raise ValueError("reused paired-L1/Native point parity drift")
        bindings["coverage_"+surface]=bind(path)
    bindings.update(coverage_completion=bind(SOURCE/"completion.json"),coverage_protocol=bind(SOURCE/"protocol.json"),
                    readout_seed_display=bind(PAPER/"generated/evidence_v7_seed_r1/receipt.json"))
    return previous,data,bindings


def decompositions(data):
    result={}
    for surface,d in data.items():
        n=d["population"]["records"];pc=d["population"]["C"]/n;pw=d["population"]["W"]/n;pn=d["population"]["N"]/n
        result[surface]={}
        for name in EFFECT:
            e=d["effects"][name]
            cw=-pc*pw*e["correctness_auroc"]["mean"]
            cn=-pc*pn*e["correct_vs_no_target_auroc"]["mean"]
            if abs(cw+cn-e["mixed_augrc"]["mean"])>2e-12:raise ValueError("risk explanation algebra failed")
            result[surface][name]={"cw_pair_contribution":cw,"cn_pair_contribution":cn,"sum":cw+cn,
                                  "role":"exact point decomposition; not a new interval or causal fraction"}
    return result


def main_tables(previous,data):
    rows=[]
    for loc in old.LOCALIZERS:
        d=previous["finecops_val"]["localizers"][loc]
        for readout,e,y,effect in (("G",v7.GE,v7.GY,"global_emit_minus_exists"),("S",v7.SE,v7.SY,"selected_emit_minus_exists")):
            rows.append(" & ".join([old.LOC[loc],readout,*[old.point_sd(d["summary"][a]["mixed_augrc"]) for a in (e,y)],
                *[old.estimate(d["effects"][effect][m],True,True) for m in ("mixed_augrc","correctness_auroc","correct_vs_no_target_auroc")]])+r"\\")
    a=old.table("Target and readout under paired-L1 supervision on FineCops validation. "
        "Risk is mixed AUGRC $\\times100$ (mean, sample SD); target effects are Y$-$E with paired 95\\% intervals. "
        "Lower risk and higher pairwise AUROC are better. Each confidence route preserves its localizer's Native boxes.",
        "tab:initial_readout","@{}llccccc@{}",r"Localizer & Read & Risk E & Risk Y & $\Delta$ risk & $\Delta U_{CW}$ & $\Delta U_{CN}$",rows)
    rows=[]
    for surface in SURFACES:
        d=data[surface]
        rows.append(" & ".join([SUR[surface],*[old.point_sd(d["summary"][a]["mixed_augrc"]) for a in CELLS],
            *[old.estimate(d["effects"][e]["mixed_augrc"],True,True) for e in ("D_emit","D_exists","interaction")]])+r"\\")
    b=old.table("Positive supervision coverage changes target preference with fixed MM-GDINO boxes, Global-max readout and training budget. "
        "The four risk cells show mean (sample SD), and differences show paired 95\\% intervals, all $\\times100$. "
        "$D_Y=R_{\\mathrm{Full},Y}-R_{\\mathrm{L1},Y}$, $D_E=R_{\\mathrm{Full},E}-R_{\\mathrm{L1},E}$, $I=D_Y-D_E$. "
        "gRef uses frozen heads; its two populations overlap.","tab:coverage_matrix","@{}lccccccc@{}",
        r"Population & L1 E & L1 Y & Full E & Full Y & $D_Y$ & $D_E$ & $I$",rows)
    rows=[]
    for surface in (SURFACES[0],SURFACES[1]):
        for target,effect in (("Y","D_emit"),("E","D_exists")):
            d=data[surface]["effects"][effect]
            rows.append(" & ".join([SUR[surface],target,*[old.estimate(d[m],True,True) for m in
                ("correctness_auroc","correct_vs_no_target_auroc","wrong_positive_vs_no_target_auroc","existence_auroc","mixed_augrc")]])+r"\\")
    c=old.table("Which comparisons change when supervision expands from L1 to Full? "
        "Full$-$L1 effects $\\times100$ [paired 95\\% CI]. $C/W$ and $C/N$ determine complete risk; "
        "$W/N$ compares two failures but contributes to existence AUROC. The target-specific absolute changes "
        "separate improvement in Y from degradation in E.","tab:coverage_states","@{}llccccc@{}",
        r"Population & Target & $\Delta U_{CW}$ & $\Delta U_{CN}$ & $\Delta U_{WN}$ & $\Delta\mathrm{AUC}_E$ & $\Delta$ risk",rows)
    return {"table_readout.tex":a,"table_coverage.tex":b,"table_states.tex":c}


def supplement_tables(data):
    texts={}
    for surface in SURFACES:
        d=data[surface];rows=[]
        for arm in ARM:
            rows.append(" & ".join([ARM[arm],*[old.point_sd(d["summary"][arm][m]) for m in METRICS]])+r"\\")
        texts["supp_endpoints_"+surface+".tex"]=old.table(SUR[surface]+": complete coverage-block endpoints, mean (sample SD), all $\\times100$. "
            "Paired-L1 is the reused reference, not the new L1-uniform control. FPR95 is diagnostic; no deployment threshold is fitted.",
            "tab:coverage_endpoints_"+surface,"@{}lccccccc@{}",
            r"Score & AUGRC & $U_{CW}$ & $U_{CN}$ & $U_{WN}$ & $\mathrm{AUC}_E$ & AURC & FPR95",rows,small=False)
        rows=[]
        for arm in CELLS:
            for seed in old.SEEDS:
                rows.append(" & ".join([ARM[arm],seed,*[old.number(d["per_seed"][seed][arm][m]) for m in METRICS]])+r"\\")
        texts["supp_seeds_"+surface+".tex"]=old.table(SUR[surface]+": every new head seed; raw endpoints $\\times100$.",
            "tab:coverage_seeds_"+surface,"@{}llccccccc@{}",
            r"Score & Seed & AUGRC & $U_{CW}$ & $U_{CN}$ & $U_{WN}$ & $\mathrm{AUC}_E$ & AURC & FPR95",rows,small=False)
        rows=[]
        for name in EFFECT:
            rows.append(" & ".join([EFFECT[name],*[old.estimate(d["effects"][name][m],True,True) for m in METRICS]])+r"\\")
        texts["supp_effects_"+surface+".tex"]=old.table(SUR[surface]+": coverage/target contrasts with paired 95\\% intervals, $\\times100$.",
            "tab:coverage_effects_"+surface,"@{}lccccccc@{}",
            r"Contrast & AUGRC & $U_{CW}$ & $U_{CN}$ & $U_{WN}$ & $\mathrm{AUC}_E$ & AURC & FPR95",rows,small=False)
    rows=[]
    for name in EFFECT:
        e=data["finecops_val"]["effects"][name]
        rows.append(" & ".join([EFFECT[name],*[old.estimate(e[m],True,True) for m in
            ("correct_vs_no_target_auroc","difficulty_cn_level1","difficulty_cn_within_level_contribution","difficulty_cn_cross_level_contribution")]])+r"\\")
    texts["supp_cn_difficulty.tex"]=old.table("FineCops C/N changes: full AUROC, L1-conditional AUROC and weighted within/cross-level contributions. "
        "The last two columns sum to the first, not to the conditional column. All N parents are L1; cross-level means L2/L3 C against those N. "
        "Conditional L2/L3 same-level C/N is undefined, not zero.","tab:coverage_cn_difficulty","@{}lcccc@{}",
        r"Contrast & Full $C/N$ & L1 $C/N$ & Within contribution & Cross contribution",rows,small=False)
    rows=[]
    for surface in SURFACES:
        for name,label in (("l1_emit_minus_exists","L1"),("all_emit_minus_exists","Full")):
            r=data[surface]["augrc_crossovers"][name];v=r["point"]["prior"];ci=r["ci95"]
            point="none" if v is None else f"{v:.3f}"
            interval="--" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            rows.append(f"{SUR[surface]} & {label} & {point} & {interval} & {r['bootstrap_status_counts'].get('interior',0)}/5,000"+r"\\")
    texts["supp_crossovers.tex"]=old.table("Y$-$E AUGRC crossover in no-target prior. Class-conditional requests remain fixed. "
        "An unconditional root interval is given only when all 5,000 replicates have an interior root; missing roots are not imputed. "
        "A mean curve without a root does not imply a uniform confidence guarantee.","tab:coverage_roots","@{}llccc@{}",
        r"Population & Supervision & Mean root & 95\% CI & Interior draws",rows,small=False)
    return texts


def macros(data):
    names={}
    for surface,prefix in (("finecops_val","CovFC"),("gref_source_disjoint","CovGR"),("gref_full","CovFull")):
        for effect,suffix in (("D_emit","DY"),("D_exists","DE"),("interaction","I"),("l1_emit_minus_exists","LoneGap"),("all_emit_minus_exists","FullGap")):
            names[prefix+suffix]=old.estimate(data[surface]["effects"][effect]["mixed_augrc"],True)
    return "\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in names.items())+"\n"


def figures(previous,data,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"coverage-v8"})
    fig,axs=plt.subplots(1,3,figsize=(9.1,3.65),gridspec_kw={"width_ratios":[1,1.25,1.35]})
    fig.subplots_adjust(left=.11,right=.98,bottom=.28,top=.70,wspace=.75)
    fig.text(.02,.97,"What does confidence learn to trust?",fontsize=16,weight="bold",va="top")
    ax=axs[0];ax.axvline(0,color=".6",ls="--",lw=.8)
    for y,loc,col in ((1,v7.MM,"#236b91"),(0,v7.MD,"#c8742f")):
        r=previous["finecops_val"]["localizers"][loc]["effects"]["global_emit_minus_exists"]["mixed_augrc"]
        mean=100*r["mean"];ax.plot([100*x for x in r["ci95"]],[y,y],color=col,lw=2);ax.plot(mean,y,"o",color=col)
        ax.annotate(f"{mean:+.3f}",(mean,y),xytext=(0,11),textcoords="offset points",ha="center",color=col,fontsize=9)
    ax.set_yticks([1,0],["MM-GDINO","MDETR"]);ax.set_ylim(-.5,1.5);ax.margins(x=.25)
    ax.set_xlabel("Y − E risk change ×100\nPaired-L1 supervision",fontsize=8)
    ax.set_title("(a) Same labels, different risk",fontsize=9,pad=14,weight="bold")
    ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0,labelsize=8)
    ax=axs[1];v=seed_display.seed_effects(previous)["per_seed"];ax.axhline(0,color=".6",ls="--",lw=.8)
    for seed,col,marker in zip(old.SEEDS,["#8456a2","#26867b","#c8742f"],["o","s","^"]):
        ax.plot([0,1],[v[seed]["inference_only"],v[seed]["matched_retraining"]],color=col,marker=marker,ms=4,lw=1,label=seed)
    ax.scatter([-.1,1.1],[sum(v[s][k] for s in old.SEEDS)/3 for k in ("inference_only","matched_retraining")],color=".15",marker="D",s=23,label="Mean")
    ax.annotate("Seed 17: +4.726",(0,v["17"]["inference_only"]),xytext=(0,8),textcoords="offset points",fontsize=7,color="#8456a2")
    ax.set_ylim(-.75,5.5);ax.set_xlim(-.25,1.25);ax.set_xticks([0,1],["Inference\nG → S","Train +\ndeploy S"],fontsize=8)
    ax.set_ylabel("MDETR risk change ×100",fontsize=8);ax.set_title("(b) Readout sensitivity varies",fontsize=9,pad=14,weight="bold")
    ax.legend(loc="upper right",fontsize=6.5,frameon=False,handlelength=1)
    ax.spines[["top","right"]].set_visible(False)
    ax=axs[2];d=data["finecops_val"]
    for target,col,lab in (("exists","#c8742f","Existence E"),("emit","#236b91","Correct-output Y")):
        arms=["l1_uniform__"+target,"all_uniform__"+target]
        means=[100*d["summary"][a]["mixed_augrc"]["mean"] for a in arms]
        ax.plot([0,1],means,color=col,lw=2.2,marker="o",ms=5,label=lab)
        for seed in old.SEEDS:
            ax.scatter([-.045,1.045],[100*d["per_seed"][seed][a]["mixed_augrc"] for a in arms],s=9,color=col,alpha=.5)
        ax.annotate(f"{means[1]-means[0]:+.3f}",(.6,(means[0]+means[1])/2),xytext=(2,6 if target=='exists' else -15),textcoords="offset points",fontsize=9,color=col,weight="bold")
    ax.set_xlim(-.17,1.17);ax.set_ylim(25.45,27.15);ax.set_xticks([0,1],["L1-uniform","Full-uniform"],fontsize=8)
    ax.set_ylabel("MM-GDINO AUGRC ×100 ↓",fontsize=8)
    ax.legend(loc="upper left",fontsize=7,frameon=False,handlelength=1.4)
    ax.set_title("(c) Coverage changes preference",fontsize=9,pad=14,weight="bold")
    ax.spines[["top","right"]].set_visible(False)
    for ax in axs:ax.tick_params(axis="y",labelsize=8)
    fig.text(.02,.09,"FineCops val • Fixed output boxes • (a) mean + image CI; (b) paired seeds; (c) coverage intervention, means + seeds",fontsize=8)
    fig.text(.02,.035,"Coverage improves Y but worsens E: the target preference changes without changing the grounder or training budget.",fontsize=8,color=".3")
    for ext in ("pdf","svg"):fig.savefig(out/("figure1_study."+ext),metadata={"Date":None} if ext=="svg" else {"CreationDate":None,"ModDate":None})
    plt.close(fig)
    # Fixed-mean risk curves are analytic displays of already estimated AUROCs.
    d=data["finecops_val"];p=d["population"];a=p["C"]/(p["C"]+p["W"]);pi=np.linspace(0,1,501)
    fig,ax=plt.subplots(figsize=(4.0,2.7));fig.subplots_adjust(left=.16,right=.96,bottom=.26,top=.88)
    ax.axhline(0,color=".5",lw=.8);ax.axvline(p["N"]/p["records"],color=".6",ls=":",lw=1,label="Observed mixture")
    for name,col,label in (("l1_emit_minus_exists","#c8742f","L1-uniform"),("all_emit_minus_exists","#236b91","Full-uniform")):
        e=d["effects"][name];y=-(1-pi)*a*((1-pi)*(1-a)*e["correctness_auroc"]["mean"]+pi*e["correct_vs_no_target_auroc"]["mean"])*100
        ax.plot(pi,y,color=col,lw=2,label=label);root=d["augrc_crossovers"][name]
        ax.plot(root["ci95"],[0,0],color=col,lw=4,alpha=.45);ax.plot(root["point"]["prior"],0,"o",color=col,ms=4)
    ax.set_xlabel("No-target request fraction π");ax.set_ylabel("Y − E AUGRC ×100")
    ax.set_title("Coverage moves the risk crossover",fontsize=10,weight="bold")
    ax.legend(fontsize=7,loc="lower right",frameon=False);ax.spines[["top","right"]].set_visible(False)
    for ext in ("pdf","svg"):fig.savefig(out/("figure2_crossover."+ext),metadata={"Date":None} if ext=="svg" else {"CreationDate":None,"ModDate":None})
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--check",action="store_true");args=p.parse_args()
    previous,data,sources=load_sources();out=PAPER/"generated/coverage_v8_r2"
    if args.check:
        r=json.loads((out/"receipt.json").read_text());assert r["sources"]==sources and r["generator"]==bind(__file__)
        for name,digest in r["outputs"].items():assert old.sha(out/name)==digest
        assert json.loads((out/"risk_decomposition.json").read_text())==decompositions(data)
        print("completed coverage results, seed effects and publication assets verified");return
    if out.exists():raise FileExistsError("append-only asset directory exists")
    out.mkdir(parents=True)
    for name,text in {**main_tables(previous,data),**supplement_tables(data),"numbers.tex":macros(data)}.items():(out/name).write_text(text)
    (out/"risk_decomposition.json").write_text(json.dumps(decompositions(data),indent=2,sort_keys=True)+"\n")
    figures(previous,data,out)
    (out/"receipt.json").write_text(json.dumps({"schema":"arrow.paper.coverage_v8_assets/v1","sources":sources,
        "generator":bind(__file__),"outputs":{p.name:old.sha(p) for p in out.iterdir()},
        "new_training_or_scoring":False,"new_bootstrap":False},indent=2,sort_keys=True)+"\n")

if __name__=="__main__":main()
