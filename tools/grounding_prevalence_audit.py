"""Record-only prior sensitivity with class-normalized image-cluster draws.

Reweight requests before rebuilding the risk curve. Do not mix endpoint AURCs
linearly, fit a deployed prior/threshold, or change class-conditional scores.
"""
from dataclasses import dataclass
import numpy as np
from tools.grounding_confidence_ordering import _validate_runs,_draw_cluster_weights

PREVALENCES=(0.,.1,.25,.5,.75,.9,1.)
ARMS=("native","exists","emit")
METRICS=("mixed_aurc","wrong_box_aurc","no_target_aurc")


@dataclass
class PriorCurve:
    clusters: np.ndarray
    positive: np.ndarray
    wrong: np.ndarray
    starts: np.ndarray

    @classmethod
    def prepare(cls,scores,state,clusters):
        order=np.argsort(-scores,kind="stable")
        starts=np.r_[0,np.flatnonzero(scores[order][1:]!=scores[order][:-1])+1]
        return cls(clusters[order],state[order]>0,state[order]==1,starts)

    def evaluate(self,weights,prevalences):
        w=weights[self.clusters]
        gp=np.add.reduceat(w*self.positive,self.starts)
        gn=np.add.reduceat(w*~self.positive,self.starts)
        gw=np.add.reduceat(w*self.wrong,self.starts)
        pos,neg=float(gp.sum()),float(gn.sum())
        result=np.full((len(prevalences),3),np.nan)
        for i,pi in enumerate(prevalences):
            if (pi<1 and pos<=0) or (pi>0 and neg<=0):continue
            a=(1-pi)/pos if pos else 0.;b=pi/neg if neg else 0.
            count=a*gp+b*gn;active=count>0
            retained=np.cumsum(count[active]);coverage=retained/retained[-1]
            wrong=np.cumsum(a*gw[active])/retained
            absent=np.cumsum(b*gn[active])/retained
            def area(risk):
                return float(np.sum(np.diff(np.r_[0.,coverage])*(risk+np.r_[risk[0],risk[:-1]])/2))
            wa,na=area(wrong),area(absent)
            result[i]=(wa+na,wa,na)
        return result


def analyze_prevalence(runs,iterations=5000,seed=20260909,prevalences=PREVALENCES):
    if type(iterations) is not int or iterations<=0:raise ValueError("positive integer iterations required")
    if type(seed) is not int or seed<0:raise ValueError("nonnegative integer seed required")
    grid=np.asarray(prevalences,dtype=float)
    if grid.ndim!=1 or not len(grid) or not np.isfinite(grid).all() or (grid<0).any() or (grid>1).any() or len(set(grid))!=len(grid):
        raise ValueError("unique finite prevalence values in [0,1] required")
    aligned,lookup,strata=_validate_runs(runs);first=next(iter(aligned.values()))
    for rows in aligned.values():
        for r,b in zip(rows,first):
            if type(r.get("native_score")) not in (int,float) or not np.isfinite(r["native_score"]) or r["native_score"]!=b["native_score"]:
                raise ValueError("finite cross-seed-identical Native score required")
    state=np.asarray([0 if r["kind"]!="positive" else (2 if r["correct"] else 1) for r in first])
    clusters=np.asarray([lookup[r["cluster_id"]] for r in first],dtype=int)
    observed=float(np.mean(state==0));units=np.ones(len(lookup))
    prep={s:{a:PriorCurve.prepare(np.asarray([r[col] for r in rows]),state,clusters)
             for a,col in (("native","native_score"),("exists","baseline_score"),("emit","candidate_score"))}
          for s,rows in aligned.items()}
    points={s:{a:p.evaluate(units,grid) for a,p in arms.items()} for s,arms in prep.items()}
    observed_points={s:{a:p.evaluate(units,[observed])[0] for a,p in arms.items()} for s,arms in prep.items()}
    draws={a:np.full((iterations,len(grid),3),np.nan) for a in ARMS}
    rng=np.random.Generator(np.random.PCG64(seed));native=next(iter(prep.values()))["native"]
    for i in range(iterations):
        weights=_draw_cluster_weights(rng,strata,len(lookup))
        draws["native"][i]=native.evaluate(weights,grid)
        for a in ("exists","emit"):
            draws[a][i]=np.mean([p[a].evaluate(weights,grid) for p in prep.values()],axis=0)
    def value_summary(values,samples):
        invalid=int(np.count_nonzero(~np.isfinite(samples)))
        return {"mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)) if len(values)>1 else None,
                "ci95":np.percentile(samples,[2.5,97.5]).tolist() if not invalid else None,"undefined_replicates":invalid}
    summaries={};contrasts={}
    for j,pi in enumerate(grid):
        key=f"{pi:g}";summaries[key]={};contrasts[key]={}
        for a in ARMS:
            summaries[key][a]={m:value_summary([points[s][a][j,k] for s in prep],draws[a][:,j,k]) for k,m in enumerate(METRICS)}
        for a,b in (("exists","native"),("emit","native"),("emit","exists")):
            contrasts[key][a+"_minus_"+b]={}
            for k,m in enumerate(METRICS):
                vals=[points[s][a][j,k]-points[s][b][j,k] for s in prep]
                item=value_summary(vals,draws[a][:,j,k]-draws[b][:,j,k]);item["delta"]=item.pop("mean")
                contrasts[key][a+"_minus_"+b][m]=item
    result={"schema":"arrow.prevalence_audit/v1","role":"posthoc_prior_reweighting_no_selection",
        "prevalences":grid.tolist(),"observed_no_target_fraction":observed,
        "population":{"records":len(first),"positive":int((state>0).sum()),"no_target":int((state==0).sum())},
        "bootstrap":{"iterations":iterations,"seed":seed,"rng":"PCG64","unit":"image_cluster",
                     "strata":{s:len(v) for s,v in strata.items()},"same_draw_all_scores_seeds_prevalences":True,
                     "renormalize_classes_within_each_draw":True},
        "reweighting":"positive row multiplicity*(1-pi)/positive_total; absent row multiplicity*pi/absent_total",
        "risk_integration":"rebuild tie-group risk curve under new row weights; not a mixture of endpoint AURCs",
        "policy_selection":False,"summary":summaries,"contrasts":contrasts,
        "per_seed":{s:{a:{f"{pi:g}":dict(zip(METRICS,points[s][a][j].tolist())) for j,pi in enumerate(grid)} for a in ARMS} for s in prep},
        "observed_prior_per_seed":{s:{a:dict(zip(METRICS,v.tolist())) for a,v in arms.items()} for s,arms in observed_points.items()}}
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    return clean(result)
