"""Model-free three-score evaluation of existence and emission decisions.

Whole-pool failure = no referent OR incorrect positive localization. Keeps
actual population mixture explicit; does not select or calibrate a policy.
"""
from dataclasses import dataclass
import numpy as np
from tools.grounding_confidence_ordering import _GroupedMetric, _validate_runs, _draw_cluster_weights

ARMS = ("native", "exists", "emit")
COVERAGES = (.1, .25, .5, .75, .9, 1.)


def weighted_q05(scores, weights):
    """NumPy linear q05 of the integer-weight expanded sample, without expansion."""
    scores, weights = np.asarray(scores), np.asarray(weights)
    if scores.ndim != 1 or scores.shape != weights.shape or not np.isfinite(scores).all():
        raise ValueError("quantile alignment/finite contract")
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights != np.floor(weights)).any():
        raise ValueError("integer replicate multiplicities required")
    if not weights.sum():
        return np.nan
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    h = .05 * (int(cumulative[-1])-1)
    low, high = int(np.floor(h)), int(np.ceil(h))
    a, b = np.searchsorted(cumulative, [low, high], side="right")
    return float(scores[order[a]] + (h-low)*(scores[order[b]]-scores[order[a]]))


@dataclass
class RiskCurve:
    clusters: np.ndarray
    wrong: np.ndarray
    absent: np.ndarray
    starts: np.ndarray

    @classmethod
    def prepare(cls, scores, state, clusters):
        order = np.argsort(-scores, kind="stable")
        starts = np.r_[0, np.flatnonzero(scores[order][1:] != scores[order][:-1])+1]
        return cls(clusters[order], (state[order] == 1).astype(float), (state[order] == 0).astype(float), starts)

    def evaluate(self, cluster_weights):
        weights = cluster_weights[self.clusters]
        count = np.add.reduceat(weights, self.starts)
        wrong = np.add.reduceat(weights*self.wrong, self.starts)
        absent = np.add.reduceat(weights*self.absent, self.starts)
        active = count > 0
        count, wrong, absent = count[active], wrong[active], absent[active]
        if not len(count):
            return {}
        retained = np.cumsum(count)
        cov = retained / retained[-1]
        wr, ar = np.cumsum(wrong)/retained, np.cumsum(absent)/retained
        result = {}
        for name, risk in (("wrong_box", wr), ("no_target", ar), ("mixed", wr+ar)):
            result[name+"_aurc"] = float(np.sum(np.diff(np.r_[0., cov])*(risk+np.r_[risk[0],risk[:-1]])/2))
        for c in COVERAGES:
            i = min(int(np.searchsorted(cov, c, side="left")), len(cov)-1)
            suffix = f"_cov{int(c*100)}"
            result["mixed_risk"+suffix] = float(wr[i]+ar[i])
            result["wrong_box_risk"+suffix] = float(wr[i])
            result["no_target_risk"+suffix] = float(ar[i])
            result["achieved_coverage"+suffix] = float(cov[i])
        return result


class Prepared:
    def __init__(self, scores, state, clusters):
        self.scores, self.state, self.clusters = scores, state, clusters
        positive, correct = state > 0, state == 2
        definitions = {
            "existence_auroc": (np.ones(len(state),dtype=bool), positive, "auc"),
            "emission_auroc": (np.ones(len(state),dtype=bool), correct, "auc"),
            "correctness_auroc": (positive, correct, "auc"),
            "correct_vs_no_target_auroc": (state != 1, correct, "auc"),
            "wrong_positive_vs_no_target_auroc": (state != 2, positive, "auc"),
            "positive_aurc": (positive, ~correct, "aurc"),
        }
        self.metrics = {name: _GroupedMetric.prepare(kind, scores[mask], labels[mask].astype(float), clusters[mask])
                        for name,(mask,labels,kind) in definitions.items()}
        self.curve = RiskCurve.prepare(scores, state, clusters)

    def evaluate(self, weights):
        result = {k: (v.evaluate(weights)[0]) for k,v in self.metrics.items()}
        result.update(self.curve.evaluate(weights))
        positive = self.state > 0
        pw = weights[self.clusters[positive]]
        nw = weights[self.clusters[~positive]]
        threshold = weighted_q05(self.scores[positive], pw)
        result["diagnostic_positive_q05"] = threshold
        result["diagnostic_fpr95"] = float(np.dot(nw, self.scores[~positive] >= threshold)/nw.sum()) if nw.sum() and np.isfinite(threshold) else np.nan
        return {k: float(v) if v is not None else np.nan for k,v in result.items()}


def analyze(runs, iterations=5000, seed=20260907):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    aligned, lookup, strata = _validate_runs(runs)
    first = next(iter(aligned.values()))
    for rows in aligned.values():
        for r,b in zip(rows,first):
            if not isinstance(r.get("native_score"),(float,int)) or isinstance(r["native_score"],bool) or not np.isfinite(r["native_score"]):
                raise ValueError("finite native score required")
            if r["native_score"] != b["native_score"]:
                raise ValueError("cross-seed Native parity")
    state = np.asarray([0 if r["kind"] != "positive" else (2 if r["correct"] else 1) for r in first])
    clusters = np.asarray([lookup[r["cluster_id"]] for r in first],dtype=np.int64)
    prep = {}
    columns = {"native": "native_score", "exists": "baseline_score", "emit": "candidate_score"}
    for s, rows in aligned.items():
        prep[s] = {a: Prepared(np.asarray([r[col] for r in rows]),state,clusters) for a,col in columns.items()}
    units = np.ones(len(lookup))
    point = {s: {a:p.evaluate(units) for a,p in arms.items()} for s,arms in prep.items()}
    metrics = tuple(next(iter(point.values()))["native"])
    samples = {a: {m: np.full(iterations,np.nan) for m in metrics} for a in ARMS}
    rng = np.random.Generator(np.random.PCG64(seed))
    native = next(iter(prep.values()))["native"]
    for i in range(iterations):
        weights = _draw_cluster_weights(rng,strata,len(lookup))
        for a in ARMS:
            values = [native.evaluate(weights)] if a == "native" else [p[a].evaluate(weights) for p in prep.values()]
            for m in metrics:
                samples[a][m][i] = np.mean([v[m] for v in values])
    summary = {}
    for a in ARMS:
        summary[a] = {}
        for m in metrics:
            vals = [point[s][a][m] for s in prep]
            draws = samples[a][m]
            invalid = int((~np.isfinite(draws)).sum())
            summary[a][m] = {"mean":float(np.mean(vals)), "sample_sd":float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                             "ci95":np.percentile(draws,[2.5,97.5]).tolist() if not invalid else None,
                             "undefined_replicates":invalid}
    contrasts = {}
    for a,b in (("exists","native"),("emit","native"),("emit","exists")):
        contrasts[a+"_minus_"+b] = {}
        for m in metrics:
            draws = samples[a][m]-samples[b][m]
            invalid = int((~np.isfinite(draws)).sum())
            contrasts[a+"_minus_"+b][m] = {
                "delta": summary[a][m]["mean"]-summary[b][m]["mean"],
                "ci95":np.percentile(draws,[2.5,97.5]).tolist() if not invalid else None,
                "undefined_replicates":invalid}
    result = {"schema":"arrow.emission_audit/v1", "role":"posthoc_exploratory_no_selection",
            "population":{"records":len(first),"positive_correct":int((state==2).sum()),
                          "positive_wrong":int((state==1).sum()),"no_target":int((state==0).sum()),
                          "mixture":"original validation population; no reweighting"},
            "bootstrap":{"iterations":iterations,"seed":seed,"rng":"PCG64","unit":"image_cluster",
                         "clusters":len(lookup),"same_draw_all_arms_and_seeds":True,"q05_recomputed_each_draw":True},
            "coverage_rule":"accept entire boundary tie group; report achieved coverage; no deploy threshold fitted",
            "risk_definition":"all accepted wrong-positive boxes and no-target records are failures",
            "per_seed":point,"summary":summary,"contrasts":contrasts}
    def finite_json(value):
        if isinstance(value,dict):
            return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):
            return [finite_json(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):
            return None
        return value
    return finite_json(result)
