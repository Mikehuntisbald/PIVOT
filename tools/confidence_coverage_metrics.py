"""Record-only coverage x target contrasts using the sealed v6 metric kernels."""
import hashlib
import numpy as np
from tools import confidence_readout_metrics as old

CELLS = ("l1_uniform__exists", "l1_uniform__emit", "all_uniform__exists", "all_uniform__emit")
EFFECTS = {
    "D_emit": {CELLS[3]: 1., CELLS[1]: -1.},
    "D_exists": {CELLS[2]: 1., CELLS[0]: -1.},
    "interaction": {CELLS[3]: 1., CELLS[2]: -1., CELLS[1]: -1., CELLS[0]: 1.},
    "l1_emit_minus_exists": {CELLS[1]: 1., CELLS[0]: -1.},
    "all_emit_minus_exists": {CELLS[3]: 1., CELLS[2]: -1.},
}


def analyze_coverage(runs, iterations=5000, seed=20260912, required_seeds=("17", "42", "73"),
                     conditionals=True, progress=None):
    if type(iterations) is not int or iterations < 1:
        raise ValueError("positive bootstrap count required")
    # Use the frozen identity validator via a private slot adapter, NOT by
    # relabelling published coverage records as readout experiments.
    aliases = dict(zip(CELLS, old.CELLS))
    adapted = {s: [{**r, "scores": {aliases.get(k,k):v for k,v in r["scores"].items()},
                    "readout_diagnostics": {}} for r in rows] for s,rows in runs.items()}
    aligned, lookup, strata, _ = old._validate({"mmgdino_positive": adapted}, required_seeds)
    reverse = {v:k for k,v in aliases.items()}
    rows_by_seed = {s: [{**r, "scores": {reverse.get(k,k):v for k,v in r["scores"].items()}}
                        for r in rows] for s,rows in aligned["mmgdino_positive"].items()}
    first = next(iter(rows_by_seed.values()))
    clusters = np.asarray([lookup[r["cluster_id"]] for r in first], dtype=np.int64)
    arms = tuple(sorted(first[0]["scores"]))
    prepared = {s: {a: old._PreparedReadout(rows,a,clusters,len(lookup),conditionals) for a in arms}
                for s,rows in rows_by_seed.items()}
    unit = np.ones(len(lookup))
    points = {s:{a:p.evaluate(unit) for a,p in models.items()} for s,models in prepared.items()}
    metrics = tuple(next(iter(points.values()))["native"])
    draws = {a:{m:np.full(iterations,np.nan) for m in metrics} for a in arms}
    rng = np.random.Generator(np.random.PCG64(seed)); digest = hashlib.sha256()
    for i in range(iterations):
        weights = old._draw_cluster_weights(rng,strata,len(lookup))
        digest.update(weights.astype("<u4").tobytes())
        for a in arms:
            models = list(prepared.values())[:1] if a == "native" else prepared.values()
            values = [model[a].evaluate(weights) for model in models]
            for m in metrics:
                draws[a][m][i] = np.mean([v[m] for v in values])
        if progress and ((i+1)%25 == 0 or i+1 == iterations): progress(i+1,iterations)
    summaries = {a:{m:old._summary([p[a][m] for p in points.values()],draws[a][m])
                    for m in metrics} for a in arms}
    effects = dict(EFFECTS)
    for coverage in ("l1_uniform","all_uniform"):
        for target in ("exists","emit"):
            reference = "paired_l1__"+target
            if reference in arms:
                effects[coverage+"_minus_paired__"+target] = {coverage+"__"+target:1.,reference:-1.}
    output, crossovers = {}, {}
    for name, coefficients in effects.items():
        delta = {m:sum(c*draws[a][m] for a,c in coefficients.items()) for m in metrics}
        per_seed = {s:{m:old._effect_values(p,coefficients,m) for m in metrics} for s,p in points.items()}
        output[name] = {m:{**old._summary([p[m] for p in per_seed.values()],delta[m]),
                          "per_seed":{s:p[m] for s,p in per_seed.items()},"definition":coefficients}
                        for m in metrics}
        roots = [old.augrc_crossover(draws["native"]["native_p_at_1"][i],
                    delta["correctness_auroc"][i],delta["correct_vs_no_target_auroc"][i]) for i in range(iterations)]
        valid = [r["prior"] for r in roots if r["prior"] is not None]
        crossovers[name] = {"point":old.augrc_crossover(summaries["native"]["native_p_at_1"]["mean"],
            output[name]["correctness_auroc"]["mean"],output[name]["correct_vs_no_target_auroc"]["mean"]),
            "bootstrap_status_counts":{status:sum(r["status"]==status for r in roots) for status in {r["status"] for r in roots}},
            "ci95":np.percentile(valid,[2.5,97.5]).tolist() if len(valid)==iterations else None}
    errors=[]
    for models in points.values():
        for v in models.values():
            implied=old.augrc_from_states(v["native_p_at_1"],v["observed_no_target_fraction"],
                                         v["correctness_auroc"],v["correct_vs_no_target_auroc"])
            if np.isfinite(implied): errors.append(abs(implied-v["mixed_augrc"]))
    if errors and max(errors)>2e-12: raise ValueError("AUGRC state identity failed")
    state=old._state(first)
    return old.finite_json({"schema":"arrow.confidence_coverage.metrics/v1",
        "role":"posthoc_motivated_prospectively_locked_coverage_intervention",
        "primary_metric":"mixed_augrc","matched_cells":list(CELLS),
        "population":{"records":len(first),"images":len(lookup),"C":int((state==2).sum()),
                      "W":int((state==1).sum()),"N":int((state==0).sum())},
        "per_seed":points,"summary":summaries,"effects":output,"augrc_crossovers":crossovers,
        "conditional_counts":next(iter(prepared.values()))["native"].conditional.counts if conditionals else None,
        "max_state_identity_error":max(errors) if errors else None,
        "bootstrap":{"iterations":iterations,"seed":seed,"rng":"PCG64","unit":"image_cluster",
            "strata":{s:len(v) for s,v in strata.items()},"draws_sha256":digest.hexdigest(),
            "same_draw_all_arms_and_seeds":True,"q05_recomputed_each_draw":True,
            "fixed_threshold_fit":False,"uncertainty":"images conditional on fixed heads; equal seed mean"},
        "interpretation":{"negative_interaction_alone_is_not_emit_repair":True,
            "coverage_is_positive_population_intervention_not_difficulty_label_causality":True}})
