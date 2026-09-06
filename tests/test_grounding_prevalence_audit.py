import numpy as np
import pytest
from tools.grounding_prevalence_audit import PriorCurve,analyze_prevalence
from tools.grounding_emission_audit import RiskCurve


def fixture():
    out=[]
    for i in range(3):
        for j,(kind,correct,score) in enumerate((("positive",True,.9),("positive",False,.5),("no_target",None,.8))):
            out.append(dict(sample_id=f"{i}-{j}",cluster_id=str(i),stratum="toy",kind=kind,correct=correct,level=None,
                            native_score=score,baseline_score=score,candidate_score=1-score))
    return out


def test_endpoints_and_observed_prior_reconstruct_original():
    scores=np.array([.9,.7,.5,.8]);state=np.array([2,2,1,0]);ids=np.arange(4);weights=np.ones(4)
    curve=PriorCurve.prepare(scores,state,ids)
    values=curve.evaluate(weights,[0.,.25,1.])
    base=RiskCurve.prepare(scores,state,ids).evaluate(weights)
    assert values[1,0]==pytest.approx(base["mixed_aurc"])
    positive=RiskCurve.prepare(scores[:3],state[:3],np.arange(3)).evaluate(np.ones(3))
    assert values[0,0]==pytest.approx(positive["mixed_aurc"])
    assert values[2].tolist()==pytest.approx([1,0,1])
    assert np.allclose(values[:,0],values[:,1]+values[:,2])


def test_rebuild_not_linear_mixture_of_aurcs():
    c=PriorCurve.prepare(np.array([.9,.1]),np.array([2,0]),np.arange(2))
    pi=.25
    got=c.evaluate(np.ones(2),[pi])[0,0]
    assert got==pytest.approx(pi*pi/2)
    assert got!=pytest.approx(pi)


def test_weighted_draw_equivalent_to_expanded_sample():
    s=np.array([.9,.8,.8,.5,.3]);state=np.array([2,0,1,2,0]);ids=np.arange(5);w=np.array([0,2,1,3,2])
    take=np.repeat(ids,w)
    a=PriorCurve.prepare(s,state,ids).evaluate(w,[0,.1,.5,1])
    b=PriorCurve.prepare(s[take],state[take],np.arange(len(take))).evaluate(np.ones(len(take)),[0,.1,.5,1])
    assert np.allclose(a,b)


def test_constant_tie_mass_and_deterministic_cluster_bootstrap():
    c=PriorCurve.prepare(np.ones(3),np.array([2,1,0]),np.arange(3))
    assert c.evaluate(np.ones(3),[.25])[0,0]==pytest.approx(.25+.75*.5)
    runs={"17":fixture(),"42":list(reversed(fixture()))}
    a=analyze_prevalence(runs,iterations=10,seed=7)
    assert a==analyze_prevalence(runs,iterations=10,seed=7)
    assert a["bootstrap"]["same_draw_all_scores_seeds_prevalences"]
    assert a["contrasts"]["0.5"]["exists_minus_native"]["mixed_aurc"]["ci95"]==[0,0]


def test_invalid_inputs_rejected_and_missing_classes_not_hidden():
    with pytest.raises(ValueError):analyze_prevalence({"1":fixture()},iterations=True)
    with pytest.raises(ValueError):analyze_prevalence({"1":fixture()},iterations=1,prevalences=[-.1])
    with pytest.raises(ValueError):analyze_prevalence({"1":fixture()},iterations=1,prevalences=[.5,.5])
    rows=[r for r in fixture() if r["kind"]=="no_target"]
    out=analyze_prevalence({"1":rows},iterations=3)
    assert out["summary"]["0.5"]["native"]["mixed_aurc"]["ci95"] is None
    assert out["summary"]["0.5"]["native"]["mixed_aurc"]["undefined_replicates"]==3
    assert out["summary"]["1"]["native"]["mixed_aurc"]["mean"]==1
