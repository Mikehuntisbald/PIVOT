import copy
import numpy as np
import pytest
from tools.grounding_emission_audit import weighted_q05, RiskCurve, Prepared, analyze


def fixture():
    rows=[]
    for i in range(3):
        for j,(kind,correct,score) in enumerate((("positive",True,.9),("positive",False,.6),("text",None,.2))):
            rows.append(dict(sample_id=f"{i}-{j}",cluster_id=str(i),stratum="val",kind=kind,correct=correct,
                             level=1+i,native_score=score,baseline_score=score,candidate_score=.5))
    return rows


def test_quantile_matches_expansion_and_recomputes():
    scores=np.array([.1,.2,.2,.8,.9])
    weights=np.array([0,2,3,1,4])
    assert weighted_q05(scores,weights)==pytest.approx(np.quantile(np.repeat(scores,weights),.05))
    assert weighted_q05(scores,np.ones(5)) != weighted_q05(scores,weights)
    with pytest.raises(ValueError):
        weighted_q05(scores,np.array([1,1,1,1,.5]))
    prep=Prepared(np.array([.1,.9,.3,.4]),np.array([2,1,0,0]),np.arange(4))
    assert prep.evaluate(np.ones(4))["diagnostic_fpr95"]==1
    changed=prep.evaluate(np.array([0,2,1,1]))
    assert changed["diagnostic_positive_q05"]==.9
    assert changed["diagnostic_fpr95"]==0


def test_ties_components_and_achieved_coverage():
    curve=RiskCurve.prepare(np.ones(3),np.array([0,1,2]),np.arange(3))
    out=curve.evaluate(np.ones(3))
    assert out["mixed_aurc"]==pytest.approx(2/3)
    assert out["wrong_box_aurc"]+out["no_target_aurc"]==out["mixed_aurc"]
    assert out["achieved_coverage_cov25"]==1
    assert out["mixed_risk_cov25"]==pytest.approx(2/3)


def test_weighted_curve_equals_expanded_sample():
    scores=np.array([1.,.8,.8,.4,.2]); state=np.array([0,1,2,2,0]); weights=np.array([0,2,3,1,2])
    got=RiskCurve.prepare(scores,state,np.arange(5)).evaluate(weights)
    take=np.repeat(np.arange(5),weights)
    want=RiskCurve.prepare(scores[take],state[take],np.arange(len(take))).evaluate(np.ones(len(take)))
    assert got==pytest.approx(want)


def test_shared_draws_determinism_and_native_parity():
    runs={"17":fixture(),"42":list(reversed(fixture()))}
    a=analyze(runs,iterations=20,seed=7)
    assert a==analyze(runs,iterations=20,seed=7)
    assert a["bootstrap"]["q05_recomputed_each_draw"]
    assert a["population"]==dict(records=9,positive_correct=3,positive_wrong=3,no_target=3,mixture="original validation population; no reweighting")
    assert a["contrasts"]["exists_minus_native"]["mixed_aurc"]["ci95"]==[0,0]
    assert a["summary"]["emit"]["mixed_aurc"]["mean"]==pytest.approx(2/3)
    bad=copy.deepcopy(runs);bad["42"][0]["native_score"]+=.1
    with pytest.raises(ValueError,match="Native parity"):
        analyze(bad,iterations=2)


def test_curve_errors_partition_and_no_undefined_hiding():
    out=analyze({"17":fixture()},iterations=10)
    for arm in out["per_seed"]["17"].values():
        assert arm["mixed_aurc"]==pytest.approx(arm["no_target_aurc"]+arm["wrong_box_aurc"])
    rows=fixture()
    for r in rows:
        if r["kind"]=="positive":r["correct"]=True
    out=analyze({"17":rows},iterations=3)
    metric=out["summary"]["native"]["correctness_auroc"]
    assert metric["mean"] is None and metric["ci95"] is None and metric["undefined_replicates"]==3
