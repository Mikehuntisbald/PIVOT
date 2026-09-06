"""Finished experiments, accurate contribution sizes, and result-integrated prose."""
import copy
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"paper/scripts"))
import build_coverage_v8_assets as v8


@pytest.fixture(scope="module")
def sources():return v8.load_sources()


def test_all_surfaces_seeds_and_hashes_completed(sources):
    previous,data,bindings=sources
    assert set(data)==set(v8.SURFACES)
    assert len(bindings)>=8
    for surface in data:v8.validate(data[surface],surface)


@pytest.mark.parametrize("mutation",["seed","counts","draws","matrix"])
def test_incomplete_sources_rejected(sources,mutation):
    d=copy.deepcopy(sources[1]["finecops_val"])
    if mutation=="seed":del d["per_seed"]["17"]
    elif mutation=="counts":d["population"]["C"]+=1
    elif mutation=="draws":d["bootstrap"]["iterations"]=4999
    else:d["matched_cells"].pop()
    with pytest.raises(ValueError):v8.validate(d,"finecops_val")


def test_absolute_gain_is_not_interaction(sources):
    d=sources[1]["finecops_val"]["effects"]
    y=d["D_emit"]["mixed_augrc"];e=d["D_exists"]["mixed_augrc"];i=d["interaction"]["mixed_augrc"]
    assert y["mean"]==pytest.approx(-.003315217307526)
    assert e["mean"]>0 and i["mean"]==pytest.approx(y["mean"]-e["mean"])
    assert abs(i["mean"])>3*abs(y["mean"])
    for surface in sources[1].values():
        assert all(v<0 for v in surface["effects"]["D_emit"]["mixed_augrc"]["per_seed"].values())
        assert all(v>0 for v in surface["effects"]["D_exists"]["mixed_augrc"]["per_seed"].values())


def test_pairwise_tradeoff_and_exact_risk_explanation(sources):
    result=v8.decompositions(sources[1])
    for surface,d in sources[1].items():
        y=d["effects"]["D_emit"]
        assert y["correctness_auroc"]["ci95"][1]<0
        assert y["correct_vs_no_target_auroc"]["ci95"][0]>0
        r=result[surface]["D_emit"]
        assert r["cw_pair_contribution"]>0>r["cn_pair_contribution"]
        assert r["sum"]==pytest.approx(y["mixed_augrc"]["mean"],abs=2e-12)


def test_coverage_does_not_erase_all_cn_cost(sources):
    d=sources[1]["finecops_val"]
    assert d["effects"]["all_emit_minus_exists"]["correct_vs_no_target_auroc"]["ci95"][1]<0
    assert d["effects"]["D_emit"]["difficulty_cn_level1"]["mean"]<0
    assert d["effects"]["D_emit"]["difficulty_cn_cross_level_contribution"]["mean"]>0
    a=d["augrc_crossovers"]["l1_emit_minus_exists"]
    b=d["augrc_crossovers"]["all_emit_minus_exists"]
    assert a["point"]["prior"]==pytest.approx(.43424541257995336)
    assert b["point"]["prior"]==pytest.approx(.7685801476871171)
    assert a["bootstrap_status_counts"]==b["bootstrap_status_counts"]=={"interior":5000}


def test_gref_cn_uncertainty_and_absent_roots_not_hidden(sources):
    d=sources[1]["gref_source_disjoint"]
    assert d["effects"]["all_emit_minus_exists"]["correct_vs_no_target_auroc"]["ci95"][0]<0
    assert d["effects"]["all_emit_minus_exists"]["correct_vs_no_target_auroc"]["ci95"][1]>0
    r=d["augrc_crossovers"]["all_emit_minus_exists"]
    assert r["ci95"] is None and r["point"]["prior"] is None
    assert r["bootstrap_status_counts"]["no_interior_root"]>0


def test_generated_cells_and_claim_scope(sources):
    previous,data,_=sources
    tables=v8.main_tables(previous,data)
    assert "-0.332" in tables["table_coverage.tex"] and "+0.916" in tables["table_coverage.tex"]
    assert "-11.411" in tables["table_states.tex"] and "+4.412" in tables["table_states.tex"]
    main=(ROOT/"paper/empirical_study_v8.tex").read_text()
    assert "not a 1.248-point improvement of Y" in main
    assert "Coverage training was\nperformed only on MM-GDINO" in main
    assert "seed17-driven" not in main and "seed 17-driven" in main
    assert "SHA-256" not in main and "postflight" not in main
    assert "The novelty is" not in main
    assert "testing coverage as its cause remains" not in main
