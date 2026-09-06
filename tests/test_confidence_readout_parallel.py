import copy

import pytest

from tools.confidence_readout_metrics import CELLS, SEEDS, analyze_readout
from tools.analyze_confidence_readout_parallel import (
    code_hashes, full_contract, merge_blocks, run_parts, validate_reuse,
)


def synthetic():
    runs = {}
    for localizer, flipped in (("mmgdino_positive", False), ("mdetr_r101_refcoco_ema", True)):
        runs[localizer] = {}
        for seed in SEEDS:
            rows = []
            for image in range(4):
                for j in range(3):
                    pos, native = j < 2, .9 - j * .3 - .02 * image
                    correct = (j == 0) != flipped if pos else None
                    scores = dict(zip(CELLS, (native, 1-native, native+.1, 1.1-native)))
                    scores.update(joint_product=native*.9, joint_sirc=native*.8)
                    diagnostic = {}
                    for arm in CELLS:
                        value, glob = scores[arm], arm.startswith("global")
                        diagnostic[arm] = {
                            "max_logit": value if glob else value+.03,
                            "selected_logit": value-.03 if glob else value,
                            "confidence_winner_index": 2, "native_selected_index": 0,
                            "winner_native_box_iou": .7,
                            "native_gt_iou": .8 if correct else .2 if pos else None,
                            "winner_gt_iou": .6 if pos else None,
                        }
                    rows.append({"sample_id": f"{image}:{j}", "cluster_id": str(image),
                                 "stratum": "A" if image < 2 else "B", "kind": "positive" if pos else "text",
                                 "correct": correct, "level": image % 2 + 1 if pos else None,
                                 "parent_positive_id": None if pos else f"{image}:0",
                                 "native_score": native, "scores": scores, "readout_diagnostics": diagnostic})
            runs[localizer][seed] = rows
    return runs


def parts(runs, iterations=7):
    return {loc: analyze_readout({loc: seeds}, iterations=iterations, seed=20260911)
            for loc, seeds in runs.items()}


def test_joint_matches_spawned_parts_exactly_all_fields_and_diagnostics():
    runs = synthetic()
    contract = full_contract(runs, 7, 20260911)
    joined = run_parts(runs, contract, workers=2)
    expected = analyze_readout(runs, iterations=7, seed=20260911)
    assert joined == expected
    assert len(joined["cross_readout_scores"]) == 4
    assert len(joined["localizers"]["mmgdino_positive"]["summary"]) == 11


def test_cross_localizer_identity_must_be_validated_before_splitting():
    runs = synthetic()
    for rows in runs["mdetr_r101_refcoco_ema"].values():
        rows[0]["sample_id"] = "different"
        rows[2]["parent_positive_id"] = "different"
    with pytest.raises(ValueError, match="identities"):
        full_contract(runs, 7, 20260911)


@pytest.mark.parametrize("mutation", ("draw", "metadata", "population", "missing_seed", "missing_loc"))
def test_merge_refuses_any_scope_or_header_mismatch(mutation):
    runs = synthetic()
    contract = full_contract(runs, 7, 20260911)
    results = parts(runs)
    md = results["mdetr_r101_refcoco_ema"]
    if mutation == "draw":
        md["bootstrap"]["draws_sha256"] = "0" * 64
    elif mutation == "metadata":
        md["interpretation"]["spatial_alignment_is_unproven"] = False
    elif mutation == "population":
        md["localizers"]["mdetr_r101_refcoco_ema"]["population"]["C"] += 1
    elif mutation == "missing_seed":
        del md["localizers"]["mdetr_r101_refcoco_ema"]["per_seed"]["73"]
    else:
        del results["mdetr_r101_refcoco_ema"]
    with pytest.raises(ValueError):
        merge_blocks(results, contract)


def stage_fixture():
    runs = synthetic()
    contract = full_contract(runs, 7, 20260911)
    result = parts(runs)["mmgdino_positive"]
    bindings = {loc: {s: {"sha256": str(i) * 64, "rows": 12, "sirc_statistics_sha256": str(i+1) * 64}
                     for i, s in enumerate(SEEDS)} for loc in runs}
    source = {"surface": "finecops_val", "protocol_sha256": "a" * 64}
    result["receipt"] = {"stage_mm_only": True, "formal_requested_configuration": False,
        "study_final_receipt": False, "surface": "finecops_val", "protocol_sha256": "a" * 64,
        "records": {"mmgdino_positive": bindings["mmgdino_positive"]}, "code_sha256": code_hashes(),
        "model_forward": False, "checkpoint_selection": False, "threshold_fitting": False}
    return runs, contract, result, source, bindings


def test_reused_completed_block_and_computed_other_block_match_joint():
    runs, contract, result, source, bindings = stage_fixture()
    block = validate_reuse(result, source, bindings, contract, code_hashes())
    joined = run_parts(runs, contract, reusable={"mmgdino_positive": block}, workers=2)
    assert joined == analyze_readout(runs, iterations=7, seed=20260911)


@pytest.mark.parametrize("mutation", ("record", "stat", "code", "surface", "iterations", "missing_seed", "flag"))
def test_reuse_requires_full_record_stat_code_bootstrap_scope(mutation):
    _, contract, result, source, bindings = stage_fixture()
    if mutation == "record":
        result["receipt"]["records"] = copy.deepcopy(result["receipt"]["records"])
        result["receipt"]["records"]["mmgdino_positive"]["42"]["sha256"] = "f" * 64
    elif mutation == "stat":
        result["receipt"]["records"] = copy.deepcopy(result["receipt"]["records"])
        result["receipt"]["records"]["mmgdino_positive"]["42"]["sirc_statistics_sha256"] = "f" * 64
    elif mutation == "code":
        result["receipt"]["code_sha256"]["confidence_readout_metrics.py"] = "f" * 64
    elif mutation == "surface":
        source["surface"] = "gref_full"
    elif mutation == "iterations":
        result["bootstrap"]["iterations"] = 6
    elif mutation == "missing_seed":
        result["receipt"]["records"] = copy.deepcopy(result["receipt"]["records"])
        del result["receipt"]["records"]["mmgdino_positive"]["73"]
    else:
        result["receipt"]["model_forward"] = True
    with pytest.raises(ValueError):
        validate_reuse(result, source, bindings, contract, code_hashes())
