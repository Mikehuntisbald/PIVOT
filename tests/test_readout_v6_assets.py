"""Synthetic v6 rendering/gating regressions; never build publication assets."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readout_v6_assets", ROOT/"paper/scripts/build_readout_v6_assets.py")
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def summary(mean=.1, width=.002):
    return {"mean":mean,"sample_sd":.0008,"ci95":[mean-width,mean+width],"undefined_replicates":0}


def fixture(surface="finecops_val"):
    metrics=(*assets.PRIMARY,"native_p_at_1","existence_auroc","mixed_aurc","diagnostic_fpr95")
    images,positives,negatives=assets.COUNTS[surface]
    local={"population":{"images":images,"C":positives-1000,"W":1000,"N":negatives,
                         "records":positives+negatives},
           "per_seed":{seed:{} for seed in assets.SEEDS},"conditional_counts":{},
           "max_state_identity_error":1e-15,
           "summary":{arm:{metric:summary() for metric in metrics} for arm in assets.ARMS},
           "effects":{},
           "winner_geometry":{seed:{arm:{state:{} for state in ("C","W","N")} for arm in assets.CELLS}
                              for seed in assets.SEEDS}}
    for effect,value in (("D_emit",-.002),("D_exists",-.001),("interaction",-.001),
                         ("global_emit_minus_exists",.002),("selected_emit_minus_exists",.001)):
        local["effects"][effect]={metric:summary(value,.0005) for metric in metrics}
    total=positives+negatives
    pc,pw,pn=(positives-1000)/total,1000/total,negatives/total
    global_delta=-pc*(pw+pn)*.002
    local["effects"]["global_emit_minus_exists"]["mixed_augrc"]=summary(global_delta,.0005)
    local["effects"]["selected_emit_minus_exists"]["mixed_augrc"]=summary(global_delta-.001,.0005)
    for arm,mean in zip(assets.CELLS,(.1,.1+global_delta,.099,.1+global_delta-.002)):
        local["summary"][arm]["mixed_augrc"]=summary(mean)
    for effect in ("D_emit","D_exists","interaction","global_emit_minus_exists"):
        local["effects"][effect]["mixed_augrc"]["sample_sd"]=0.
    for arm in ("joint_product","joint_sirc"):
        for baseline in ("native",*assets.CELLS[:2]):
            local["effects"][arm+"_minus_"+baseline]={metric:summary(-.01 if baseline=="native" else .01)
                                                     for metric in metrics}
    for arm in assets.ARMS:
        for coverage in (50,90):
            for metric,value in (("wrong_box_risk",.1),("no_target_risk",.2),("mixed_risk",.3),
                                 ("achieved_coverage",coverage/100+.00003)):
                local["summary"][arm][metric+"_cov"+str(coverage)]=summary(value)
    for state in ("cw","cn","wn"):
        local["conditional_counts"]["same_image_"+state]={"eligible_images":30,"eligible_records":100,"within_image_pairs":160}
        for level in (1,2,3):
            local["conditional_counts"][f"difficulty_{state}_level{level}"]={
                "high":100,"low":20 if state=="cw" or level==1 else 0,"images":10}
        for condition in ("same_image","difficulty"):
            prefix=f"{condition}_{state}"
            conditional=prefix if condition=="same_image" else prefix+"_same_level_pair_auroc"
            difference=prefix+("_unconditional_minus_conditional" if condition=="same_image" else "_unconditional_minus_same_level")
            for effect in ("D_emit","global_emit_minus_exists"):
                local["effects"][effect].update({prefix+"_comparable_unconditional":summary(-.02),
                                                conditional:summary(-.01),difference:summary(-.01)})
    for state in ("all","C","W"):
        metric="parent_pair_"+state
        local["conditional_counts"][metric]={"pairs":50,"images":10}
        for arm in assets.ARMS:local["summary"][arm][metric]=summary(.6)
        for effect in ("D_emit","global_emit_minus_exists"):local["effects"][effect][metric]=summary(-.01)
    cross={}
    for arm in assets.CELLS:
        read="native_selected" if arm.startswith("global_max") else "global_max"
        name=arm+("__eval_selected" if read=="native_selected" else "__eval_global")
        cross[name]={"trained_head":arm,"eval_readout":read}
        local["summary"][name]={"mixed_augrc":summary(.11)}
        local["effects"]["fixed_weights__"+name+"_minus_matched"]={"mixed_augrc":summary(.01)}
    local["per_seed"]={seed:{arm:{metric:value["mean"] for metric,value in estimates.items()}
                             for arm,estimates in local["summary"].items()} for seed in assets.SEEDS}
    local["augrc_crossovers"]={name:{"point":{"prior":None,"status":"no_interior_root"},"ci95":None,
                                    "conditional_on_interior_ci95":None,"bootstrap_status_counts":{"no_interior_root":5000}}
                               for name in ("global_emit_minus_exists","selected_emit_minus_exists","D_emit")}
    for seed in assets.SEEDS:
        for arm in assets.CELLS:
            for state in ("C","W","N"):
                local["winner_geometry"][seed][arm][state]={"records":100,"winner_differs_mean":.2,
                    "winner_native_box_iou_mean":.7,"confidence_winner_correct_fraction":.1 if state=="W" else None}
    return {"schema":"arrow.confidence_readout_metrics/v1","primary_metric":"mixed_augrc",
            "matched_cells":list(assets.CELLS),"localizers":{key:copy.deepcopy(local) for key in assets.LOCALIZERS},
            "bootstrap":{"iterations":5000,"seed":20260911,"rng":"PCG64","unit":"image_cluster",
                         "required_seeds":list(assets.SEEDS),"same_draw_all_localizers_heads_seeds":True,
                         "q05_recomputed_each_draw":True,"fixed_threshold_fit":False,
                         "strata":{"validation":images} if surface=="finecops_val" else {"testA":700,"testB":images-700}},
            "receipt":{"formal_requested_configuration":True,"protocol_sha256":"a"*64},
            "cross_readout_scores":cross}


class ReadoutPaperAssetsTest(unittest.TestCase):
    def test_complete_synthetic_shape_is_accepted_without_rendering(self):
        for surface in assets.SURFACES:
            assets.validate_analysis(fixture(surface),surface)

    def test_first_localizer_cannot_finalize_cross_model_paper(self):
        data=fixture()
        del data["localizers"][assets.LOCALIZERS[1]]
        with self.assertRaisesRegex(ValueError,"two-localizer"):
            assets.validate_analysis(data,"finecops_val")

    def test_missing_seed_short_bootstrap_and_wrong_strata_rejected(self):
        for change in (
            lambda d:d["bootstrap"].update(iterations=100),
            lambda d:d["bootstrap"].update(unit="expression_iid"),
            lambda d:d["bootstrap"].update(strata={"pooled":3567}),
            lambda d:d["receipt"].update(formal_requested_configuration=False),
            lambda d:d["localizers"][assets.LOCALIZERS[0]]["per_seed"].pop("73"),
        ):
            data=fixture();change(data)
            with self.assertRaises(ValueError):assets.validate_analysis(data,"finecops_val")

    def test_incomplete_primary_interval_or_interaction_identity_rejected(self):
        data=fixture();data["localizers"][assets.LOCALIZERS[0]]["summary"][assets.CELLS[0]]["mixed_augrc"]["ci95"]=None
        with self.assertRaises(ValueError):assets.validate_analysis(data,"finecops_val")
        data=fixture();data["localizers"][assets.LOCALIZERS[0]]["effects"]["interaction"]["mixed_augrc"]["mean"]=-.1
        with self.assertRaisesRegex(ValueError,"interaction"):
            assets.validate_analysis(data,"finecops_val")

    def test_interaction_only_is_not_emission_repair(self):
        text=assets.readout_interpretation(summary(.002,.001),summary(-.01,.001))
        self.assertIn("not an absolute improvement",text)
        self.assertNotIn("selected design reduces",text)

    def test_equal_target_improvement_is_general_not_emit_specific(self):
        text=assets.readout_interpretation(summary(-.01,.001),summary(0,.001))
        self.assertIn("general readout effect",text)
        self.assertNotIn("benefits emission more",text)

    def test_resolved_both_effects_never_claim_unique_spatial_cause(self):
        text=assets.readout_interpretation(summary(-.01,.001),summary(-.002,.001))
        self.assertIn("reduces emission risk",text)
        self.assertIn("both paired intervals",text)
        self.assertNotIn("spatial",text)

    def test_weak_native_baseline_win_does_not_hide_learned_head_costs(self):
        data={s:fixture(s) for s in assets.SURFACES}
        text=assets.result_prose(data)["combination_results.tex"]
        self.assertIn("lower than Native",text)
        self.assertIn("higher risk than $G/E$, $G/Y$",text)
        self.assertNotIn("solves",text.lower())
        self.assertNotIn("incompatible",text.lower())

    def test_primary_table_contains_all_cells_sd_and_three_paired_effects(self):
        text=assets.main_tables({s:fixture(s) for s in assets.SURFACES})["table_target_readout.tex"]
        for value in ("$G/E$","$G/Y$","$S/E$","$S/Y$","$D_Y$","$D_E$","$I$","sample SD","paired 95"):
            self.assertIn(value,text)
        for loc in assets.LOCALIZERS:self.assertIn(assets.LOC[loc],text)
        for surface in assets.SURFACES:self.assertIn(assets.SUR[surface],text)

    def test_small_positive_ci_endpoint_keeps_its_sign(self):
        value={"mean":-.00011758,"ci95":[-.00026533,.00003363]}
        self.assertIn("+0.003",assets.estimate(value,True))
        self.assertEqual(assets.number(-1e-10,True),"+0.000")

    def test_caption_tex_commands_not_linebreaks(self):
        tables=assets.main_tables({s:fixture(s) for s in assets.SURFACES})
        for text in tables.values():
            caption=next(line for line in text.splitlines() if line.startswith("\\caption"))
            self.assertNotIn(r"\\times",caption)
            self.assertNotIn(r"\\%",caption)
            self.assertIn(r"\times100",caption)

    def test_prose_percent_is_tex_escape_not_linebreak(self):
        text=assets.result_prose({s:fixture(s) for s in assets.SURFACES})["target_results.tex"]
        self.assertNotIn(r"\\%",text)
        self.assertIn(r"\%",text)

    def test_real_mm_stage_cannot_be_adopted_as_final(self):
        path=ROOT/"paper/data/readout_v6/finecops_val_mm_stage.json"
        if not path.exists():self.skipTest("stage artifact not in this checkout")
        data=json.loads(path.read_text())
        self.assertEqual(data["bootstrap"]["iterations"],5000)
        self.assertEqual(set(data["localizers"]),{assets.LOCALIZERS[0]})
        with self.assertRaises(ValueError):assets.validate_analysis(data,"finecops_val")

    def test_conditionals_report_both_estimates_difference_and_counts(self):
        result=assets.supplementary_mechanism({s:fixture(s) for s in assets.SURFACES})
        self.assertIn("Comparable uncond.",result["supp_conditionals.tex"])
        self.assertIn("Conditional",result["supp_conditionals.tex"])
        self.assertIn("Uncond.$-$cond.",result["supp_conditionals.tex"])
        self.assertIn("$Y:S-G$",result["supp_conditionals.tex"])
        self.assertIn("$G:Y-E$",result["supp_conditionals.tex"])
        self.assertIn("not a difficulty-matched",result["supp_conditionals.tex"])
        counts=result["supp_condition_counts.tex"]
        self.assertIn("Within-image pairs",counts)
        self.assertIn("High-state requests",counts)
        self.assertIn("Low-state requests",counts)
        self.assertIn("MM-GDINO-T & CN & L2 & 100 & 0 & 10",counts)
        self.assertIn("MDETR-R101 & WN & L3 & 100 & 0 & 10",counts)

    def test_cross_readouts_print_matched_alternate_and_paired_delta(self):
        result=assets.supplementary_mechanism({s:fixture(s) for s in assets.SURFACES})
        text=result["supp_cross_readouts.tex"]
        for label in ("Matched (SD)","Alternate (SD)","Alt.$-$matched [CI]","Native boxes fixed","not substituted"):
            self.assertIn(label,text)
        self.assertIn("MM-GDINO-T & FineCops val & $G/E$ & $S$",text)
        self.assertIn("MDETR-R101 & gRef Full & $S/Y$ & $G$",text)

    def test_selected_global_winner_is_explicitly_counterfactual(self):
        text=assets.supplementary({s:fixture(s) for s in assets.SURFACES})["supp_winner_geometry.tex"]
        self.assertIn("Dense-logit global-winner",text)
        self.assertIn("For S-trained heads",text)
        self.assertIn("counterfactual diagnostic",text)
        self.assertIn("zero readout-index disagreement",text)

    def test_parent_pairs_keep_counts_and_official_metric_scope(self):
        text=assets.supplementary_mechanism({s:fixture(s) for s in assets.SURFACES})["supp_parent_pairs.tex"]
        self.assertIn("Pairs (images)",text)
        self.assertIn("not official Recall@1",text)
        self.assertIn("image counts may overlap",text)
        self.assertIn("$G:Y-E$",text)
        self.assertIn("$Y:S-G$",text)

    def test_new_supplementary_captions_have_correct_tex_escaping(self):
        results=assets.supplementary_mechanism({s:fixture(s) for s in assets.SURFACES})
        for text in results.values():
            for line in text.splitlines():
                if line.startswith("\\caption"):
                    self.assertNotIn(r"\\times",line)
                    self.assertNotIn(r"\\%",line)

    def test_fixed_coverage_components_and_denominators_are_explicit(self):
        result=assets.supplementary_fixed_coverage({s:fixture(s) for s in assets.SURFACES})
        text=result["supp_fixed_coverage.tex"]
        self.assertIn("among accepted requests",text)
        self.assertIn("not counts or failure rates divided by all requests",text)
        self.assertIn("W+N=mixed",text)
        self.assertIn("50.003",text)
        self.assertIn("90.003",text)
        for localizer in assets.LOCALIZERS:
            for arm in assets.CELLS:
                for coverage in (50,90):
                    self.assertIn(assets.LOC[localizer]+" & "+assets.ARM[arm]+" & "+str(coverage),text)

    def test_fixed_coverage_failure_sum_and_boundary_tie_checks(self):
        data={s:fixture(s) for s in assets.SURFACES}
        values=data["finecops_val"]["localizers"][assets.LOCALIZERS[0]]["summary"][assets.CELLS[0]]
        values["mixed_risk_cov50"]["mean"]=.31
        with self.assertRaisesRegex(ValueError,"sum"):assets.supplementary_fixed_coverage(data)
        values["mixed_risk_cov50"]["mean"]=.3
        values["achieved_coverage_cov50"]["mean"]=.499
        with self.assertRaisesRegex(ValueError,"coverage"):assets.supplementary_fixed_coverage(data)

    def test_seed_effect_table_separates_sample_sd_from_image_ci(self):
        text=assets.supplementary_seed_effects({s:fixture(s) for s in assets.SURFACES})["supp_seed_effects.tex"]
        for label in ("Seed 17","Seed 42","Seed 73","Mean (SD)","Image CI","$D_Y$","$D_E$","$I$","$G:Y-E$"):
            self.assertIn(label,text)
        self.assertIn("not a training-seed interval",text)
        self.assertIn("different signs",text)
        self.assertIn("No $n=3$ t-test",text)

    def test_seed_effect_mean_sd_and_optional_direct_effects_are_checked(self):
        r=fixture()["localizers"][assets.LOCALIZERS[0]]
        points=assets.seed_effect_values(r,"D_emit")
        self.assertTrue(all(abs(v+.002)<1e-12 for v in points.values()))
        r["effects"]["D_emit"]["mixed_augrc"]["sample_sd"]=.1
        with self.assertRaisesRegex(ValueError,"sample SD"):assets.seed_effect_values(r,"D_emit")
        r["effects"]["D_emit"]["mixed_augrc"]["sample_sd"]=0.
        r["per_seed_effects"]={seed:{"D_emit":{"mixed_augrc":points[seed]}} for seed in assets.SEEDS}
        assets.seed_effect_values(r,"D_emit")
        r["per_seed_effects"]["17"]["D_emit"]["mixed_augrc"]+=.01
        with self.assertRaisesRegex(ValueError,"provided per-seed"):assets.seed_effect_values(r,"D_emit")

    def test_real_mdetr_disjoint_seed_signs_not_hidden_by_negative_mean_ci(self):
        path=ROOT/"paper/data/readout_v6/gref_finecops_train_val_source_disjoint.json"
        if not path.exists():self.skipTest("full source-disjoint analysis not in checkout")
        r=json.loads(path.read_text())["localizers"]["mdetr_r101_refcoco_ema"]
        points=assets.seed_effect_values(r,"global_emit_minus_exists")
        self.assertLess(points["17"],0)
        self.assertGreater(points["42"],0)
        self.assertGreater(points["73"],0)
        self.assertLess(r["effects"]["global_emit_minus_exists"]["mixed_augrc"]["ci95"][1],0)
        for effect in ("D_emit","D_exists","interaction"):
            assets.seed_effect_values(r,effect)


if __name__=="__main__":unittest.main()
