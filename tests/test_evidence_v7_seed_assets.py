import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"paper/scripts"))
import build_evidence_v7_seed_assets as display


def test_seed_effects_reproduce_mean_but_expose_outlier():
    data,_=display.v7.load_sources();r=display.seed_effects(data);p=r["per_seed"]
    assert p["17"]["inference_only"]==pytest.approx(4.725805834794916)
    assert p["42"]["inference_only"]<0<p["73"]["inference_only"]
    assert all(v["matched_retraining"]<0 for v in p.values())
    assert r["inference_only_sample_sd"]==pytest.approx(2.702728826,abs=1e-8)
    assert r["image_ci_is_not_training_seed_uncertainty"]
    assert r["new_bootstrap"] is False


def test_manuscript_and_caption_show_seed_heterogeneity():
    text=(ROOT/"paper/empirical_study_v7_1.tex").read_text()
    assert "inference-only mean is driven by seed 17" in text
    assert "$+4.726$, $-0.061$, and $+0.158$" in text
    assert "low training-seed" in text
    assert "evidence_v7_seed_r1/figure1_evidence.pdf" in text
