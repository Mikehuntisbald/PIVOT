"""Static contract checks for the data-FT total-trust config leaf."""

import ast
from pathlib import Path


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/ablations/cfg_stageb_gdino_score_adapter_dataft_total_trust.py"
)
BASE_CONFIG_PATH = CONFIG_PATH.with_name("cfg_stageb_gdino_score_adapter_dataft.py")
PIPELINE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/run_stagea_b58_r100_c100.sh"
)


def _literal_assignments(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_total_trust_leaf_preserves_dataft_scope_and_confidence_ownership():
    config = _literal_assignments(CONFIG_PATH)
    base = _literal_assignments(BASE_CONFIG_PATH)

    assert base["stage_b_gdino_score_adapter"] is True
    assert config["stage_b_gdino_adapter_train_mode"] == "confidence_only"
    assert config["stage_b_gdino_tn_scope"] == "benchmark_dataft_alltn"
    assert config["stage_b_gdino_confidence_objective"] == (
        "detached_recent_q05_total_trust"
    )
    assert config["stage_b_gdino_rank_weight"] == 0.0
    assert config["stage_b_gdino_confidence_weight"] == 1.0
    assert config["stage_b_gdino_paired_margin_weight"] == 0.0
    assert config["batch_size"] == 8


def test_stagea_r100_c100_pipeline_runs_and_enforces_all_sealed_replays():
    launcher = PIPELINE_PATH.read_text(encoding="utf-8")
    assert "--options batch_size=32" in launcher
    assert "--confidence-max-target 100" in launcher
    assert "run_stageb_gdino_adapter_total_trust_evaluation.py run" in launcher
    assert "stagea_r100_c100_all_sealed_gates_passed" in launcher
