from __future__ import annotations

from pathlib import Path

import pytest

from tools import (
    run_stageb_confidence_adapter_candidate_split_tail_aligned_probe_evaluation as controller,
)


def test_command_is_fixed_u400_strict1607():
    command = controller.build_command()
    assert command[:2] == [str(controller.FIXED_PYTHON), str(controller._BASE.EVALUATOR)]
    assert command[command.index("--config") + 1] == str(controller.CONFIG)
    assert command[command.index("--ckpts") + 1] == str(controller.CHECKPOINT)
    assert command[command.index("--output_dir") + 1] == str(controller.OUTPUT)
    assert command[command.index("--batch_size") + 1] == "16"
    assert command[command.index("--num_workers") + 1] == "4"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--topk") + 1] == "1"
    assert "--amp" in command
    assert "--skip_ref" in command
    assert command[command.index("--tn_splits") + 1 : command.index("--tn_splits") + 3] == [
        "refcocop_val",
        "refcocog_umd_val",
    ]


def test_main_wiring_detector_requires_exact_contract(tmp_path: Path):
    source = tmp_path / "main.py"
    source.write_text(
        """
def _bind_stage_b_confidence_probe_admission(args):
    revision = 'word_veto_candidate_split_tail_aligned_v45'
    head = 'split_token_veto_global_absolute_joint_clip_v3'
    routing = 'balanced_top_quarter_cvar_v2'
    trust = 'top_quarter_cvar_v2'
    formal = 'u400_word_veto_candidate_split_tail_aligned_confidence_strict1607_v45'
    from tools import run_stageb_confidence_adapter_candidate_split_tail_aligned_probe_evaluation as promotion
    return revision, head, routing, trust, formal, promotion
""",
        encoding="utf-8",
    )
    assert controller._formal_main_admission_is_wired(source) is True
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "split_token_veto_global_absolute_joint_clip_v3", "wrong_head"
        ),
        encoding="utf-8",
    )
    assert controller._formal_main_admission_is_wired(source) is False


def test_admission_verifier_fails_before_reading_any_report(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        controller, "_formal_main_admission_is_wired", lambda path=None: False
    )
    with pytest.raises(
        controller._BASE.ProbeEvaluationError,
        match="cannot promote formal training",
    ):
        controller.verify_admission_report(Path("/does/not/exist.json"))


def test_v45_postflight_relabels_terminal_and_binds_joint_clip(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        controller,
        "_BASE_POSTFLIGHT",
        lambda preflight, summary_path=None: {
            "decision": "admit_to_formal_training",
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "contracts": {"terminal_u300_diagnostic": True},
        },
    )
    result = controller._v45_postflight({})
    assert result["diagnostic_only"] is True
    assert result["formal_gate_eligible"] is False
    assert result["contracts"]["terminal_u400_diagnostic"] is True
    assert result["contracts"][
        "split_token_veto_global_absolute_joint_clip_v3"
    ] is True
    assert "terminal_u300_diagnostic" not in result["contracts"]
