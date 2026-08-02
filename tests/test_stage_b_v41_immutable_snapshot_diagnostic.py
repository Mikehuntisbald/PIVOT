import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_adapter_candidate_role_complete_carrier_snapshot_screen
    as screen,
)
from util.slconfig import SLConfig


CONFIG = combined_eval._CANDIDATE_ROLE_COMPLETE_CARRIER_CONFIDENCE_U0400_CONFIG
ROLE_SCOPE = (
    "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
)


def _args(update: int = 100, **overrides):
    values = {
        "immutable_v39_archived_snapshot_diagnostic": False,
        "immutable_v40_archived_snapshot_diagnostic": False,
        "immutable_v41_archived_snapshot_diagnostic": True,
        "partial_dense_duty_rank_diagnostic": False,
        "partial_dense_duty_confidence_diagnostic": True,
        "config": str(CONFIG),
        "output_dir": str(
            combined_eval._V41_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS[update]
        ),
        "ckpts": [
            str(combined_eval._V41_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS[update])
        ],
        "tn_jsonl": str(
            combined_eval._PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]["path"]
        ),
        "tn_splits": ["refcocop_val", "refcocog_umd_val"],
        "skip_tn": False,
        "skip_ref": True,
        "device": "cuda:0",
        "batch_size": 16,
        "num_workers": 4,
        "seed": 42,
        "amp": True,
        "topk": [1],
        "threshold_tprs": [0.75, 0.9, 0.95],
        "score_thresholds": [0.5],
        "max_ref_batches": 0,
        "max_tn_batches": 0,
        "log_every": 50,
        "no_per_example_records": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _saved_args():
    terminal = Path(ref_eval._V41_IMMUTABLE_ARCHIVED_TERMINAL["path"])
    return {
        "output_dir": str(terminal.parent),
        "max_train_iters": 400,
        "stage_b_dense_duty_confidence_expected_optimizer_updates": 400,
        "stage_b_dense_duty_execution_scope": "probe",
        "stage_b_dense_duty_evaluation_scope": "probe",
        "stage_b_dense_duty_phase": "confidence",
        "stage_b_dense_duty_confidence_revision": (
            "word_veto_candidate_asymmetric_confidence_v32"
        ),
        "stage_b_v21_token_edit_query_scope": ROLE_SCOPE,
    }


def test_v41_snapshot_allowlist_and_wrapper_command_are_exact():
    assert set(ref_eval._V41_IMMUTABLE_ARCHIVED_SNAPSHOTS) == {100, 200, 300}
    assert {
        update: spec["sha256"]
        for update, spec in ref_eval._V41_IMMUTABLE_ARCHIVED_SNAPSHOTS.items()
    } == {
        100: "b0f80ec833bd65be8f25f9a6e6360025845f5023555a22ebbfcd38d9fe26ad31",
        200: "c3ccdd5f9cdbc4ed250f4067daa7043e69f23cfee1814784e5ad49a475c44fc2",
        300: "cae68b5cab47f6158e4419ef3afcd56e66b6b5bfd31bc39193621f60395d0808",
    }
    assert ref_eval._V41_IMMUTABLE_ARCHIVED_TERMINAL["sha256"] == (
        "c6dd000dcb10fbfc1aec5c897b64628130fd9213f7c8cbe13b0725c8283d840a"
    )
    command = screen.build_command(200)
    assert "--partial_dense_duty_confidence_diagnostic" in command
    assert "--immutable_v41_archived_snapshot_diagnostic" in command
    assert "--immutable_v39_archived_snapshot_diagnostic" not in command
    assert "--immutable_v40_archived_snapshot_diagnostic" not in command
    assert "--skip_ref" in command
    assert "promotion" not in " ".join(command)
    assert "admission" not in " ".join(command)


def test_v41_immutable_cli_accepts_only_the_fixed_surface():
    cfg = SLConfig.fromfile(str(CONFIG))
    combined_eval._validate_immutable_v41_archived_snapshot_diagnostic_args(
        _args(300), cfg
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"partial_dense_duty_confidence_diagnostic": False}, "also required"),
        ({"skip_ref": False}, "TN-only"),
        ({"ckpts": [str(screen.SNAPSHOTS[100].parent)]}, "exact archived"),
        ({"output_dir": "/tmp/not-the-fixed-v41-screen"}, "fixed immutable"),
        ({"log_every": 10}, "runtime must be"),
        ({"immutable_v39_archived_snapshot_diagnostic": True}, "mutually exclusive"),
        ({"immutable_v40_archived_snapshot_diagnostic": True}, "mutually exclusive"),
    ],
)
def test_v41_immutable_cli_rejects_contract_drift(overrides, message):
    cfg = SLConfig.fromfile(str(CONFIG))
    with pytest.raises(ValueError, match=message):
        combined_eval._validate_immutable_v41_archived_snapshot_diagnostic_args(
            _args(**overrides), cfg
        )


def test_v41_snapshot_and_terminal_metadata_bind_role_complete_scope():
    checkpoint = Path(screen.SNAPSHOTS[100]).resolve(strict=True)
    payload = {
        "optimizer_updates": 100,
        "checkpoint_reason": "interval",
        "args": _saved_args(),
    }
    ref_eval._validate_v41_immutable_archived_snapshot_metadata(
        payload,
        checkpoint_path=checkpoint,
        optimizer_updates=100,
        checkpoint_reason="interval",
    )
    drifted = copy.deepcopy(payload)
    drifted["args"]["stage_b_v21_token_edit_query_scope"] = (
        "target_iou_union_detached_final_confidence_base_argmax_v2"
    )
    with pytest.raises(RuntimeError, match="directory/update/reason"):
        ref_eval._validate_v41_immutable_archived_snapshot_metadata(
            drifted,
            checkpoint_path=checkpoint,
            optimizer_updates=100,
            checkpoint_reason="interval",
        )

    terminal = Path(ref_eval._V41_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal_payload = {
        "optimizer_updates": 400,
        "checkpoint_reason": "max_train_iters",
        "args": _saved_args(),
    }
    ref_eval._validate_v41_immutable_archived_terminal_metadata(
        terminal_payload, checkpoint_path=terminal
    )


def test_v41_file_verifier_rejects_sha_drift_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(ref_eval, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        ref_eval._verify_v41_immutable_archived_diagnostic_files(
            screen.SNAPSHOTS[100]
        )

    target = tmp_path / "checkpoint_iter.pth"
    target.write_bytes(b"checkpoint")
    symlink = tmp_path / "snapshot.pth"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="must not be symlinks"):
        ref_eval._v41_immutable_archived_file_record(symlink)


def test_v41_summary_provenance_uses_only_the_v41_flag(tmp_path, monkeypatch):
    config = tmp_path / "config.py"
    checkpoint = tmp_path / "checkpoint.pth"
    data_root = tmp_path / "data"
    config.write_text("value = 1\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    data_root.mkdir()
    monkeypatch.setattr(
        combined_eval,
        "file_record",
        lambda path: {"path": str(path), "sha256": "1" * 64},
    )
    immutable_provenance = {
        "schema": "pivot.stageb.v41_immutable_archived_diagnostic/v1",
        "optimizer_updates": 100,
        "checkpoint_reason": "interval",
    }
    cfg = SimpleNamespace(
        stage_b_dense_duty=True,
        stage_b_v22_score_ownership="rank_tower_stopgrad_token_adapter_two_phase",
        stage_b_dense_duty_partial_confidence_diagnostic=True,
        stage_b_dense_duty_partial_confidence_diagnostic_optimizer_updates=100,
        stage_b_dense_duty_partial_confidence_diagnostic_expected_optimizer_updates=400,
        stage_b_dense_duty_partial_confidence_diagnostic_checkpoint_reason="interval",
        stage_b_dense_duty_partial_confidence_diagnostic_terminal_checkpoint=False,
        stage_b_dense_duty_immutable_v41_archived_snapshot_diagnostic=True,
        stage_b_dense_duty_immutable_v41_archived_snapshot_audit={
            "immutable_archived_snapshot_diagnostic": True,
            "immutable_archived_snapshot_version": "v41",
            "immutable_archived_provenance": immutable_provenance,
        },
    )
    args = SimpleNamespace(config=str(config), amp=True, device="cuda:0")
    result = combined_eval._evaluation_summary_provenance(
        cfg=cfg,
        args=args,
        checkpoint=checkpoint,
        data_root=data_root,
    )
    assert result["immutable_v41_archived_snapshot_diagnostic"] is True
    assert "immutable_v39_archived_snapshot_diagnostic" not in result
    assert "immutable_v40_archived_snapshot_diagnostic" not in result
    assert result["immutable_archived_snapshot_provenance"] == immutable_provenance


def _summary_row(update: int):
    snapshot = {
        "path": str(Path(screen.SNAPSHOTS[update]).resolve(strict=True)),
        "size_bytes": 1,
        "sha256": ref_eval._V41_IMMUTABLE_ARCHIVED_SNAPSHOTS[update]["sha256"],
    }
    terminal_path = Path(ref_eval._V41_IMMUTABLE_ARCHIVED_TERMINAL["path"])
    terminal = {
        "path": str(terminal_path.resolve(strict=True)),
        "size_bytes": 1,
        "sha256": ref_eval._V41_IMMUTABLE_ARCHIVED_TERMINAL["sha256"],
    }
    provenance = {
        "schema": "pivot.stageb.v41_immutable_archived_diagnostic/v1",
        "optimizer_updates": update,
        "checkpoint_reason": "interval",
    }
    for key in (
        "snapshot_before_validation",
        "snapshot_after_validation",
        "snapshot_after_model_load",
        "snapshot_after_evaluation",
    ):
        provenance[key] = dict(snapshot)
    for key in (
        "terminal_before_validation",
        "terminal_after_validation",
        "terminal_after_model_load",
        "terminal_after_evaluation",
    ):
        provenance[key] = dict(terminal)
    return {
        "diagnostic_only": True,
        "formal_gate_eligible": False,
        "confidence_evaluated": True,
        "terminal_checkpoint": False,
        "immutable_v41_archived_snapshot_diagnostic": True,
        "optimizer_updates": update,
        "expected_optimizer_updates": 400,
        "remaining_optimizer_updates": 400 - update,
        "checkpoint_reason": "interval",
        "num_pairs": 1607,
        "immutable_archived_snapshot_provenance": provenance,
    }


def test_v41_summary_verifier_rejects_old_formal_or_terminal_evidence(
    tmp_path, monkeypatch
):
    update = 100
    output = tmp_path / "screen"
    output.mkdir()
    monkeypatch.setitem(screen.OUTPUTS, update, output)
    row = _summary_row(update)
    summary = output / "summary.json"
    summary.write_text(
        json.dumps({"refcoco": [], "tn": [row]}), encoding="utf-8"
    )
    accepted = screen.verify_summary(update)
    assert accepted["diagnostic_only"] is True

    for key, value in (
        ("formal_gate_eligible", True),
        ("terminal_checkpoint", True),
        ("immutable_v39_archived_snapshot_diagnostic", True),
        ("immutable_v40_archived_snapshot_diagnostic", True),
    ):
        drifted = copy.deepcopy(row)
        drifted[key] = value
        summary.write_text(
            json.dumps({"refcoco": [], "tn": [drifted]}), encoding="utf-8"
        )
        with pytest.raises(screen.SnapshotScreenError):
            screen.verify_summary(update)
