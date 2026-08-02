import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools import (
    run_stageb_confidence_adapter_candidate_gate_zero_offset_snapshot_screen
    as screen,
)
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = combined_eval._CANDIDATE_GATE_ZERO_OFFSET_CONFIDENCE_U0400_CONFIG


def _args(update: int = 100, **overrides):
    values = {
        "immutable_v39_archived_snapshot_diagnostic": True,
        "partial_dense_duty_rank_diagnostic": False,
        "partial_dense_duty_confidence_diagnostic": True,
        "config": str(CONFIG),
        "output_dir": str(
            combined_eval._V39_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS[update]
        ),
        "ckpts": [
            str(combined_eval._V39_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS[update])
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
    terminal = Path(ref_eval._V39_IMMUTABLE_ARCHIVED_TERMINAL["path"])
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
    }


def test_v39_snapshot_allowlist_and_wrapper_command_are_exact():
    assert set(ref_eval._V39_IMMUTABLE_ARCHIVED_SNAPSHOTS) == {100, 200, 300}
    assert {
        update: spec["sha256"]
        for update, spec in ref_eval._V39_IMMUTABLE_ARCHIVED_SNAPSHOTS.items()
    } == {
        100: "0f38783c015d544a9ff9f6ede47a7d66ac158d791a2f6a7f70cef2896fb3b584",
        200: "b091b29b9076c933a4190c308ba041f8af0227da3275ef6bba4aca005b8286af",
        300: "82822bc271950f1cdb31f791102b414d0ce8a986974a93ecc728645f6a409ab4",
    }
    assert ref_eval._V39_IMMUTABLE_ARCHIVED_TERMINAL["sha256"] == (
        "202b067beb7cdd71343599872a1ae911b45bbb7f375b168739dab659b611c6c0"
    )
    command = screen.build_command(200)
    assert "--partial_dense_duty_confidence_diagnostic" in command
    assert "--immutable_v39_archived_snapshot_diagnostic" in command
    assert "--skip_ref" in command
    assert "--skip_tn" not in command
    assert "promotion" not in " ".join(command)
    assert "admission" not in " ".join(command)


def test_immutable_cli_requires_existing_partial_diagnostic_flag():
    cfg = SLConfig.fromfile(str(CONFIG))
    with pytest.raises(ValueError, match="partial_dense_duty_confidence"):
        combined_eval._validate_immutable_v39_archived_snapshot_diagnostic_args(
            _args(partial_dense_duty_confidence_diagnostic=False), cfg
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"skip_ref": False}, "TN-only"),
        ({"ckpts": [str(screen.SNAPSHOTS[100].parent)]}, "exact archived"),
        ({"output_dir": "/tmp/not-the-fixed-screen"}, "fixed immutable"),
        ({"log_every": 10}, "runtime must be"),
    ],
)
def test_immutable_cli_rejects_scope_path_and_runtime_drift(overrides, message):
    cfg = SLConfig.fromfile(str(CONFIG))
    with pytest.raises(ValueError, match=message):
        combined_eval._validate_immutable_v39_archived_snapshot_diagnostic_args(
            _args(**overrides), cfg
        )


def test_immutable_cli_contract_accepts_only_the_fixed_v39_surface():
    cfg = SLConfig.fromfile(str(CONFIG))
    combined_eval._validate_immutable_v39_archived_snapshot_diagnostic_args(
        _args(300), cfg
    )


def test_snapshot_metadata_rejects_update_and_reason_drift():
    checkpoint = Path(screen.SNAPSHOTS[100]).resolve(strict=True)
    payload = {
        "optimizer_updates": 100,
        "checkpoint_reason": "interval",
        "args": _saved_args(),
    }
    ref_eval._validate_v39_immutable_archived_snapshot_metadata(
        payload,
        checkpoint_path=checkpoint,
        optimizer_updates=100,
        checkpoint_reason="interval",
    )
    drifted = copy.deepcopy(payload)
    drifted["optimizer_updates"] = 101
    with pytest.raises(RuntimeError, match="directory/update/reason"):
        ref_eval._validate_v39_immutable_archived_snapshot_metadata(
            drifted,
            checkpoint_path=checkpoint,
            optimizer_updates=100,
            checkpoint_reason="interval",
        )
    drifted = copy.deepcopy(payload)
    drifted["checkpoint_reason"] = "signal"
    with pytest.raises(RuntimeError, match="directory/update/reason"):
        ref_eval._validate_v39_immutable_archived_snapshot_metadata(
            drifted,
            checkpoint_path=checkpoint,
            optimizer_updates=100,
            checkpoint_reason="interval",
        )


def test_terminal_metadata_requires_u400_max_train_iters():
    terminal = Path(ref_eval._V39_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    payload = {
        "optimizer_updates": 400,
        "checkpoint_reason": "max_train_iters",
        "args": _saved_args(),
    }
    ref_eval._validate_v39_immutable_archived_terminal_metadata(
        payload, checkpoint_path=terminal
    )
    payload["args"]["max_train_iters"] = 401
    with pytest.raises(RuntimeError, match="terminal U400"):
        ref_eval._validate_v39_immutable_archived_terminal_metadata(
            payload, checkpoint_path=terminal
        )


def test_file_verifier_rejects_snapshot_and_terminal_sha_drift(monkeypatch):
    monkeypatch.setattr(ref_eval, "_sha256_file", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        ref_eval._verify_v39_immutable_archived_diagnostic_files(
            screen.SNAPSHOTS[100]
        )


def test_immutable_file_record_rejects_symlink(tmp_path):
    target = tmp_path / "checkpoint_iter.pth"
    target.write_bytes(b"checkpoint")
    symlink = tmp_path / "snapshot.pth"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="must not be symlinks"):
        ref_eval._v39_immutable_archived_file_record(symlink)


def _summary_row(update: int):
    snapshot = {
        "path": str(Path(screen.SNAPSHOTS[update]).resolve(strict=True)),
        "size_bytes": 1,
        "sha256": ref_eval._V39_IMMUTABLE_ARCHIVED_SNAPSHOTS[update]["sha256"],
    }
    terminal_path = Path(ref_eval._V39_IMMUTABLE_ARCHIVED_TERMINAL["path"])
    terminal = {
        "path": str(terminal_path.resolve(strict=True)),
        "size_bytes": 1,
        "sha256": ref_eval._V39_IMMUTABLE_ARCHIVED_TERMINAL["sha256"],
    }
    provenance = {
        "schema": "pivot.stageb.v39_immutable_archived_diagnostic/v1",
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
        "immutable_v39_archived_snapshot_diagnostic": True,
        "optimizer_updates": update,
        "expected_optimizer_updates": 400,
        "remaining_optimizer_updates": 400 - update,
        "checkpoint_reason": "interval",
        "num_pairs": 1607,
        "immutable_archived_snapshot_provenance": provenance,
    }


def test_summary_verifier_never_accepts_formal_or_terminal_evidence(
    tmp_path, monkeypatch
):
    update = 100
    output = tmp_path / "screen"
    output.mkdir()
    monkeypatch.setitem(screen.OUTPUTS, update, output)
    row = _summary_row(update)
    (output / "summary.json").write_text(
        json.dumps({"refcoco": [], "tn": [row]}), encoding="utf-8"
    )
    assert screen.verify_summary(update)["diagnostic_only"] is True
    row["formal_gate_eligible"] = True
    (output / "summary.json").write_text(
        json.dumps({"refcoco": [], "tn": [row]}), encoding="utf-8"
    )
    with pytest.raises(screen.SnapshotScreenError):
        screen.verify_summary(update)
