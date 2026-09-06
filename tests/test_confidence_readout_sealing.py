"""Lightweight postflight gate tests; no real checkpoint or benchmark access."""
import json
from pathlib import Path

import pytest

from tools.seal_confidence_readout_heads import bind, check_postflight, publish_json


def fixture_postflight(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"study_protocol": bind(protocol)}))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint for binding only")
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"synthetic initialization")
    arms = ["native_selected__exists", "native_selected__emit"]
    post = {"schema": "arrow.confidence_readout.training_postflight/v1", "status": "complete",
        "seed": 17, "localizer": "mmgdino_positive", "arms": arms,
        "updates_per_head": 12575, "epochs": 5, "no_optimizer_skips": True, "no_amp": True,
        "validation_model_evaluations": 0, "test_forwards": 0, "gref_forwards": 0,
        "checkpoint": bind(checkpoint), "design": bind(design), "initial_state": bind(initial),
        "history": [{"epoch": epoch, "updates": 2515 * epoch, "amp_skips": 0, "nonfinite": 0}
                    for epoch in range(1, 6)],
        "ownership": {arm: {"confidence_tensors": 8, "confidence_parameters": 50179} for arm in arms}}
    path = tmp_path / "postflight.json"
    path.write_text(json.dumps(post))
    return path, bind(protocol), post


def test_complete_endpoint_passes(tmp_path):
    path, protocol, post = fixture_postflight(tmp_path)
    assert check_postflight(path, protocol, "mmgdino_positive", 17) == post


@pytest.mark.parametrize("key,value", [("status", "running"), ("updates_per_head", 12574),
    ("seed", 42), ("gref_forwards", 1), ("test_forwards", 1), ("no_amp", False)])
def test_incomplete_or_altered_endpoint_rejected(tmp_path, key, value):
    path, protocol, post = fixture_postflight(tmp_path)
    post[key] = value
    path.write_text(json.dumps(post))
    with pytest.raises(ValueError):
        check_postflight(path, protocol, "mmgdino_positive", 17)


def test_hash_drift_rejected(tmp_path):
    path, protocol, post = fixture_postflight(tmp_path)
    Path(post["checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        check_postflight(path, protocol, "mmgdino_positive", 17)


def test_optimizer_epoch_skip_rejected(tmp_path):
    path, protocol, post = fixture_postflight(tmp_path)
    post["history"][2]["updates"] -= 1
    path.write_text(json.dumps(post))
    with pytest.raises(ValueError, match="health/update"):
        check_postflight(path, protocol, "mmgdino_positive", 17)


def test_wrong_capacity_rejected(tmp_path):
    path, protocol, post = fixture_postflight(tmp_path)
    post["ownership"]["native_selected__emit"]["confidence_parameters"] += 1
    path.write_text(json.dumps(post))
    with pytest.raises(ValueError, match="capacity"):
        check_postflight(path, protocol, "mmgdino_positive", 17)


def test_atomic_publication_does_not_replace_existing_seal(tmp_path):
    path = tmp_path / "gate.json"
    publish_json(path, {"status": "complete"})
    before = path.read_bytes()
    assert json.loads(before) == {"status": "complete"}
    assert not path.with_name("gate.json.partial").exists()
    with pytest.raises(FileExistsError):
        publish_json(path, {"status": "changed"})
    assert path.read_bytes() == before
