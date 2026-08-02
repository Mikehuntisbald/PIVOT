from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.eval_stageb_gdino_adapter_o64_direct_rank as legacy
import tools.eval_stageb_native_residual_fresh_o64 as fresh
from util.slconfig import SLConfig


SEALED_LEGACY_EVALUATOR_SHA256 = (
    "3869b769924b43d2f17584f1d4b705af8cc8894db82326166c657ce0bb522520"
)
FRESH_CONFIG = Path(
    "config/ablations/"
    "cfg_stageb_gdino_score_adapter_rank_fresh_o64_aspect_b32a2_u500.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_fresh_evaluator_reuses_sealed_lineage_tensor_forward_and_optimizer_audits():
    legacy_path = Path(legacy.__file__).resolve()
    assert _sha256(legacy_path) == SEALED_LEGACY_EVALUATOR_SHA256
    assert fresh.audit_b58_lineage is legacy.audit_b58_lineage
    assert fresh.audit_tensor_isolation is legacy.audit_tensor_isolation
    assert fresh.audit_batch_outputs is legacy.audit_batch_outputs
    assert fresh.aggregate_o64_records is legacy.aggregate_o64_records
    assert fresh._audit_optimizer is legacy._audit_optimizer


def test_actual_fresh_config_is_exact_v2_aspect_preserving_rank_only_contract():
    cfg = SLConfig.fromfile(str(FRESH_CONFIG))
    contract = fresh.validate_config(cfg)
    assert contract["stage_b_native_residual_contract_version"] == 2
    assert contract["stage_b_gdino_adapter_train_mode"] == "rank_only"
    assert contract["batch_size"] == 32
    assert contract["fix_size"] is False
    assert contract["data_aug_train_deterministic_aspect_resize"] is True
    assert contract["strong_aug"] is False
    assert contract["data_aug_hflip_prob"] == 0.0
    assert contract["data_aug_scales"] == [800]
    assert contract["data_aug_max_size"] == 1333


def test_initializer_validation_reopens_fresh_config_and_b58_bindings(
    monkeypatch, tmp_path
):
    initializer = tmp_path / "initializer.pth"
    config = tmp_path / "fresh_config.py"
    b58 = tmp_path / "b58.pth"
    calls = []

    def validate_payload(model, payload, *, checkpoint_label):
        calls.append(("payload", model, payload, checkpoint_label))

    def verify_bindings(payload, *, config, b58_path, checkpoint_label):
        calls.append(("external", payload, config, b58_path, checkpoint_label))

    monkeypatch.setattr(fresh, "validate_initializer_payload", validate_payload)
    monkeypatch.setattr(fresh, "verify_external_bindings", verify_bindings)
    model = torch.nn.Linear(1, 1)
    payload = {"model": {}}
    fresh.validate_fresh_initializer(
        model,
        payload,
        initializer_path=initializer,
        config_path=config,
        b58_path=b58,
    )
    assert calls[0][:3] == ("payload", model, payload)
    assert calls[1][0:4] == ("external", payload, config, b58)
    assert str(initializer) in calls[0][3]
    assert calls[0][3] == calls[1][4]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("stage_b_native_residual_contract_version", 1),
        ("stage_b_gdino_adapter_train_mode", "confidence_only"),
        ("batch_size", 64),
        ("fix_size", True),
        ("data_aug_train_deterministic_aspect_resize", False),
        ("strong_aug", True),
        ("data_aug_hflip_prob", 0.5),
        ("data_aug_scales", [480, 800]),
        ("data_aug_max_size", 1600),
    ),
)
def test_config_audit_fails_closed_on_v2_resize_or_rank_drift(field, bad_value):
    values = fresh._expected_config_contract()
    values[field] = bad_value
    with pytest.raises(fresh.FreshO64AuditError, match="config drifted"):
        fresh.validate_config(SimpleNamespace(**values))


def _dataset_config(annotation: Path, *, strong_aug: bool = False) -> dict:
    return {
        "artifact_binding": {
            "manifest": {
                "path": str(annotation),
                "sha256": fresh.EXPECTED_FRESH_MANIFEST_SHA256,
                "rows": 128,
            },
            "receipt": {
                "path": str(annotation.parent / "receipt.json"),
                "sha256": fresh.EXPECTED_FRESH_RECEIPT_SHA256,
                "schema": fresh.OUTPUT_RECEIPT_SCHEMA,
            },
        },
        "train": [
            {
                "dataset_mode": "odvg",
                "root": "/",
                "anno": str(annotation),
                "mix_weight": 1.0,
                "strong_aug": strong_aug,
            }
        ],
        "val": [],
    }


def test_dataset_entry_requires_explicit_strong_aug_false_and_fresh_manifest(tmp_path):
    annotation = tmp_path / fresh.OUTPUT_MANIFEST
    annotation.write_text("{}\n", encoding="ascii")
    (tmp_path / "receipt.json").write_text("{}\n", encoding="ascii")
    dataset_path = tmp_path / "datasets.json"
    dataset_path.write_text(
        json.dumps(_dataset_config(annotation), sort_keys=True), encoding="ascii"
    )
    loaded = fresh._load_dataset_config(dataset_path)
    assert loaded["train"][0]["strong_aug"] is False

    dataset_path.write_text(
        json.dumps(_dataset_config(annotation, strong_aug=True), sort_keys=True),
        encoding="ascii",
    )
    with pytest.raises(fresh.FreshO64AuditError, match="strong_aug=false"):
        fresh._load_dataset_config(dataset_path)


def test_dataset_artifact_binding_is_not_inert_metadata(tmp_path):
    annotation = tmp_path / fresh.OUTPUT_MANIFEST
    annotation.write_text("{}\n", encoding="ascii")
    (tmp_path / "receipt.json").write_text("{}\n", encoding="ascii")
    value = _dataset_config(annotation)
    value["artifact_binding"]["receipt"]["sha256"] = "0" * 64
    dataset_path = tmp_path / "datasets.json"
    dataset_path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")
    with pytest.raises(fresh.FreshO64AuditError, match="artifact binding drifted"):
        fresh._load_dataset_config(dataset_path)


def _fresh_rows() -> list[dict]:
    rows = []
    for pair_index in range(fresh.EXPECTED_PAIRS):
        pair_id = f"{pair_index + 1:064x}"
        image_id = 20_000 + pair_index
        for direction_index, direction in enumerate(("anchor", "partner")):
            target_id = 30_000 + 2 * pair_index + direction_index
            rows.append(
                {
                    "direction": direction,
                    "filename": f"images/{image_id}.jpg",
                    "grounding": {
                        "regions": [
                            {
                                "bbox": [10.0, 20.0, 30.0, 40.0],
                                "phrase": f"target {target_id}",
                            }
                        ]
                    },
                    "image_id": image_id,
                    "pair_index": pair_index,
                    "row_schema": fresh.OUTPUT_ROW_SCHEMA,
                    "source_assignment_line_number": pair_index + 1,
                    "source_assignment_manifest": "refcoco_train.jsonl",
                    "source_member_pair_id": pair_id,
                    "source_priority_sha256": f"{10_000 + pair_index:064x}",
                    "source_row_sha256": f"{40_000 + pair_index:064x}",
                    "target_coco_ann_id": target_id,
                }
            )
    return rows


def test_fresh_rows_require_order_and_retain_attribution_identity():
    metadata = fresh.validate_fresh_o64_rows(_fresh_rows())
    assert len(metadata) == fresh.EXPECTED_ROWS == 128
    assert metadata[0] == {
        "row_index": 0,
        "pair_index": 0,
        "direction": "anchor",
        "image_id": 20_000,
        "filename": "images/20000.jpg",
        "expression": "target 30000",
        "target_bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
        "target_coco_ann_id": 30_000,
        "source_assignment_manifest": "refcoco_train.jsonl",
        "source_assignment_line_number": 1,
        "source_member_pair_id": f"{1:064x}",
        "source_priority_sha256": f"{10_000:064x}",
        "source_row_sha256": f"{40_000:064x}",
    }
    rows = _fresh_rows()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(fresh.FreshO64AuditError, match="pair order"):
        fresh.validate_fresh_o64_rows(rows)


def _terminal_payload(root: Path):
    config = root / "config.py"
    datasets = root / "datasets.json"
    initializer = root / "initializer.pth"
    output = root / "run"
    output.mkdir()
    checkpoint = output / "checkpoint_iter.pth"
    for path in (config, datasets, initializer, checkpoint):
        path.write_bytes(b"fixture")
    params = list(range(fresh.EXPECTED_RANK_TENSORS))
    state = {
        index: {
            "step": torch.tensor(500.0),
            "exp_avg": torch.ones(1),
            "exp_avg_sq": torch.ones(1),
        }
        for index in params
    }
    args = {
        **fresh._expected_config_contract(),
        "config_file": str(config),
        "datasets": str(datasets),
        "pretrain_model_path": str(initializer),
        "resume": "",
        "output_dir": str(output),
        "max_train_iters": 500,
        "gradient_accumulation_steps": 2,
        "amp": True,
        "stage_b_u0_patch_rank": False,
        "stage_b_data_driven_score": False,
    }
    payload = {
        "optimizer_updates": 500,
        "checkpoint_reason": "max_train_iters",
        "epoch": 249,
        "iteration": 3,
        "epoch_finished": True,
        "args": args,
        "criterion": {
            "criterion_train_mode_code": torch.tensor(1),
            "criterion_scope_code": torch.tensor(0),
            "criterion_queue_size": torch.tensor(0),
            "criterion_queue_min_count": torch.tensor(0),
        },
        "optimizer": {
            "param_groups": [
                {
                    "params": params,
                    "lr": 3.0e-4,
                    "stage_b_gdino_branch": "rank",
                }
            ],
            "state": state,
        },
        "lr_scheduler": {},
        "scaler": {},
        "rng_state": {},
    }
    paths = {
        "config_path": config.resolve(),
        "dataset_path": datasets.resolve(),
        "initializer_path": initializer.resolve(),
        "checkpoint_path": checkpoint.resolve(),
        "loader_batches": 4,
    }
    return payload, paths


def test_u500_checkpoint_audits_v2_transform_and_legacy_optimizer(tmp_path):
    payload, paths = _terminal_payload(tmp_path)
    audit = fresh.audit_training_checkpoint(payload, **paths)
    assert audit["optimizer_updates"] == 500
    assert audit["derived_terminal_epoch"] == 249
    assert audit["train_micro_batch_size"] == 32
    assert audit["train_gradient_accumulation_steps"] == 2
    assert audit["train_effective_batch_size"] == 64
    assert audit["optimizer"]["parameter_states"] == 8

    payload["args"]["data_aug_train_deterministic_aspect_resize"] = False
    with pytest.raises(fresh.FreshO64AuditError, match="saved args drifted"):
        fresh.audit_training_checkpoint(payload, **paths)


def test_u500_checkpoint_rejects_nonterminal_optimizer_step(tmp_path):
    payload, paths = _terminal_payload(tmp_path)
    payload["optimizer"]["state"][0]["step"] = torch.tensor(499.0)
    with pytest.raises(fresh.FreshO64AuditError, match="expected 500"):
        fresh.audit_training_checkpoint(payload, **paths)


def test_identity_and_u500_attribution_output_requires_all_128_ordered_records():
    records = []
    for metadata in fresh.validate_fresh_o64_rows(_fresh_rows()):
        records.append(
            {
                **metadata,
                "base_winner_query": 0,
                "adapted_winner_query": 1,
                "base_top1_iou": 0.25,
                "adapted_top1_iou": 0.75,
                "base_correct50": False,
                "adapted_correct50": True,
            }
        )
    output = fresh._validated_attribution_records(records)
    assert len(output) == 128
    assert output[0]["expression"] == "target 30000"
    assert output[-1]["row_index"] == 127
    with pytest.raises(fresh.FreshO64AuditError, match="needs 128"):
        fresh._validated_attribution_records(records[:-1])


def test_published_hash_constants_must_be_literal_lowercase_sha256():
    for value in (
        fresh.EXPECTED_FRESH_MANIFEST_SHA256,
        fresh.EXPECTED_FRESH_RECEIPT_SHA256,
    ):
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")
