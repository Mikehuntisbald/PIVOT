from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.eval_stageb_gdino_adapter_o64_direct_rank import (
    ADAPTER_PREFIX,
    EXPECTED_BASE_TENSORS,
    EXPECTED_CONFIDENCE_TENSORS,
    EXPECTED_PAIRS,
    EXPECTED_QUERIES,
    EXPECTED_RANK_TENSORS,
    EXPECTED_ROWS,
    O64DirectRankAuditError,
    OUTPUT_ROW_SCHEMA,
    aggregate_o64_records,
    audit_batch_outputs,
    audit_tensor_isolation,
    audit_training_checkpoint,
    validate_o64_rows,
)


RANK_NAMES = (
    "rank_norm.weight",
    "rank_norm.bias",
    "rank_trunk.0.weight",
    "rank_trunk.0.bias",
    "rank_trunk.2.weight",
    "rank_trunk.2.bias",
    "rank_output.weight",
    "rank_output.bias",
)
CONFIDENCE_NAMES = (
    "confidence_norm.weight",
    "confidence_norm.bias",
    "confidence_trunk.0.weight",
    "confidence_trunk.0.bias",
    "confidence_trunk.2.weight",
    "confidence_trunk.2.bias",
    "confidence_gate.0.weight",
    "confidence_gate.0.bias",
    "confidence_gate.2.weight",
    "confidence_gate.2.bias",
    "confidence_gate.4.weight",
    "confidence_gate.4.bias",
)


def _initializer_and_checkpoint():
    base = [f"base.tensor.{index:03d}" for index in range(EXPECTED_BASE_TENSORS)]
    rank = [ADAPTER_PREFIX + name for name in RANK_NAMES]
    confidence = [ADAPTER_PREFIX + name for name in CONFIDENCE_NAMES]
    assert len(rank) == EXPECTED_RANK_TENSORS
    assert len(confidence) == EXPECTED_CONFIDENCE_TENSORS
    state = {
        key: torch.tensor([float(index)], dtype=torch.float32)
        for index, key in enumerate(base + rank + confidence)
    }
    initializer = {
        "model": state,
        "native_residual_initializer": {
            "role_keys": {
                "b58_base": base,
                "random_identity_adapter": rank + confidence,
            }
        },
    }
    trained_state = {key: value.clone() for key, value in state.items()}
    for key in rank:
        trained_state[key].add_(1.0)
    return initializer, {"model": trained_state}, base, rank, confidence


def test_tensor_isolation_allows_exactly_eight_rank_changes():
    initializer, checkpoint, base, rank, confidence = _initializer_and_checkpoint()
    audit = audit_tensor_isolation(initializer, checkpoint, identity=False)
    assert audit["base_tensor_count"] == len(base) == 938
    assert audit["rank_tensor_count"] == len(rank) == 8
    assert audit["confidence_tensor_count"] == len(confidence) == 12
    assert audit["changed_rank_tensors"] == 8
    assert audit["changed_base_tensors"] == 0
    assert audit["changed_confidence_tensors"] == 0

    checkpoint["model"][base[0]].add_(1.0)
    with pytest.raises(O64DirectRankAuditError, match="frozen b58"):
        audit_tensor_isolation(initializer, checkpoint, identity=False)


def test_tensor_isolation_rejects_confidence_or_partial_rank_and_supports_identity():
    initializer, checkpoint, _base, rank, confidence = _initializer_and_checkpoint()
    checkpoint["model"][confidence[0]].add_(1.0)
    with pytest.raises(O64DirectRankAuditError, match="frozen confidence"):
        audit_tensor_isolation(initializer, checkpoint, identity=False)

    initializer, checkpoint, _base, rank, _confidence = _initializer_and_checkpoint()
    checkpoint["model"][rank[0]] = initializer["model"][rank[0]].clone()
    with pytest.raises(O64DirectRankAuditError, match="change count"):
        audit_tensor_isolation(initializer, checkpoint, identity=False)

    identity = {"model": {key: value.clone() for key, value in initializer["model"].items()}}
    audit = audit_tensor_isolation(initializer, identity, identity=True)
    assert audit["full_model_bitwise_equal_initializer"] is True
    assert audit["changed_rank_tensors"] == 0


def _terminal_payload(root: Path):
    config = root / "config.py"
    datasets = root / "datasets.json"
    initializer = root / "initializer.pth"
    output = root / "run"
    output.mkdir()
    checkpoint = output / "checkpoint_iter.pth"
    for path in (config, datasets, initializer, checkpoint):
        path.write_bytes(b"fixture")
    params = list(range(EXPECTED_RANK_TENSORS))
    optimizer_state = {
        index: {
            "step": torch.tensor(500.0),
            "exp_avg": torch.ones(1),
            "exp_avg_sq": torch.ones(1),
        }
        for index in params
    }
    args = {
        "config_file": str(config),
        "datasets": str(datasets),
        "pretrain_model_path": str(initializer),
        "resume": "",
        "output_dir": str(output),
        "stage_b_native_residual_data_only": True,
        "stage_b_native_residual_contract_version": 1,
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_tn_scope": "",
        "stage_b_gdino_rank_weight": 1.0,
        "stage_b_gdino_confidence_weight": 0.0,
        "stage_b_gdino_paired_margin_weight": 0.0,
        "stage_b_gdino_queue_size": 0,
        "stage_b_gdino_queue_min_count": 0,
        "stage_b_gdino_rank_lr": 3.0e-4,
        "lr": 3.0e-4,
        "batch_size": 32,
        "epochs": 250,
        "max_train_iters": 500,
        "lr_drop": 1000,
        "fix_size": True,
        "data_aug_hflip_prob": 0.0,
        "data_aug_scales": [800],
        "data_aug_max_size": 1333,
        "gradient_accumulation_steps": 2,
        "amp": True,
        "enable_patch_branch": False,
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
            "state": optimizer_state,
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


def test_training_checkpoint_audits_saved_args_and_all_500_optimizer_steps(tmp_path):
    payload, paths = _terminal_payload(tmp_path)
    audit = audit_training_checkpoint(payload, **paths)
    assert audit["optimizer_updates"] == 500
    assert audit["derived_terminal_epoch"] == 249
    assert audit["optimizer_updates_per_epoch"] == 2
    assert audit["train_micro_batch_size"] == 32
    assert audit["train_gradient_accumulation_steps"] == 2
    assert audit["train_effective_batch_size"] == 64
    assert audit["optimizer"]["parameter_states"] == 8

    payload["optimizer"]["state"][0]["step"] = torch.tensor(499.0)
    with pytest.raises(O64DirectRankAuditError, match="expected 500"):
        audit_training_checkpoint(payload, **paths)


def _o64_rows():
    rows = []
    for pair_index in range(EXPECTED_PAIRS):
        pair_id = f"{pair_index + 1:064x}"
        for direction_index, direction in enumerate(("anchor", "partner")):
            rows.append(
                {
                    "row_schema": OUTPUT_ROW_SCHEMA,
                    "pair_index": pair_index,
                    "direction": direction,
                    "source_member_pair_id": pair_id,
                    "target_coco_ann_id": 10_000 + 2 * pair_index + direction_index,
                }
            )
    return rows


def test_o64_rows_require_exact_128_anchor_partner_order():
    metadata = validate_o64_rows(_o64_rows())
    assert len(metadata) == EXPECTED_ROWS
    assert metadata[0]["direction"] == "anchor"
    assert metadata[-1]["pair_index"] == 63

    rows = _o64_rows()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(O64DirectRankAuditError, match="pair order"):
        validate_o64_rows(rows)


def _batch_outputs():
    batch_size = 4
    boxes = torch.zeros(batch_size, EXPECTED_QUERIES, 4)
    boxes[:, 0] = torch.tensor([0.1, 0.1, 0.1, 0.1])
    boxes[:, 1] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    base = torch.zeros(batch_size, EXPECTED_QUERIES)
    rank = torch.zeros_like(base)
    base[0, 0], rank[0, 1] = 2.0, 2.0
    base[1, 1], rank[1, 1] = 2.0, 2.0
    base[2, 1], rank[2, 0] = 2.0, 2.0
    base[3, 0], rank[3, 0] = 2.0, 2.0
    return {
        "stage_b_gdino_base_score": base,
        "stage_b_gdino_rank_residual": rank - base,
        "stage_b_gdino_rank_score": rank,
        "pred_boxes": boxes,
    }


def test_forward_metrics_and_identity_are_computed_from_native_scores():
    outputs = _batch_outputs()
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])} for _ in range(4)]
    metadata = [
        {"row_index": index, "pair_index": index // 2, "direction": ("anchor", "partner")[index % 2]}
        for index in range(4)
    ]
    records, checks = audit_batch_outputs(
        outputs, targets, metadata, identity=False
    )
    assert checks["rank_score_bitwise_equals_base"] is False
    assert [record["base_correct50"] for record in records] == [False, True, True, False]
    assert [record["adapted_correct50"] for record in records] == [True, True, False, False]

    identity_outputs = dict(outputs)
    identity_outputs["stage_b_gdino_rank_score"] = outputs[
        "stage_b_gdino_base_score"
    ].clone()
    identity_outputs["stage_b_gdino_rank_residual"] = torch.zeros_like(
        outputs["stage_b_gdino_base_score"]
    )
    _records, identity_checks = audit_batch_outputs(
        identity_outputs, targets, metadata, identity=True
    )
    assert all(identity_checks.values())

    identity_outputs["stage_b_gdino_rank_residual"][0, 0] = 1.0
    with pytest.raises(O64DirectRankAuditError, match="identity forward"):
        audit_batch_outputs(identity_outputs, targets, metadata, identity=True)


def test_aggregate_reports_acc_fixes_regressions_and_bidirectional_pairs():
    records = []
    for row in validate_o64_rows(_o64_rows()):
        index = row["row_index"]
        base = index not in {0, 3}
        adapted = index not in {2, 3}
        records.append(
            {
                **row,
                "base_correct50": base,
                "adapted_correct50": adapted,
            }
        )
    metrics = aggregate_o64_records(records)
    assert metrics["base_correct50"] == 126
    assert metrics["adapted_correct50"] == 126
    assert metrics["wrong_fixed"] == 1
    assert metrics["correct_regressed"] == 1
    assert metrics["base_bidirectional_correct_pairs"] == 62
    assert metrics["adapted_bidirectional_correct_pairs"] == 63
