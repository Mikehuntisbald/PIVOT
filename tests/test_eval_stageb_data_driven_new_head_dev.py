import copy
import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tools.eval_stageb_data_driven_new_head_dev import (
    EXPRESSION_RECORD_SCHEMA,
    EVALUATION_DATA_VARIANT,
    EVALUATION_CONFIG_CHECKPOINT_EQUAL_KEYS,
    FORMAL_NEW_HEAD_CONTRACT,
    FORMAL_NEW_HEAD_BINDING_SCHEMA,
    FORMAL_NEW_HEAD_EXECUTION_SCOPE,
    FORMAL_NEW_HEAD_OPTIMIZER_UPDATES,
    IDENTITY_KEYS,
    MANIFEST_NAMES,
    PAIRED_TRAINING_EQUAL_KEYS,
    PATCH_PROJECTION_KEYS,
    PARTITION_RECEIPT_SCHEMA,
    PARTITION_RECEIPT_REQUIRED_INVARIANTS,
    PARTITIONS,
    SOURCE_MANIFESTS,
    SOURCE_NAMES,
    SUPPORT_RECEIPT_REQUIRED_INVARIANTS,
    SUPPORT_RECEIPT_SCHEMA,
    VARIANTS,
    ManifestRow,
    NewHeadDevEvalError,
    _canonical_bytes,
    _direct_support_pool_content_record,
    _audit_frozen_initializer_tensors,
    _is_sealed_formal_checkpoint_contract,
    _new_head_execution_status,
    _paired_training_contract,
    _shared_training_provenance,
    _training_partition_status,
    _validate_eval_config_training_contract,
    effective_eval_contract,
    evaluate_rank_only_batch,
    load_partition_binding,
    load_shared_evaluation_binding,
    load_support_binding,
    paired_cluster_bootstrap,
    query_image_content_record,
    summarize_records,
    validate_rank_only_runtime,
    data_driven_tensor_state_sha256,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_row(source: str, *, sent_id: int = 7) -> dict:
    return {
        "source": f"{source}_unc_train",
        "image_id": 123,
        "ann_id": 456,
        "ref_id": 789,
        "sent_id": sent_id,
        "split": "train",
        "filename": (
            "/legacy/COCO/coco2014/train2014/"
            "COCO_train2014_000000000123.jpg"
        ),
        "instances": [
            {
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "class_id": 1,
                "raw_phrase": "object",
            }
        ],
    }


def _manifest_row(source: str = "refcoco") -> ManifestRow:
    row = _identity_row(source)
    identity = {key: row[key] for key in IDENTITY_KEYS}
    return ManifestRow(
        identity=tuple(identity[key] for key in IDENTITY_KEYS),
        identity_object=identity,
        coco_split="train2014",
        image_id=123,
    )


def _target(row: ManifestRow, boxes: torch.Tensor, primary: torch.Tensor) -> dict:
    identity = row.identity_object
    return {
        "boxes": boxes,
        "primary_instance_mask": primary,
        "image_id": torch.tensor([identity["image_id"]], dtype=torch.int64),
        "ann_id": torch.tensor([identity["ann_id"]], dtype=torch.int64),
        "ref_id": torch.tensor([identity["ref_id"]], dtype=torch.int64),
        "sent_id": torch.tensor([identity["sent_id"]], dtype=torch.int64),
        "dataset_name": identity["source"],
    }


def _outputs(
    *, query_count: int = 900, all_match: bool = False, text_delta: float = 0.0
) -> dict:
    score = torch.zeros((1, query_count), dtype=torch.float32)
    boxes = torch.tensor([0.8, 0.8, 0.1, 0.1], dtype=torch.float32).repeat(
        1, query_count, 1
    )
    if all_match:
        boxes[:] = torch.tensor([0.2, 0.2, 0.2, 0.2])
    elif query_count:
        boxes[0, 0] = torch.tensor([0.2, 0.2, 0.2, 0.2])
    return {
        "stage_b_data_driven_rank_score": score,
        "stage_b_data_driven_text_rank_score": score.clone() + text_delta,
        "stage_b_data_driven_candidate_mask": torch.ones(
            (1, query_count), dtype=torch.bool
        ),
        "pred_boxes": boxes,
    }


def _expression_record(
    *, source: str, variant: str, acc50: bool, sent_id: int = 1
) -> dict:
    identity = _identity_row(source, sent_id=sent_id)
    identity = {key: identity[key] for key in IDENTITY_KEYS}
    return {
        "schema": EXPRESSION_RECORD_SCHEMA,
        "variant": variant,
        "partition": "dev_screen",
        "dataset_source": source,
        "identity": identity,
        "image_key": {"coco_split": "train2014", "image_id": 123},
        "acc50": acc50,
    }


class RankOnlyBatchTest(unittest.TestCase):
    def test_first_argmax_tie_and_nonfirst_primary_are_respected(self):
        row = _manifest_row()
        target = _target(
            row,
            torch.tensor(
                [[0.9, 0.9, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]],
                dtype=torch.float32,
            ),
            torch.tensor([False, True]),
        )
        records = evaluate_rank_only_batch(
            _outputs(),
            [target],
            [row],
            source="refcoco",
            variant=VARIANTS[0],
            partition="dev_screen",
            temperature=1.0,
        )
        self.assertEqual(records[0]["winner_query_index"], 0)
        self.assertTrue(records[0]["acc50"])
        self.assertEqual(records[0]["positive_query_count"], 1)
        self.assertAlmostEqual(records[0]["listwise_nll"], math.log(900), places=5)

    def test_requires_exactly_900_queries(self):
        row = _manifest_row()
        target = _target(
            row,
            torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=torch.float32),
            torch.tensor([True]),
        )
        with self.assertRaisesRegex(NewHeadDevEvalError, "900"):
            evaluate_rank_only_batch(
                _outputs(query_count=899),
                [target],
                [row],
                source="refcoco",
                variant=VARIANTS[0],
                partition="dev_screen",
                temperature=0.1,
            )

    def test_rank_and_text_rank_must_be_elementwise_equal(self):
        row = _manifest_row()
        target = _target(
            row,
            torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=torch.float32),
            torch.tensor([True]),
        )
        with self.assertRaisesRegex(NewHeadDevEvalError, "elementwise equal"):
            evaluate_rank_only_batch(
                _outputs(text_delta=1e-6),
                [target],
                [row],
                source="refcoco",
                variant=VARIANTS[0],
                partition="dev_screen",
                temperature=0.1,
            )

    def test_reports_no_positive_and_no_negative_rows(self):
        row = _manifest_row()
        no_positive_target = _target(
            row,
            torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float32),
            torch.tensor([True]),
        )
        no_positive_outputs = _outputs()
        no_positive_outputs["pred_boxes"][:] = torch.tensor(
            [0.9, 0.9, 0.05, 0.05]
        )
        record = evaluate_rank_only_batch(
            no_positive_outputs,
            [no_positive_target],
            [row],
            source="refcoco",
            variant=VARIANTS[0],
            partition="dev_full",
            temperature=0.1,
        )[0]
        self.assertTrue(record["no_positive_query"])
        self.assertIsNone(record["listwise_nll"])

        all_positive_target = _target(
            row,
            torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=torch.float32),
            torch.tensor([True]),
        )
        record = evaluate_rank_only_batch(
            _outputs(all_match=True),
            [all_positive_target],
            [row],
            source="refcoco",
            variant=VARIANTS[0],
            partition="dev_full",
            temperature=0.1,
        )[0]
        self.assertTrue(record["no_negative_query"])
        self.assertIsNone(record["listwise_nll"])

    def test_runtime_requires_cfg_and_model_gate_off_and_900_queries(self):
        heads = SimpleNamespace(category_gate=False)
        model = SimpleNamespace(
            num_queries=900, stage_b_data_driven_score_heads=heads
        )
        cfg = SimpleNamespace(
            num_queries=900, stage_b_data_driven_category_gate=False
        )
        validate_rank_only_runtime(cfg, model)
        heads.category_gate = True
        with self.assertRaisesRegex(NewHeadDevEvalError, "category gate"):
            validate_rank_only_runtime(cfg, model)


class FormalContractUnitTest(unittest.TestCase):
    def test_effective_eval_contract_replays_transform_math(self):
        cfg = SimpleNamespace(
            data_aug_scales=[100, 200],
            data_aug_max_size=300,
            data_aug_scale_overlap=0.5,
            max_text_len=64,
            text_encoder_type="fixture-tokenizer",
        )
        contract = effective_eval_contract(cfg, torch.device("cpu"))
        self.assertEqual(contract["configured_resize_scales"], [100, 200])
        self.assertEqual(contract["effective_resize_scales"], [50, 100])
        self.assertEqual(contract["effective_resize_max_size"], 150)
        self.assertEqual(contract["effective_eval_short_side"], 100)
        self.assertEqual(contract["max_text_len"], 64)
        self.assertEqual(contract["text_encoder_type"], "fixture-tokenizer")
        self.assertEqual(contract["device"], "cpu")

    def test_effective_eval_contract_rejects_gflops_resize_override(self):
        with mock.patch.dict(
            os.environ, {"GFLOPS_DEBUG_SHILONG": "INFO"}, clear=False
        ):
            with self.assertRaisesRegex(NewHeadDevEvalError, "forbidden"):
                effective_eval_contract(SimpleNamespace(), torch.device("cpu"))

    def test_paired_training_contract_requires_every_key_and_json_values(self):
        saved_args = {key: None for key in PAIRED_TRAINING_EQUAL_KEYS}
        self.assertEqual(
            set(_paired_training_contract(saved_args)),
            set(PAIRED_TRAINING_EQUAL_KEYS),
        )
        missing = dict(saved_args)
        del missing["lr_drop"]
        with self.assertRaisesRegex(NewHeadDevEvalError, "missing keys"):
            _paired_training_contract(missing)
        noncanonical = dict(saved_args)
        noncanonical["lr_drop"] = object()
        with self.assertRaisesRegex(NewHeadDevEvalError, "canonical JSON"):
            _paired_training_contract(noncanonical)

    def test_evaluation_config_must_match_checkpoint_rank_training_fields(self):
        saved_args = {key: None for key in PAIRED_TRAINING_EQUAL_KEYS}
        saved_args["stage_b_data_driven_rank_lr"] = 3e-4
        cfg_values = {
            key: saved_args[key]
            for key in EVALUATION_CONFIG_CHECKPOINT_EQUAL_KEYS
        }
        cfg = SimpleNamespace(**cfg_values)
        matched = _validate_eval_config_training_contract(cfg, saved_args)
        self.assertEqual(matched["stage_b_data_driven_rank_lr"], 3e-4)
        cfg.stage_b_data_driven_rank_lr = 1e-4
        with self.assertRaisesRegex(NewHeadDevEvalError, "differs"):
            _validate_eval_config_training_contract(cfg, saved_args)

    def test_only_complete_sealed_fresh_execution_is_formal(self):
        formal_args = {
            "stage_b_data_driven_execution_scope": (
                FORMAL_NEW_HEAD_EXECUTION_SCOPE
            ),
            "stage_b_data_driven_formal_fresh_start": True,
            "stage_b_data_driven_formal_expected_optimizer_updates": (
                FORMAL_NEW_HEAD_OPTIMIZER_UPDATES
            ),
            "stage_b_data_driven_new_head_formal_contract": (
                FORMAL_NEW_HEAD_CONTRACT
            ),
        }
        complete = _new_head_execution_status(
            formal_args, optimizer_updates=FORMAL_NEW_HEAD_OPTIMIZER_UPDATES
        )
        self.assertTrue(complete["formal"])
        intermediate = _new_head_execution_status(
            formal_args, optimizer_updates=4119
        )
        self.assertFalse(intermediate["formal"])
        diagnostic_args = {
            "stage_b_data_driven_execution_scope": (
                "fresh_a0_new_head_lr_probe_u1000_v1"
            ),
            "stage_b_data_driven_formal_fresh_start": False,
            "stage_b_data_driven_formal_expected_optimizer_updates": 1000,
            "stage_b_data_driven_new_head_formal_contract": (
                "diagnostic_lr_probe_u1000_v1"
            ),
        }
        diagnostic = _new_head_execution_status(
            diagnostic_args, optimizer_updates=1000
        )
        self.assertFalse(diagnostic["formal"])
        diagnostic_args["stage_b_data_driven_new_head_formal_contract"] = (
            FORMAL_NEW_HEAD_CONTRACT
        )
        with self.assertRaisesRegex(NewHeadDevEvalError, "partially claims"):
            _new_head_execution_status(diagnostic_args, optimizer_updates=1000)

    def test_shared_training_provenance_excludes_variant_dataset_assets(self):
        saved_args = {
            "stage_b_data_driven_required_allocator_env": (
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
            "stage_b_data_driven_required_allocator_conf": (
                "expandable_segments:True"
            ),
            "stage_b_data_driven_training_provenance": {
                "schema": "fixture/v1",
                "code_files": [{"path": "/code.py", "sha256": "a" * 64}],
                "software": {"torch": "fixture"},
                "required_allocator": {
                    "environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
                    "value": "expandable_segments:True",
                },
                "allocator_environment": {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
                },
                "support_patch_pool_content": [
                    {"ordered_content_sha256": "b" * 64}
                ],
                "dataset_asset_files": [{"path": "/variant-specific.jsonl"}],
            },
        }
        shared = _shared_training_provenance(saved_args)
        self.assertNotIn("dataset_asset_files", shared)
        self.assertIn("code_files", shared)
        saved_args["stage_b_data_driven_training_provenance"][
            "required_allocator"
        ]["value"] = "drifted"
        with self.assertRaisesRegex(NewHeadDevEvalError, "required allocator"):
            _shared_training_provenance(saved_args)

    def test_formal_tensor_audit_allows_only_rank_and_patch_changes(self):
        roles = {
            "b58_base": ["backbone.weight"],
            "shared_backbone_alias": ["patch_encoder.backbone.weight"],
            "random_patch_projection": list(PATCH_PROJECTION_KEYS),
            "random_absolute_heads": [
                "stage_b_data_driven_score_heads._contract_version",
                "stage_b_data_driven_score_heads.rank_branch.weight",
                "stage_b_data_driven_score_heads.confidence_branch.weight",
            ],
        }
        state = {
            key: torch.tensor([float(index)], dtype=torch.float32)
            for index, key in enumerate(
                [key for values in roles.values() for key in values], start=1
            )
        }
        metadata = {
            "schema": "pivot.stageb.data_driven.initializer/v1",
            "role_keys": roles,
            "invariants": {
                "b58_is_only_checkpoint_source": True,
                "no_r100_p50_u0_or_stagea_tensor_source": True,
                "canonical_query_and_full_text_heads_are_separate": True,
                "rank_and_confidence_parameters_are_disjoint": True,
                "patch_backbone_aliases_b58": True,
            },
        }
        for role, keys in roles.items():
            metadata[f"{role}_tensor_sha256"] = (
                data_driven_tensor_state_sha256(state, keys)
            )
        payload = {"model": state, "data_driven_initializer": metadata}
        trained = {key: value.clone() for key, value in state.items()}
        trained[roles["random_absolute_heads"][1]] += 1
        trained[roles["random_patch_projection"][0]] += 1
        audit = _audit_frozen_initializer_tensors(payload, trained)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["frozen_tensor_sha256_equal"])
        self.assertGreater(
            audit["mutable_roles"]["absolute_rank"]["changed_tensor_count"],
            0,
        )
        trained[roles["random_absolute_heads"][2]] += 1
        with self.assertRaisesRegex(NewHeadDevEvalError, "frozen A0 tensors"):
            _audit_frozen_initializer_tensors(payload, trained)

    def test_formal_summary_flag_cannot_hide_wrong_update_count(self):
        execution = {
            "formal": True,
            "reason": "sealed_formal_fresh_run_complete",
            "execution_scope": FORMAL_NEW_HEAD_EXECUTION_SCOPE,
            "formal_fresh_start": True,
            "declared_optimizer_updates": FORMAL_NEW_HEAD_OPTIMIZER_UPDATES,
            "observed_optimizer_updates": FORMAL_NEW_HEAD_OPTIMIZER_UPDATES,
            "formal_contract": FORMAL_NEW_HEAD_CONTRACT,
        }
        paired = {
            "stage_b_data_driven_execution_scope": (
                FORMAL_NEW_HEAD_EXECUTION_SCOPE
            ),
            "stage_b_data_driven_formal_fresh_start": True,
            "stage_b_data_driven_formal_expected_optimizer_updates": (
                FORMAL_NEW_HEAD_OPTIMIZER_UPDATES
            ),
            "stage_b_data_driven_new_head_formal_contract": (
                FORMAL_NEW_HEAD_CONTRACT
            ),
        }
        contract = {
            "formal_new_head_partition_evaluation": True,
            "optimizer_updates": FORMAL_NEW_HEAD_OPTIMIZER_UPDATES,
            "formal_execution_status": execution,
            "formal_runtime_binding_status": {
                "formal": True,
                "reason": "main_validated_formal_runtime_binding",
                "binding": {"schema": FORMAL_NEW_HEAD_BINDING_SCHEMA},
            },
            "frozen_initializer_tensor_audit": {
                "schema": "pivot.stageb.data_driven.a0_frozen_tensor_audit/v1",
                "passed": True,
                "frozen_tensor_sha256_equal": True,
                "mutable_roles": {
                    role: {
                        "changed_tensor_count": 1,
                        "initializer_tensor_sha256": "a" * 64,
                        "checkpoint_tensor_sha256": "b" * 64,
                    }
                    for role in ("absolute_rank", "random_patch_projection")
                },
            },
            "training_partition_status": {"formal": True},
            "paired_training_contract": paired,
        }
        self.assertTrue(
            _is_sealed_formal_checkpoint_contract(contract, label="fixture")
        )
        contract["optimizer_updates"] = 1000
        with self.assertRaisesRegex(NewHeadDevEvalError, "falsely claims"):
            _is_sealed_formal_checkpoint_contract(contract, label="fixture")

    def test_direct_support_digest_uses_sorted_class_and_within_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            for name, payload in (("a.jpg", b"a"), ("b.jpg", b"bb"), ("c.jpg", b"ccc")):
                path = root / name
                path.write_bytes(payload)
                images.append(path.resolve())
            tsv = root / "support.tsv"
            tsv.write_text(
                "path\tclass_id\n"
                f"{images[0]}\t2\n"
                f"{images[1]}\t1\n"
                f"{images[2]}\t2\n",
                encoding="ascii",
            )
            observed = _direct_support_pool_content_record(tsv)
            digest = hashlib.sha256()
            ordered = ((1, 0, images[1]), (2, 0, images[0]), (2, 1, images[2]))
            for class_id, candidate_index, path in ordered:
                header = json.dumps(
                    [
                        class_id,
                        candidate_index,
                        str(path),
                        path.stat().st_size,
                        _sha256(path.read_bytes()),
                    ],
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("ascii")
                digest.update(len(header).to_bytes(8, "little"))
                digest.update(header)
            self.assertEqual(observed["class_count"], 2)
            self.assertEqual(observed["file_count"], 3)
            self.assertEqual(
                observed["ordered_content_sha256"], digest.hexdigest()
            )


class SummaryAndBootstrapTest(unittest.TestCase):
    def test_macro_ref3_is_equal_source_weight_not_micro(self):
        records = {
            "refcoco": [
                {"acc50": True, "listwise_nll": 1.0},
                {"acc50": False, "listwise_nll": 2.0},
                {"acc50": False, "listwise_nll": 3.0},
            ],
            "refcocoplus": [{"acc50": True, "listwise_nll": 4.0}],
            "refcocog": [{"acc50": True, "listwise_nll": 5.0}],
        }
        summary = summarize_records(records)
        self.assertAlmostEqual(summary["macro_ref3_acc50"], 7.0 / 9.0)
        self.assertAlmostEqual(summary["micro"]["acc50"], 3.0 / 5.0)
        self.assertNotEqual(
            summary["macro_ref3_acc50"], summary["micro"]["acc50"]
        )

    def test_global_image_cluster_bootstrap_is_paired_and_deterministic(self):
        d0 = {
            source: [
                _expression_record(
                    source=source, variant=VARIANTS[0], acc50=False
                )
            ]
            for source in SOURCE_NAMES
        }
        d1 = {
            source: [
                _expression_record(
                    source=source, variant=VARIANTS[1], acc50=True
                )
            ]
            for source in SOURCE_NAMES
        }
        first = paired_cluster_bootstrap(d0, d1, iterations=50, seed=17)
        second = paired_cluster_bootstrap(d0, d1, iterations=50, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["unique_image_clusters"], 1)
        self.assertEqual(first["point_estimate"], 1.0)
        self.assertEqual(first["ci95"], [1.0, 1.0])

    def test_bootstrap_rejects_identity_misalignment(self):
        d0 = {
            source: [
                _expression_record(
                    source=source, variant=VARIANTS[0], acc50=False
                )
            ]
            for source in SOURCE_NAMES
        }
        d1 = {
            source: [
                _expression_record(
                    source=source, variant=VARIANTS[1], acc50=True
                )
            ]
            for source in SOURCE_NAMES
        }
        d1 = copy.deepcopy(d1)
        d1["refcoco"][0]["identity"]["sent_id"] = 99
        with self.assertRaisesRegex(NewHeadDevEvalError, "identity/order"):
            paired_cluster_bootstrap(d0, d1, iterations=10, seed=17)


class PartitionReceiptFixtureTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        outputs = {variant: {partition: {} for partition in PARTITIONS} for variant in VARIANTS}
        for variant in VARIANTS:
            for partition in PARTITIONS:
                for manifest_name, source in SOURCE_MANIFESTS:
                    path = root / variant / partition / manifest_name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    row = _identity_row(source)
                    data = _canonical_bytes(row) + b"\n"
                    path.write_bytes(data)
                    identity = tuple(row[key] for key in IDENTITY_KEYS)
                    identity_stream = _canonical_bytes(identity) + b"\n"
                    outputs[variant][partition][manifest_name] = {
                        "path": str(path.resolve()),
                        "rows": 1,
                        "unique_identities": 1,
                        "unique_image_keys": 1,
                        "ordered_identity_stream_sha256": _sha256(identity_stream),
                        "size_bytes": len(data),
                        "sha256": _sha256(data),
                    }
        partition_summary = {
            partition: {
                "rows": 3,
                "unique_image_keys": 1,
                "rows_by_manifest": {name: 1 for name in MANIFEST_NAMES},
                "ordered_image_key_stream_sha256": "0" * 64,
            }
            for partition in PARTITIONS
        }
        receipt = {
            "schema": PARTITION_RECEIPT_SCHEMA,
            "source_manifest_order": list(MANIFEST_NAMES),
            "output_layout": "<variant>/<partition>/<source_manifest>",
            "output_stream_encoding": (
                "raw_input_record_including_original_line_ending_v1"
            ),
            "outputs": outputs,
            "partition_summary": partition_summary,
            "invariants": {
                key: True for key in PARTITION_RECEIPT_REQUIRED_INVARIANTS
            },
        }
        receipt["canonical_payload_sha256"] = _sha256(_canonical_bytes(receipt))
        path = root / "receipt.json"
        path.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return path

    @staticmethod
    def _file_record(path: Path) -> dict:
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path.read_bytes()),
        }

    def _write_support_fixture(
        self, root: Path, *, binding
    ) -> tuple[Path, Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        canonical = root / "canonical.json"
        canonical.write_text("[]\n", encoding="ascii")
        image = root / "support.jpg"
        image.write_bytes(b"support-image-v1")
        runtime = root / "filtered_support.tsv"
        runtime.write_text(
            f"path\tclass_id\n{image.resolve()}\t1\n", encoding="ascii"
        )
        support = {
            "schema": SUPPORT_RECEIPT_SCHEMA,
            "inputs": {
                "partition_receipt": binding.receipt_file,
                "canonical_classes": self._file_record(canonical),
            },
            "filter_contract": {
                "candidate_stream_policy": "sealed_cache_order_delete_only_v1",
                "exclusion_policy": (
                    "dev_full_union_official_ref8_by_numeric_coco_id_v1"
                ),
                "D0_and_D1_share_identical_runtime_bank": True,
                "bank_consumers": ["D0", "D1"],
                "required_dataset_settings": {
                    "patch_bank_cache": False,
                    "patch_bank_cache_write": False,
                    "support_patch_max_per_class": 200,
                    "support_patch_use_embedding": False,
                },
            },
            "training_class_coverage": {
                "required_class_count": 78,
                "covered_class_count": 78,
                "missing_class_ids": [],
            },
            "outputs": {
                "runtime_support_tsv": {
                    **self._file_record(runtime),
                    "rows": 1,
                }
            },
            "invariants": {
                key: True for key in SUPPORT_RECEIPT_REQUIRED_INVARIANTS
            },
        }
        support["canonical_payload_sha256"] = _sha256(
            _canonical_bytes(support)
        )
        support_path = root / "support_receipt.json"
        support_path.write_text(json.dumps(support), encoding="ascii")
        return support_path, canonical, image

    def test_strict_receipt_binds_three_selected_manifest_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = self._write_fixture(Path(directory))
            binding = load_partition_binding(
                receipt_path,
                variant="d0_ordinary_primary",
                partition="dev_screen",
            )
            self.assertEqual([item.name for item in binding.manifests], list(MANIFEST_NAMES))
            self.assertEqual([item.source for item in binding.manifests], list(SOURCE_NAMES))
            self.assertTrue(all(len(item.rows) == 1 for item in binding.manifests))

    def test_both_experiments_use_the_same_ordinary_primary_dev_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = self._write_fixture(Path(directory))
            bindings = [
                load_shared_evaluation_binding(
                    receipt_path,
                    experiment_variant=variant,
                    partition="dev_screen",
                )
                for variant in VARIANTS
            ]
            self.assertTrue(
                all(binding.variant == EVALUATION_DATA_VARIANT for binding in bindings)
            )
            self.assertEqual(
                [manifest.path for manifest in bindings[0].manifests],
                [manifest.path for manifest in bindings[1].manifests],
            )

    def test_recomputed_canonical_hash_cannot_hide_ordered_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = self._write_fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            receipt["outputs"]["d0_ordinary_primary"]["dev_screen"][
                MANIFEST_NAMES[0]
            ]["ordered_identity_stream_sha256"] = "f" * 64
            del receipt["canonical_payload_sha256"]
            receipt["canonical_payload_sha256"] = _sha256(_canonical_bytes(receipt))
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(NewHeadDevEvalError, "stream contract"):
                load_partition_binding(
                    receipt_path,
                    variant="d0_ordinary_primary",
                    partition="dev_screen",
                )

    def test_partition_receipt_requires_every_named_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = self._write_fixture(Path(directory))
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            del receipt["invariants"][
                PARTITION_RECEIPT_REQUIRED_INVARIANTS[-1]
            ]
            del receipt["canonical_payload_sha256"]
            receipt["canonical_payload_sha256"] = _sha256(
                _canonical_bytes(receipt)
            )
            receipt_path.write_text(json.dumps(receipt), encoding="ascii")
            with self.assertRaisesRegex(NewHeadDevEvalError, "invariants drifted"):
                load_partition_binding(
                    receipt_path,
                    variant=VARIANTS[0],
                    partition="dev_screen",
                )

    def test_training_rows_must_agree_on_the_exact_partition_receipt_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = self._write_fixture(root)
            receipt = json.loads(receipt_path.read_text(encoding="ascii"))
            binding = load_shared_evaluation_binding(
                receipt_path,
                experiment_variant=VARIANTS[0],
                partition="dev_screen",
            )
            rows = []
            for name in MANIFEST_NAMES:
                record = receipt["outputs"][VARIANTS[0]]["train"][name]
                rows.append(
                    {
                        "anno": record["path"],
                        "stage_b_data_driven_variant": "dd0_ordinary_primary",
                        "stage_b_data_driven_partition": "train",
                        "stage_b_data_driven_manifest_sha256": record["sha256"],
                        "stage_b_data_driven_receipt": str(receipt_path.resolve()),
                        "stage_b_data_driven_receipt_sha256": _sha256(
                            receipt_path.read_bytes()
                        ),
                    }
                )
            rows[1]["stage_b_data_driven_receipt_sha256"] = "f" * 64
            dataset_path = root / "datasets.json"
            dataset_path.write_text(json.dumps({"train": rows}), encoding="ascii")
            saved_args = {
                "datasets": str(dataset_path.resolve()),
                "stage_b_data_driven_dataset_config": {
                    "path": str(dataset_path.resolve()),
                    "sha256": _sha256(dataset_path.read_bytes()),
                },
            }
            with self.assertRaisesRegex(
                NewHeadDevEvalError, "disagree on partition receipt"
            ):
                _training_partition_status(
                    saved_args,
                    binding,
                    experiment_variant=VARIANTS[0],
                )

    def test_support_binding_is_partition_and_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition_path = self._write_fixture(root / "partition")
            binding = load_shared_evaluation_binding(
                partition_path,
                experiment_variant=VARIANTS[1],
                partition="dev_screen",
            )
            support_path, canonical, image = self._write_support_fixture(
                root / "support", binding=binding
            )
            expected_sha = _sha256(support_path.read_bytes())
            observed = load_support_binding(
                support_path,
                expected_receipt_sha256=expected_sha,
                partition_binding=binding,
                canonical_classes_path=canonical,
            )
            self.assertEqual(observed.runtime_tsv["rows"], 1)
            first_content_sha = observed.support_patch_pool_content[
                "ordered_content_sha256"
            ]
            image.write_bytes(b"support-image-v2")
            changed = load_support_binding(
                support_path,
                expected_receipt_sha256=expected_sha,
                partition_binding=binding,
                canonical_classes_path=canonical,
            )
            self.assertNotEqual(
                changed.support_patch_pool_content["ordered_content_sha256"],
                first_content_sha,
            )
            runtime = support_path.parent / "filtered_support.tsv"
            runtime.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(
                NewHeadDevEvalError, "runtime TSV file identity"
            ):
                load_support_binding(
                    support_path,
                    expected_receipt_sha256=expected_sha,
                    partition_binding=binding,
                    canonical_classes_path=canonical,
                )

    def test_support_receipt_requires_every_named_invariant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition_path = self._write_fixture(root / "partition")
            binding = load_shared_evaluation_binding(
                partition_path,
                experiment_variant=VARIANTS[0],
                partition="dev_screen",
            )
            support_path, canonical, _image = self._write_support_fixture(
                root / "support", binding=binding
            )
            support = json.loads(support_path.read_text(encoding="ascii"))
            del support["invariants"][SUPPORT_RECEIPT_REQUIRED_INVARIANTS[-1]]
            del support["canonical_payload_sha256"]
            support["canonical_payload_sha256"] = _sha256(
                _canonical_bytes(support)
            )
            support_path.write_text(json.dumps(support), encoding="ascii")
            with self.assertRaisesRegex(NewHeadDevEvalError, "invariants drifted"):
                load_support_binding(
                    support_path,
                    expected_receipt_sha256=_sha256(support_path.read_bytes()),
                    partition_binding=binding,
                    canonical_classes_path=canonical,
                )

    def test_query_content_hashes_each_unique_manifest_image_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition_path = self._write_fixture(root / "partition")
            binding = load_shared_evaluation_binding(
                partition_path,
                experiment_variant=VARIANTS[1],
                partition="dev_screen",
            )
            image_dir = root / "data/COCO/coco2014/train2014"
            image_dir.mkdir(parents=True)
            image = image_dir / "COCO_train2014_000000000123.jpg"
            image.write_bytes(b"query-image-v1")
            first = query_image_content_record(binding, data_root=root / "data")
            self.assertEqual(first["image_count"], 1)
            image.write_bytes(b"query-image-v2")
            second = query_image_content_record(binding, data_root=root / "data")
            self.assertNotEqual(
                first["ordered_content_sha256"],
                second["ordered_content_sha256"],
            )

    def test_formal_training_binds_support_image_content_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition_path = self._write_fixture(root / "partition")
            partition_receipt = json.loads(
                partition_path.read_text(encoding="ascii")
            )
            binding = load_shared_evaluation_binding(
                partition_path,
                experiment_variant=VARIANTS[0],
                partition="dev_screen",
            )
            support_path, canonical, _image = self._write_support_fixture(
                root / "support", binding=binding
            )
            support_binding = load_support_binding(
                support_path,
                expected_receipt_sha256=_sha256(support_path.read_bytes()),
                partition_binding=binding,
                canonical_classes_path=canonical,
            )
            rows = []
            for name in MANIFEST_NAMES:
                record = partition_receipt["outputs"][VARIANTS[0]]["train"][name]
                rows.append(
                    {
                        "anno": record["path"],
                        "stage_b_data_driven_variant": "dd0_ordinary_primary",
                        "stage_b_data_driven_partition": "train",
                        "stage_b_data_driven_manifest_sha256": record["sha256"],
                        "stage_b_data_driven_receipt": str(
                            partition_path.resolve()
                        ),
                        "stage_b_data_driven_receipt_sha256": _sha256(
                            partition_path.read_bytes()
                        ),
                        "stage_b_data_driven_support_receipt": str(
                            support_path.resolve()
                        ),
                        "stage_b_data_driven_support_receipt_sha256": _sha256(
                            support_path.read_bytes()
                        ),
                        "support_patch_tsv": support_binding.runtime_tsv["path"],
                        "patch_bank_cache": False,
                        "patch_bank_cache_write": False,
                        "support_patch_max_per_class": 200,
                        "support_patch_use_embedding": False,
                    }
                )
            dataset_path = root / "datasets.json"
            dataset_path.write_text(json.dumps({"train": rows}), encoding="ascii")
            saved_args = {
                "datasets": str(dataset_path.resolve()),
                "stage_b_data_driven_dataset_config": {
                    "path": str(dataset_path.resolve()),
                    "sha256": _sha256(dataset_path.read_bytes()),
                },
                "stage_b_data_driven_training_provenance": {
                    "support_patch_pool_content": [
                        support_binding.support_patch_pool_content
                    ]
                },
            }
            status = _training_partition_status(
                saved_args,
                binding,
                experiment_variant=VARIANTS[0],
                support_binding=support_binding,
            )
            self.assertTrue(status["formal"])
            drifted = copy.deepcopy(saved_args)
            drifted["stage_b_data_driven_training_provenance"][
                "support_patch_pool_content"
            ][0]["ordered_content_sha256"] = "f" * 64
            with self.assertRaisesRegex(
                NewHeadDevEvalError, "support image content provenance"
            ):
                _training_partition_status(
                    drifted,
                    binding,
                    experiment_variant=VARIANTS[0],
                    support_binding=support_binding,
                )


if __name__ == "__main__":
    unittest.main()
