import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.eval_stageb_native_patch_category_dev import (
    NativePatchCategoryDevEvalError,
    RECORD_SCHEMA,
    RESULT_SCHEMA,
    _canonical_bytes,
    _category_state,
    audit_checkpoint,
    bootstrap_records,
    deterministic_bootstrap,
    evaluate_batch,
    load_receipt_binding,
    summarize_records,
    validate_metadata,
    validate_records,
    verify_result,
)
from tools.build_stageb_native_patch_category_initializer import stable_file_record


def _record(
    index: int,
    *,
    group: int,
    image: int,
    variant: int,
    source: str,
    base: bool,
    adapted: bool,
):
    native_state = "positive" if base else "negative"
    adapted_state = "positive" if adapted else "negative"
    query_states = {0: native_state, 1: adapted_state}
    if "positive" not in query_states.values():
        query_states[2] = "positive"
    state_counts = {
        state: sum(value == state for value in query_states.values())
        for state in ("positive", "negative", "neutral")
    }
    state_counts["negative"] += 900 - len(query_states)
    if native_state == "positive":
        native_best_positive_query = 0
        native_best_positive_eligible = False
        native_best_positive_patch_gap = 4.0
    elif adapted_state == "positive":
        native_best_positive_query = 1
        native_best_positive_eligible = True
        native_best_positive_patch_gap = 0.0
    else:
        native_best_positive_query = 2
        native_best_positive_eligible = True
        native_best_positive_patch_gap = 1.0
    return {
        "schema": RECORD_SCHEMA,
        "row_index": index,
        "group_id": f"group-{group}",
        "image_cluster": f"train2014:{image}",
        "image_id": image,
        "class_id": group + 10,
        "support_class_id": group + 100,
        "source_dataset": source,
        "variant_index": variant,
        "ann_id": 100 + index,
        "ref_id": 200 + index,
        "sent_id": 300 + index,
        "base_query": 0,
        "adapted_query": 1,
        "base_iou": 0.75 if base else 0.1,
        "adapted_iou": 0.75 if adapted else 0.1,
        "base_correct": base,
        "adapted_correct": adapted,
        "fixed": not base and adapted,
        "regressed": base and not adapted,
        "winner_changed": True,
        "eligible_queries": 2,
        "base_query_eligible": False,
        "adapted_native_score": 0.8,
        "base_native_score": 0.9,
        "adapted_standardized_patch_score": 1.0,
        "category_gt_count": 1,
        "category_positive_query_count": state_counts["positive"],
        "category_negative_query_count": state_counts["negative"],
        "category_neutral_query_count": state_counts["neutral"],
        "exists_category_positive": True,
        "native_winner_category_state": native_state,
        "native_winner_category_max_iou": 0.75 if base else 0.1,
        "native_winner_category_eligible": False,
        "native_winner_patch_gap": 4.0,
        "adapted_winner_category_state": adapted_state,
        "adapted_winner_category_max_iou": 0.75 if adapted else 0.1,
        "adapted_winner_category_eligible": True,
        "adapted_winner_patch_gap": 0.0,
        "native_best_positive_query": native_best_positive_query,
        "native_best_positive_query_max_iou": 0.75,
        "native_best_positive_query_eligible": native_best_positive_eligible,
        "native_best_positive_query_patch_gap": native_best_positive_patch_gap,
    }


class NativePatchCategoryDevEvalTest(unittest.TestCase):
    def test_batch_uses_full_expression_native_score_and_patch_gate(self) -> None:
        batch = 1
        queries = 900
        tokens = 4
        text = torch.full((batch, queries, tokens), -6.0)
        # Query 0 wins native only because both expression tokens are high.
        text[0, 0, 0] = 4.0
        text[0, 0, 1] = 4.0
        text[0, 1, 0] = 3.0
        text[0, 1, 1] = 3.0
        patch = torch.zeros((batch, queries, 1))
        patch[0, 1, 0] = 10.0
        boxes = torch.zeros((batch, queries, 4))
        boxes[0, 0] = torch.tensor([0.2, 0.2, 0.1, 0.1])
        boxes[0, 1] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        phrase_mask = torch.tensor([[[True, True, False, False]]])
        outputs = {
            "pred_logits_text": text,
            "pred_logits_patch": patch,
            "pred_boxes": boxes,
            "phrase_to_token_mask": phrase_mask,
        }
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "labels": torch.tensor([3], dtype=torch.int64),
            "support_class": torch.tensor([3], dtype=torch.int64),
            "primary_instance_mask": torch.tensor([True]),
            "image_id": torch.tensor([7]),
            "ann_id": torch.tensor([8]),
            "ref_id": torch.tensor([9]),
            "sent_id": torch.tensor([10]),
        }
        metadata = {
            "row_index": 0,
            "group_id": "g",
            "image_cluster": "train2014:7",
            "image_id": 7,
            "class_id": 3,
            "support_class_id": 3,
            "source_dataset": "refcoco",
            "variant_index": 0,
            "ann_id": 8,
            "ref_id": 9,
            "sent_id": 10,
        }

        rows = evaluate_batch(
            outputs,
            [target],
            [metadata],
            expected_phrase_mask=phrase_mask,
            gate_max_gap=0.0,
            gate_clip=5.0,
        )

        self.assertEqual(rows[0]["base_query"], 0)
        self.assertEqual(rows[0]["adapted_query"], 1)
        self.assertFalse(rows[0]["base_correct"])
        self.assertTrue(rows[0]["adapted_correct"])
        self.assertTrue(rows[0]["fixed"])
        self.assertEqual(rows[0]["eligible_queries"], 1)
        self.assertEqual(rows[0]["native_winner_category_state"], "negative")
        self.assertEqual(rows[0]["adapted_winner_category_state"], "positive")
        self.assertTrue(rows[0]["exists_category_positive"])
        self.assertEqual(rows[0]["native_best_positive_query"], 1)
        self.assertTrue(rows[0]["native_best_positive_query_eligible"])

    def test_batch_category_state_uses_max_iou_across_all_same_class_gt(self) -> None:
        queries = 900
        text = torch.full((1, queries, 2), -6.0)
        text[0, 0, 0] = 4.0
        text[0, 1, 0] = 3.0
        patch = torch.zeros((1, queries, 1))
        patch[0, 1, 0] = 10.0
        boxes = torch.zeros((1, queries, 4))
        auxiliary_box = torch.tensor([0.2, 0.2, 0.1, 0.1])
        primary_box = torch.tensor([0.5, 0.5, 0.2, 0.2])
        boxes[0, 0] = auxiliary_box
        boxes[0, 1] = primary_box
        phrase_mask = torch.tensor([[[True, False]]])
        outputs = {
            "pred_logits_text": text,
            "pred_logits_patch": patch,
            "pred_boxes": boxes,
            "phrase_to_token_mask": phrase_mask,
        }
        target = {
            "boxes": torch.stack((primary_box, auxiliary_box)),
            "labels": torch.tensor([3, 3], dtype=torch.int64),
            "support_class": torch.tensor([3], dtype=torch.int64),
            "primary_instance_mask": torch.tensor([True, False]),
            "image_id": torch.tensor([7]),
            "ann_id": torch.tensor([8]),
            "ref_id": torch.tensor([9]),
            "sent_id": torch.tensor([10]),
        }
        metadata = {
            "row_index": 0,
            "group_id": "g",
            "image_cluster": "train2014:7",
            "image_id": 7,
            "class_id": 3,
            "support_class_id": 3,
            "source_dataset": "refcoco",
            "variant_index": 0,
            "ann_id": 8,
            "ref_id": 9,
            "sent_id": 10,
        }

        row = evaluate_batch(
            outputs,
            [target],
            [metadata],
            expected_phrase_mask=phrase_mask,
            gate_max_gap=0.0,
            gate_clip=5.0,
        )[0]

        self.assertFalse(row["base_correct"])
        self.assertEqual(row["native_winner_category_state"], "positive")
        self.assertAlmostEqual(row["native_winner_category_max_iou"], 1.0, places=3)
        self.assertEqual(row["native_best_positive_query"], 0)
        self.assertFalse(row["native_best_positive_query_eligible"])
        self.assertEqual(row["adapted_winner_category_state"], "positive")

    def test_category_state_thresholds_are_half_open(self) -> None:
        self.assertEqual(_category_state(0.299999), "negative")
        self.assertEqual(_category_state(0.3), "neutral")
        self.assertEqual(_category_state(0.499999), "neutral")
        self.assertEqual(_category_state(0.5), "positive")

    def test_batch_rejects_mask_drift_and_non_900_geometry(self) -> None:
        outputs = {
            "pred_logits_text": torch.zeros((1, 2, 2)),
            "pred_logits_patch": torch.zeros((1, 2, 1)),
            "pred_boxes": torch.zeros((1, 2, 4)),
            "phrase_to_token_mask": torch.tensor([[[True, False]]]),
        }
        with self.assertRaisesRegex(
            NativePatchCategoryDevEvalError, "query/batch geometry"
        ):
            evaluate_batch(
                outputs,
                [],
                [],
                expected_phrase_mask=outputs["phrase_to_token_mask"],
                gate_max_gap=3.0,
                gate_clip=5.0,
            )

    def test_summary_reports_rows_groups_images_and_sources(self) -> None:
        records = [
            _record(0, group=0, image=1, variant=0, source="refcoco", base=False, adapted=True),
            _record(1, group=0, image=1, variant=1, source="refcocoplus", base=True, adapted=True),
            _record(2, group=0, image=1, variant=2, source="refcocog", base=False, adapted=False),
            _record(3, group=1, image=1, variant=0, source="refcoco", base=True, adapted=False),
            _record(4, group=1, image=1, variant=1, source="refcocoplus", base=False, adapted=False),
            _record(5, group=1, image=1, variant=2, source="refcocog", base=True, adapted=True),
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["rows"]["base_correct"], 3)
        self.assertEqual(summary["rows"]["adapted_correct"], 3)
        self.assertEqual(summary["rows"]["fixed"], 1)
        self.assertEqual(summary["rows"]["regressed"], 1)
        self.assertEqual(summary["groups"]["groups"], 2)
        self.assertEqual(summary["groups"]["improved"], 1)
        self.assertEqual(summary["groups"]["regressed"], 1)
        self.assertEqual(summary["images"]["images"], 1)
        self.assertEqual(set(summary["sources"]), {"refcoco", "refcocoplus", "refcocog"})
        mechanism = summary["category_mechanism"]
        self.assertEqual(
            mechanism["positive_winner_eligibility"],
            {"numerator": 0, "denominator": 3, "rate": 0.0},
        )
        self.assertEqual(
            mechanism["negative_winner_rejection"],
            {"numerator": 3, "denominator": 3, "rate": 1.0},
        )
        self.assertEqual(
            mechanism["category_fixable_adapted_positive_recall"],
            {"numerator": 3, "denominator": 6, "rate": 0.5},
        )
        self.assertEqual(mechanism["neutral_winner_counts"], {"native": 0, "adapted": 0})
        self.assertEqual(mechanism["eligible_queries"]["q01"], 2.0)
        self.assertEqual(mechanism["eligible_queries"]["q50"], 2.0)
        self.assertEqual(mechanism["eligible_queries"]["q99"], 2.0)

    def test_bootstrap_is_deterministic_for_group_and_image_units(self) -> None:
        records = []
        for group in range(4):
            for variant in range(3):
                records.append(
                    _record(
                        len(records),
                        group=group,
                        image=group // 2,
                        variant=variant,
                        source=("refcoco", "refcocoplus", "refcocog")[variant],
                        base=group in {1, 3},
                        adapted=group in {0, 3},
                    )
                )

        first = bootstrap_records(records, iterations=100, seed=42)
        second = bootstrap_records(records, iterations=100, seed=42)
        other = bootstrap_records(records, iterations=100, seed=43)

        self.assertEqual(first, second)
        self.assertNotEqual(
            first["group"]["derived_seed"], other["group"]["derived_seed"]
        )
        self.assertEqual(first["group"]["unit_count"], 4)
        self.assertEqual(first["image"]["unit_count"], 2)
        self.assertLessEqual(first["group"]["ci_lower"], first["group"]["point_estimate"])
        self.assertGreaterEqual(first["group"]["ci_upper"], first["group"]["point_estimate"])

    def test_records_fail_closed_on_transition_or_group_drift(self) -> None:
        records = [
            _record(i, group=0, image=1, variant=i, source="refcoco", base=False, adapted=True)
            for i in range(3)
        ]
        validate_records(records)
        records[0]["fixed"] = False
        with self.assertRaisesRegex(
            NativePatchCategoryDevEvalError, "transition flags"
        ):
            validate_records(records)

    def test_metadata_requires_ordered_three_row_groups(self) -> None:
        rows = []
        for variant in range(3):
            rows.append(
                {
                    "stage_b_native_patch_category_d1": True,
                    "stage_b_native_patch_category_d1_schema": (
                        "pivot.stageb.native_patch_category_d1_row/v1"
                    ),
                    "native_patch_category_group_id": "g",
                    "native_patch_category_source_dataset": "refcoco",
                    "native_patch_category_variant_index": variant,
                    "image_id": 1,
                    "category_complete_coco_split": "train2014",
                    "category_complete_coco_category_id": 4,
                    "category_complete_instance_count": 1,
                    "instances": [
                        {
                            "class_id": 104,
                            "category_complete_primary": True,
                        }
                    ],
                    "ann_id": variant + 10,
                    "ref_id": variant + 20,
                    "sent_id": variant + 30,
                }
            )
        normalized = validate_metadata(
            rows, {"rows": 3, "groups": 1, "unique_images": 1}
        )
        self.assertEqual([row["variant_index"] for row in normalized], [0, 1, 2])
        self.assertEqual([row["class_id"] for row in normalized], [4, 4, 4])
        self.assertEqual(
            [row["support_class_id"] for row in normalized], [104, 104, 104]
        )
        rows[1]["native_patch_category_variant_index"] = 2
        with self.assertRaisesRegex(
            NativePatchCategoryDevEvalError, "ordered three-row"
        ):
            validate_metadata(rows, {"rows": 3, "groups": 1, "unique_images": 1})

    def test_receipt_binds_canonical_payload_and_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "dev_screen.jsonl"
            manifest.write_text("{}\n", encoding="ascii")
            manifest_record = {
                "path": str(manifest.resolve()),
                "size_bytes": manifest.stat().st_size,
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "rows": 3,
            }
            receipt = {
                "schema": "pivot.stageb.native_patch_category_d1_receipt/v1",
                "splits": {
                    "dev_screen": {
                        "rows": 3,
                        "groups": 1,
                        "unique_images": 1,
                        "output": manifest_record,
                    }
                },
                "invariants": {"fixture_is_sealed": True},
            }
            receipt["canonical_payload_sha256"] = hashlib.sha256(
                _canonical_bytes(receipt)
            ).hexdigest()
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="ascii")

            binding, _receipt_record, observed_manifest = load_receipt_binding(
                receipt_path, split="dev_screen"
            )

            self.assertEqual(binding["rows"], 3)
            self.assertEqual(observed_manifest["sha256"], manifest_record["sha256"])
            manifest.write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(
                NativePatchCategoryDevEvalError, "manifest file binding"
            ):
                load_receipt_binding(receipt_path, split="dev_screen")

    def test_checkpoint_audit_allows_only_eight_patch_projection_keys(self) -> None:
        model = torch.nn.Linear(1, 1)
        initializer_state = {
            "weight": torch.tensor([[1.0]]),
            "bias": torch.tensor([0.0]),
        }
        checkpoint_state = {
            "weight": torch.tensor([[1.0]]),
            "bias": torch.tensor([0.0]),
        }
        initializer = {"model": initializer_state}
        checkpoint = {"model": checkpoint_state}
        with mock.patch(
            "tools.eval_stageb_native_patch_category_dev.validate_native_patch_category_initializer_payload"
        ):
            state, audit = audit_checkpoint(
                model,
                initializer,
                checkpoint,
                initializer_label="init",
                checkpoint_label="ckpt",
            )
        self.assertEqual(set(state), {"weight", "bias"})
        self.assertEqual(audit["changed_tensor_count"], 0)
        checkpoint["model"]["weight"] = torch.tensor([[2.0]])
        with mock.patch(
            "tools.eval_stageb_native_patch_category_dev.validate_native_patch_category_initializer_payload"
        ):
            with self.assertRaisesRegex(
                NativePatchCategoryDevEvalError, "changed frozen tensors"
            ):
                audit_checkpoint(
                    model,
                    initializer,
                    checkpoint,
                    initializer_label="init",
                    checkpoint_label="ckpt",
                )

    def test_verify_replays_records_and_rejects_checkpoint_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            datasets = root / "datasets.json"
            initializer = root / "initializer.pth"
            checkpoint = root / "checkpoint.pth"
            for path, value in (
                (config, "config"),
                (datasets, "datasets"),
                (initializer, "initializer"),
                (checkpoint, "checkpoint"),
            ):
                path.write_text(value, encoding="ascii")
            manifest = root / "dev_screen.jsonl"
            manifest.write_text("{}\n", encoding="ascii")
            manifest_output = {
                **stable_file_record(manifest, label="fixture manifest"),
                "rows": 3,
            }
            receipt = {
                "schema": "pivot.stageb.native_patch_category_d1_receipt/v1",
                "splits": {
                    "dev_screen": {
                        "rows": 3,
                        "groups": 1,
                        "unique_images": 1,
                        "output": manifest_output,
                    }
                },
                "invariants": {"fixture_is_sealed": True},
            }
            receipt["canonical_payload_sha256"] = hashlib.sha256(
                _canonical_bytes(receipt)
            ).hexdigest()
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="ascii")
            records = [
                _record(
                    i,
                    group=0,
                    image=1,
                    variant=i,
                    source=("refcoco", "refcocoplus", "refcocog")[i],
                    base=False,
                    adapted=True,
                )
                for i in range(3)
            ]
            records_path = root / "records.jsonl"
            records_path.write_bytes(
                b"".join(_canonical_bytes(record) + b"\n" for record in records)
            )
            binding, receipt_record, manifest_record = load_receipt_binding(
                receipt_path, split="dev_screen"
            )
            evaluator = Path(
                "tools/eval_stageb_native_patch_category_dev.py"
            ).resolve()
            aggregator = Path(
                "models/GroundingDINO/stage_b_gdino_score_adapter.py"
            ).resolve()
            gate = Path(
                "models/GroundingDINO/stage_b_native_patch_category.py"
            ).resolve()
            result = {
                "schema": RESULT_SCHEMA,
                "status": "complete",
                "evaluation": {
                    "split": "dev_screen",
                    "gate_max_gap": 3.0,
                    "gate_clip": 5.0,
                },
                "inputs": {
                    "config": stable_file_record(config, label="config"),
                    "config_import_chain": [
                        stable_file_record(config, label="config dependency")
                    ],
                    "datasets": stable_file_record(datasets, label="datasets"),
                    "data_receipt": receipt_record,
                    "manifest": manifest_record,
                    "initializer": stable_file_record(initializer, label="initializer"),
                    "checkpoint": stable_file_record(checkpoint, label="checkpoint"),
                },
                "code": {
                    "evaluator": stable_file_record(evaluator, label="evaluator"),
                    "native_aggregator": stable_file_record(
                        aggregator, label="aggregator"
                    ),
                    "patch_gate": stable_file_record(gate, label="gate"),
                },
                "data_binding": binding,
                "checkpoint_audit": {"fixture": True},
                "runtime": {},
                "metrics": summarize_records(records),
                "bootstrap": bootstrap_records(records, iterations=10, seed=42),
                "outputs": {
                    "records": stable_file_record(records_path, label="records")
                },
            }
            result["canonical_payload_sha256"] = hashlib.sha256(
                _canonical_bytes(result)
            ).hexdigest()
            output = root / "result.json"
            output.write_text(json.dumps(result), encoding="ascii")
            args = argparse.Namespace(
                config=config,
                datasets=datasets,
                receipt=receipt_path,
                initializer=initializer,
                checkpoint=checkpoint,
                split="dev_screen",
                output_json=output,
                records_jsonl=records_path,
                gate_max_gap=3.0,
                gate_clip=5.0,
                bootstrap_iterations=10,
                bootstrap_seed=42,
            )

            with mock.patch(
                "tools.eval_stageb_native_patch_category_dev.replay_checkpoint_audit",
                return_value={"fixture": True},
            ), mock.patch(
                "tools.eval_stageb_native_patch_category_dev.config_import_chain",
                return_value=[config],
            ):
                verified = verify_result(args)

            self.assertTrue(verified["verified"])
            self.assertEqual(verified["records_file"]["path"], str(records_path))
            checkpoint.write_text("tampered", encoding="ascii")
            with mock.patch(
                "tools.eval_stageb_native_patch_category_dev.replay_checkpoint_audit",
                return_value={"fixture": True},
            ), mock.patch(
                "tools.eval_stageb_native_patch_category_dev.config_import_chain",
                return_value=[config],
            ):
                with self.assertRaisesRegex(
                    NativePatchCategoryDevEvalError, "hash binding drifted"
                ):
                    verify_result(args)

    def test_bootstrap_rejects_invalid_seed_and_iteration(self) -> None:
        records = [
            _record(i, group=0, image=1, variant=i, source="refcoco", base=False, adapted=True)
            for i in range(3)
        ]
        with self.assertRaisesRegex(NativePatchCategoryDevEvalError, "iterations"):
            deterministic_bootstrap(
                records,
                unit_key="group_id",
                unit_name="group",
                iterations=0,
                seed=42,
            )
        with self.assertRaisesRegex(NativePatchCategoryDevEvalError, "seed"):
            deterministic_bootstrap(
                records,
                unit_key="group_id",
                unit_name="group",
                iterations=2,
                seed=True,
            )


if __name__ == "__main__":
    unittest.main()
