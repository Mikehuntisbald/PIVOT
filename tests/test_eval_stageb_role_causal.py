import unittest
import copy
import csv
import hashlib
import tempfile
from pathlib import Path

import torch

from tools import eval_stageb_role_causal as role_eval


def _outputs(*, text_logits=None):
    pred_boxes = torch.tensor(
        [
            [
                [0.15, 0.15, 0.10, 0.10],
                [0.50, 0.50, 0.20, 0.20],
                [0.85, 0.85, 0.10, 0.10],
            ]
        ],
        dtype=torch.float32,
    )
    candidate_idx = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    patch_logits = torch.tensor([[4.0, 1.0, 0.0]], dtype=torch.float32)
    if text_logits is None:
        text_logits = torch.tensor([[[0.0], [3.0], [1.0]]], dtype=torch.float32)
    fused_logits = text_logits + patch_logits.unsqueeze(-1)
    expression_valid = torch.tensor([[True]], dtype=torch.bool)
    dense_logits = fused_logits.clone()
    dense_score = fused_logits.sigmoid()
    candidate_mask = torch.ones_like(fused_logits, dtype=torch.bool)
    return {
        "pred_boxes": pred_boxes,
        "pred_logits_patch": patch_logits.clone(),
        "stage_b_v11_candidate_boxes": pred_boxes.clone(),
        "stage_b_v11_final_phrase_logits": fused_logits,
        "stage_b_v15_candidate_patch_logits": patch_logits,
        "stage_b_v11_candidate_idx": candidate_idx,
        "stage_b_v11_expression_valid_mask": expression_valid,
        "stage_b_v11_candidate_mask": candidate_mask,
        "stage_b_v15_dense_rank_logits": dense_logits,
        "stage_b_v15_dense_rank_score": dense_score,
    }


def _target():
    return {"boxes": torch.tensor([[0.50, 0.50, 0.20, 0.20]])}


class RoleRoutingTest(unittest.TestCase):
    def test_canonical_text_admission_uses_max_phrase_token_logit(self):
        logits = torch.tensor(
            [
                [
                    [100.0, 1.0, 2.0, -100.0],
                    [100.0, 3.0, 0.0, -100.0],
                ]
            ]
        )
        mask = torch.tensor([[False, True, True, False]])
        scores = role_eval.canonical_text_admission_scores(logits, mask)
        self.assertTrue(torch.equal(scores, torch.tensor([[2.0, 3.0]])))
        with self.assertRaisesRegex(ValueError, "at least one phrase token"):
            role_eval.canonical_text_admission_scores(
                logits, torch.zeros_like(mask)
            )

    def test_extracts_exact_text_surface_and_routes_fixed_candidates(self):
        outputs = _outputs()
        components = role_eval.extract_role_components(
            outputs, [_target()], patch_rank_weight=1.0
        )
        expected_text = torch.tensor([[[0.0], [3.0], [1.0]]])
        self.assertTrue(torch.equal(components["text_logits"], expected_text))

        records = role_eval.route_role_records(
            components,
            metadata=[{"sample_id": "sample-1"}],
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["routes"]["patch_only"]["query_index"], 0)
        self.assertEqual(record["routes"]["text_only"]["query_index"], 1)
        # Patch+text ties q0 and q1 at 4.0; first-argmax is deterministic.
        self.assertEqual(
            record["routes"]["patch_admission_text_rank"]["query_index"], 0
        )
        self.assertFalse(record["candidate_oracle"]["1"]["recall_iou50"])
        self.assertTrue(record["candidate_oracle"]["5"]["recall_iou50"])
        self.assertEqual(record["candidate_oracle"]["5"]["effective_k"], 3)
        self.assertFalse(record["true_role_swap"]["supported"])

        summary = role_eval.aggregate_role_records(records)
        self.assertEqual(summary["routes"]["patch_only"]["acc50"], 0.0)
        self.assertEqual(summary["routes"]["text_only"]["acc50"], 1.0)
        self.assertEqual(summary["candidate_oracle"]["1"]["recall_iou50"], 0.0)
        self.assertEqual(summary["candidate_oracle"]["5"]["recall_iou50"], 1.0)
        self.assertEqual(summary["candidate_oracle"]["all"]["recall_iou50"], 1.0)
        self.assertEqual(set(summary["table_a_rows"]), {"G1", "G2", "G3", "G4"})
        for row in summary["table_a_rows"].values():
            self.assertEqual(
                set(row["ranked_oracle"]), {"1", "5", "10", "50", "all"}
            )

    def test_candidate_boxes_must_be_exact_gather(self):
        outputs = _outputs()
        outputs["stage_b_v11_candidate_boxes"] = outputs[
            "stage_b_v11_candidate_boxes"
        ].clone()
        outputs["stage_b_v11_candidate_boxes"][0, 0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "exact gather"):
            role_eval.extract_role_components(
                outputs, [_target()], patch_rank_weight=1.0
            )

    def test_true_role_swap_is_merged_as_g5_with_matched_candidate_count(self):
        patch_components = role_eval.extract_role_components(
            _outputs(), [_target()], patch_rank_weight=1.0
        )
        swap_components = role_eval.extract_role_components(
            _outputs(
                text_logits=torch.tensor(
                    [[[0.0], [5.0], [1.0]]], dtype=torch.float32
                )
            ),
            [_target()],
            patch_rank_weight=1.0,
        )
        metadata = [
            {
                "image_id": 1,
                "ann_id": 2,
                "ref_id": 3,
                "sent_id": 4,
                "sample_id": "g5-1",
            }
        ]
        merged = role_eval.merge_true_role_swap_records(
            role_eval.route_role_records(patch_components, metadata=metadata),
            role_eval.route_role_records(swap_components, metadata=metadata),
        )
        record = merged[0]
        self.assertTrue(record["true_role_swap"]["supported"])
        self.assertIn(role_eval.TRUE_ROLE_SWAP_ROUTE, record["routes"])
        self.assertTrue(
            record["routes"][role_eval.TRUE_ROLE_SWAP_ROUTE][
                "candidate_count_matched_to_patch_route"
            ]
        )
        summary = role_eval.aggregate_role_records(merged)
        self.assertTrue(summary["true_role_swap"]["supported"])
        self.assertIn(role_eval.TRUE_ROLE_SWAP_ROUTE, summary["routes"])
        self.assertIn("true_role_swap_candidate_oracle", summary)
        self.assertIn("G5", summary["table_a_rows"])
        self.assertIn(
            "top1_query_churn_vs_patch_admission_text_rank",
            summary["table_a_rows"]["G5"],
        )

    def test_g5_equality_receipt_requires_every_bitwise_check(self):
        receipt = role_eval._new_g5_equality_receipt()
        for key in receipt:
            receipt[key] = 3
        finalized = role_eval._finalize_g5_equality_receipt(receipt)
        self.assertEqual(finalized["status"], "passed")
        self.assertTrue(finalized["no_grad_required"])
        receipt["boxes_bitwise_equal_count"] = 2
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            role_eval._finalize_g5_equality_receipt(receipt)

    def test_true_role_swap_accepts_text_order_but_patch_route_rejects_it(self):
        outputs = _outputs()
        order = torch.tensor([[1, 0, 2]], dtype=torch.int64)
        outputs["stage_b_v11_candidate_idx"] = order
        outputs["stage_b_v11_candidate_boxes"] = torch.gather(
            outputs["pred_boxes"], 1, order.unsqueeze(-1).expand(-1, -1, 4)
        )
        outputs["stage_b_v15_candidate_patch_logits"] = torch.gather(
            outputs["pred_logits_patch"], 1, order
        )
        outputs["stage_b_v11_final_phrase_logits"] = torch.gather(
            outputs["stage_b_v15_dense_rank_logits"],
            1,
            order.unsqueeze(-1),
        )
        with self.assertRaisesRegex(ValueError, "descending Stage-A Top-K"):
            role_eval.extract_role_components(
                outputs,
                [_target()],
                patch_rank_weight=1.0,
                candidate_source="patch_topk",
            )
        components = role_eval.extract_role_components(
            outputs,
            [_target()],
            patch_rank_weight=1.0,
            candidate_source="canonical_text_topk",
        )
        self.assertEqual(components["candidate_source"], "canonical_text_topk")


class CounterfactualRoleTest(unittest.TestCase):
    def test_reports_patch_invariance_and_fulltext_rank_churn(self):
        positive = role_eval.extract_role_components(
            _outputs(), [_target()], patch_rank_weight=1.0
        )
        negative_text = torch.tensor([[[3.0], [0.0], [1.0]]], dtype=torch.float32)
        negative = role_eval.extract_role_components(
            _outputs(text_logits=negative_text), [_target()], patch_rank_weight=1.0
        )
        records = role_eval.counterfactual_role_records(
            positive,
            negative,
            [
                {
                    "sample_id": "pair-1",
                    "tn_edits": [
                        {
                            "category": "color",
                            "replace_from": "red",
                            "replace_to": "blue",
                        }
                    ],
                }
            ],
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["edit_taxonomy"], "color")
        self.assertEqual(record["candidate_alignment"], "exact")
        self.assertEqual(
            record["surfaces"]["patch"][
                "delta_max_logit_negative_minus_positive"
            ],
            0.0,
        )
        self.assertFalse(record["surfaces"]["patch"]["top1_changed"])
        self.assertTrue(record["surfaces"]["fulltext"]["top1_changed"])
        self.assertGreater(
            record["surfaces"]["fulltext"]["mean_absolute_rank_change"], 0.0
        )

        summary = role_eval.aggregate_counterfactual_records(records)
        self.assertEqual(
            summary["overall"]["surfaces"]["patch"]["invariant_rate_at_1e-8"],
            1.0,
        )
        self.assertEqual(
            summary["by_edit_taxonomy"]["color"]["surfaces"]["fulltext"][
                "top1_change_rate"
            ],
            1.0,
        )

    def test_taxonomy_mapping_is_explicit_about_mixed_edits(self):
        self.assertEqual(
            role_eval.normalize_edit_taxonomy(["object_type"]), "noun_category"
        )
        self.assertEqual(role_eval.normalize_edit_taxonomy(["activity"]), "action")
        self.assertEqual(role_eval.normalize_edit_taxonomy(["spatial relation"]), "relation")
        self.assertEqual(role_eval.normalize_edit_taxonomy(["position"]), "spatial")
        self.assertEqual(role_eval.normalize_edit_taxonomy(["height"]), "size")
        self.assertEqual(
            role_eval.normalize_edit_taxonomy(["color", "position"]), "mixed"
        )
        self.assertEqual(role_eval.normalize_edit_taxonomy(["material"]), "other")

    def test_noun_edit_does_not_claim_patch_category_causality(self):
        positive = role_eval.extract_role_components(
            _outputs(), [_target()], patch_rank_weight=1.0
        )
        negative = role_eval.extract_role_components(
            _outputs(), [_target()], patch_rank_weight=1.0
        )
        record = role_eval.counterfactual_role_records(
            positive,
            negative,
            [{"replace_category": ["object_type"]}],
        )[0]
        self.assertTrue(record["canonical_prompt_held_fixed"])
        self.assertTrue(record["contains_noun_category_edit"])
        self.assertIs(record["patch_category_causal_supported"], False)
        self.assertIn("held fixed", record["patch_category_causal_reason"])
        summary = role_eval.aggregate_counterfactual_records([record])
        self.assertFalse(summary["overall"]["patch_category_evidence_eligible"])
        self.assertFalse(
            summary["overall"]["surfaces"]["patch"][
                "category_causal_evidence_eligible"
            ]
        )
        self.assertFalse(
            summary["causal_contract"][
                "noun_category_fulltext_edits_test_patch_category_role"
            ]
        )

    def test_training_pair_caption_is_rewritten_to_negative_only(self):
        target = {
            "caption": "a red bowl . a blue bowl .",
            "verifier_caption": "a red bowl . a blue bowl .",
            "rank_positive_captions": ["a red bowl ."],
            "phrase_to_token_mask": torch.ones((2, 4), dtype=torch.bool),
            "canonical_to_token_mask": torch.ones((2, 4), dtype=torch.bool),
        }
        samples = object()
        rewritten = role_eval._negative_only_pair_batch(
            (samples, [target]), [{"negative_phrase": "a blue bowl"}]
        )
        self.assertIs(rewritten[0], samples)
        rewritten_target = rewritten[1][0]
        self.assertEqual(rewritten_target["caption"], "a blue bowl .")
        self.assertEqual(rewritten_target["rank_positive_captions"], ["a red bowl ."])
        self.assertNotIn("phrase_to_token_mask", rewritten_target)
        self.assertNotIn("canonical_to_token_mask", rewritten_target)
        self.assertEqual(target["caption"], "a red bowl . a blue bowl .")

    def test_true_role_swap_counterfactual_surface_is_merged(self):
        positive = role_eval.extract_role_components(
            _outputs(), [_target()], patch_rank_weight=1.0
        )
        negative = role_eval.extract_role_components(
            _outputs(
                text_logits=torch.tensor(
                    [[[3.0], [0.0], [1.0]]], dtype=torch.float32
                )
            ),
            [_target()],
            patch_rank_weight=1.0,
        )
        metadata = [
            {
                "image_id": 1,
                "ann_id": 2,
                "ref_id": 3,
                "sent_id": 4,
                "sample_id": "g5-pair",
                "replace_category": ["color"],
            }
        ]
        patch_records = role_eval.counterfactual_role_records(
            positive, negative, metadata
        )
        swap_records = role_eval.counterfactual_role_records(
            positive, negative, metadata
        )
        merged = role_eval.merge_true_role_swap_counterfactual_records(
            patch_records, swap_records
        )
        self.assertIn(role_eval.TRUE_ROLE_SWAP_ROUTE, merged[0]["surfaces"])
        summary = role_eval.aggregate_counterfactual_records(merged)
        self.assertTrue(summary["true_role_swap"]["supported"])
        self.assertIn(
            role_eval.TRUE_ROLE_SWAP_ROUTE,
            summary["overall"]["surfaces"],
        )


class CategoryInterventionTest(unittest.TestCase):
    def test_patch_admission_uses_synchronized_category_arm(self):
        outputs = _outputs()
        outputs["pred_logits_patch"] = torch.tensor([[4.0, 1.0, 0.0]])
        intervention = {
            "schema": "stageb-table-a-category-intervention-pair-v1",
            "pair_id": "cat-pair",
            "arm": "A",
            "image_width": 100,
            "image_height": 100,
            "class_a": {
                "id": 1,
                "name": "apple",
                "boxes_xyxy": [[10, 10, 20, 20]],
                "support_sha256": "support-a",
            },
            "class_b": {
                "id": 2,
                "name": "banana",
                "boxes_xyxy": [[40, 40, 60, 60]],
                "support_sha256": "support-b",
            },
            "active_class_id": 1,
            "active_class_name": "apple",
            "counterfactual_class_id": 2,
            "counterfactual_class_name": "banana",
            "canonical_prompt": "apple .",
            "active_support_path": "/support/apple.jpg",
            "active_support_sha256": "support-a",
        }
        records = role_eval.category_intervention_arm_records(
            outputs,
            [
                {
                    "image_id": 7,
                    "sample_id": "cat-pair:A",
                    "category_intervention": intervention,
                }
            ],
        )
        self.assertTrue(
            records[0]["candidate_admission"]["1"]["active_recall_iou50"]
        )
        self.assertFalse(
            records[0]["candidate_admission"]["1"][
                "counterfactual_recall_iou50"
            ]
        )
        self.assertTrue(records[0]["category_causal_evidence_eligible"])
        self.assertEqual(
            records[0]["category_causal_route"],
            "joint_canonical_prompt_plus_support_patch",
        )
        self.assertFalse(records[0]["patch_only_category_causal_claim_eligible"])

        arm_b = copy.deepcopy(records[0])
        arm_b.update(
            {
                "arm": "B",
                "active_class_id": 2,
                "active_class_name": "banana",
                "counterfactual_class_id": 1,
                "counterfactual_class_name": "apple",
                "active_support_sha256": "support-b",
            }
        )
        arm_b["patch_top1"].update(
            {
                "query_index": 1,
                "box_xyxy_normalized": [0.4, 0.4, 0.6, 0.6],
                "active_iou": 1.0,
                "counterfactual_iou": 0.0,
            }
        )
        for row in arm_b["candidate_admission"].values():
            row.update(
                {
                    "active_best_iou": 1.0,
                    "active_recall_iou50": True,
                    "counterfactual_best_iou": 0.0,
                    "counterfactual_recall_iou50": False,
                }
            )
        summary = role_eval.aggregate_category_intervention_records(
            [records[0], arm_b]
        )
        self.assertTrue(summary["category_causal_evidence_eligible"])
        self.assertEqual(
            summary["category_causal_route"],
            "joint_canonical_prompt_plus_support_patch",
        )
        self.assertFalse(summary["patch_only_category_causal_claim_eligible"])
        self.assertEqual(summary["top1_both_match_active_rate"], 1.0)
        self.assertEqual(summary["top1_query_change_rate"], 1.0)

    def test_runtime_assets_are_rehashed_and_bound_to_support_tsv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            support_a = root / "a.jpg"
            support_b = root / "b.jpg"
            image.write_bytes(b"image")
            support_a.write_bytes(b"support-a")
            support_b.write_bytes(b"support-b")

            def sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            support_tsv = root / "support.tsv"
            with support_tsv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "class"], delimiter="\t")
                writer.writeheader()
                writer.writerow({"path": str(support_a), "class": "a"})
                writer.writerow({"path": str(support_b), "class": "b"})
            intervention = {
                "image_path": str(image),
                "image_sha256": sha(image),
                "class_a": {
                    "support_path": str(support_a),
                    "support_sha256": sha(support_a),
                },
                "class_b": {
                    "support_path": str(support_b),
                    "support_sha256": sha(support_b),
                },
            }
            receipt = role_eval.verify_category_runtime_assets(
                [{"category_intervention": intervention}], support_tsv
            )
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["images_rehashed"], 1)
            self.assertEqual(receipt["supports_rehashed"], 2)
            support_a.write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                role_eval.verify_category_runtime_assets(
                    [{"category_intervention": intervention}], support_tsv
                )


if __name__ == "__main__":
    unittest.main()
