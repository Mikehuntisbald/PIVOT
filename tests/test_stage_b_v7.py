import unittest

import torch

from models.GroundingDINO.stage_b_v7 import (
    StageBV7Criterion,
    build_stage_b_v7_candidate_scores,
    calibrate_finite_token_logits,
)


class _DummyPatchCriterion:
    matcher = None

    def compute_matching(self, outputs, targets):
        device = outputs["pred_boxes"].device
        return {
            "pred_boxes": outputs["pred_boxes"],
            "all_indices": [
                (
                    torch.tensor([0], dtype=torch.long, device=device),
                    torch.tensor([0], dtype=torch.long, device=device),
                )
            ],
            "matched_patch_idx_list": [torch.tensor([0], dtype=torch.long, device=device)],
        }


class StageBV7Test(unittest.TestCase):
    def test_finite_calibration_has_finite_gradients(self):
        token_logits = torch.tensor([[[1.0, float("-inf"), float("inf")]]])
        scale = torch.zeros((), requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        calibrated = calibrate_finite_token_logits(token_logits, scale, bias)
        calibrated.sum().backward()
        self.assertTrue(torch.isfinite(calibrated).all())
        self.assertTrue(torch.isfinite(scale.grad))
        self.assertTrue(torch.isfinite(bias.grad))

    def test_patch_is_gate_and_text_orders_candidates(self):
        predicate_logits = torch.tensor([[[0.0], [3.0], [10.0], [-1.0]]])
        patch_score = torch.tensor([[[0.9], [0.8], [0.1], [0.0]]])
        scored = build_stage_b_v7_candidate_scores(
            predicate_logits,
            patch_score,
            candidate_topk=2,
            patch_prior_weight=0.0,
        )
        self.assertEqual(scored["candidate_mask"].squeeze(-1).tolist(), [[True, True, False, False]])
        self.assertEqual(int(scored["final_score"].reshape(-1).argmax().item()), 1)

    def test_non_candidate_cannot_win_when_candidate_logit_is_very_low(self):
        predicate_logits = torch.tensor([[[-30.0], [10.0]]])
        patch_score = torch.tensor([[[0.9], [0.1]]])
        scored = build_stage_b_v7_candidate_scores(
            predicate_logits,
            patch_score,
            candidate_topk=1,
            patch_prior_weight=0.0,
        )
        self.assertEqual(scored["candidate_mask"].squeeze(-1).tolist(), [[True, False]])
        self.assertEqual(int(scored["final_logits"].reshape(-1).argmax().item()), 0)
        self.assertEqual(float(scored["final_score"][0, 1, 0]), 0.0)

    def test_mixed_pair_stride_masks_padded_phrase_slot(self):
        predicate_logits = torch.zeros((2, 4, 2))
        patch_score = torch.tensor(
            [
                [[0.9], [0.8], [0.1], [0.0]],
                [[0.7], [0.6], [0.2], [0.1]],
            ]
        )
        scored = build_stage_b_v7_candidate_scores(
            predicate_logits,
            patch_score,
            candidate_topk=2,
            pair_stride=torch.tensor([[1], [2]]),
        )
        self.assertEqual(int(scored["candidate_mask"][0, :, 0].sum()), 2)
        self.assertEqual(int(scored["candidate_mask"][0, :, 1].sum()), 0)
        self.assertEqual(int(scored["candidate_mask"][1, :, 0].sum()), 2)
        self.assertEqual(int(scored["candidate_mask"][1, :, 1].sum()), 2)

    def _base_outputs(self, num_slots):
        pred_boxes = torch.tensor(
            [[[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]]
        )
        predicate_logits = torch.zeros((1, 3, num_slots), requires_grad=True)
        return {
            "pred_logits_patch": torch.zeros((1, 3, 1)),
            "pred_boxes": pred_boxes,
            "stage_b_v7_predicate_logits": predicate_logits,
            "stage_b_v7_predicate_token_logits": torch.zeros((1, 3, 4), requires_grad=True),
            "stage_b_v7_candidate_mask": torch.ones((1, 3, num_slots), dtype=torch.bool),
            "stage_b_v7_final_logits": predicate_logits,
            "stage_b_v7_expanded_patch_score": torch.full((1, 3, num_slots), 0.5),
        }

    def test_clean_positive_is_not_overwritten_as_negative(self):
        criterion = StageBV7Criterion(patch_criterion=_DummyPatchCriterion())
        outputs = self._base_outputs(num_slots=1)
        outputs.update(
            {
                "phrase_to_token_mask": torch.tensor([[[1, 1, 0, 0]]], dtype=torch.bool),
                "canonical_to_token_mask": torch.tensor([[[0, 1, 0, 0]]], dtype=torch.bool),
                "content_to_token_mask": torch.tensor([[[1, 0, 0, 0]]], dtype=torch.bool),
                "is_tn": torch.tensor([[False]]),
                "verifier_pair_stride": torch.tensor([[1]]),
            }
        )
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]
        losses = criterion(outputs, targets)
        self.assertGreater(float(losses["stage_b_v7_phrase_positive_count"]), 0.0)
        self.assertEqual(float(losses["stage_b_v7_phrase_tn_negative_count"]), 0.0)
        self.assertTrue(torch.isfinite(losses["loss_verifier_phrase_focal"]))

    def test_tn_supervision_is_local_to_target_candidates(self):
        criterion = StageBV7Criterion(patch_criterion=_DummyPatchCriterion())
        outputs = self._base_outputs(num_slots=2)
        outputs.update(
            {
                "phrase_to_token_mask": torch.tensor(
                    [[
                        [1, 1, 0, 0],
                        [0, 0, 1, 1],
                    ]],
                    dtype=torch.bool,
                ),
                "canonical_to_token_mask": torch.zeros((1, 2, 4), dtype=torch.bool),
                "content_to_token_mask": torch.tensor(
                    [[
                        [1, 1, 0, 0],
                        [0, 0, 1, 1],
                    ]],
                    dtype=torch.bool,
                ),
                "attr_neg_to_token_mask": torch.tensor(
                    [[
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                    ]],
                    dtype=torch.bool,
                ),
                "is_tn": torch.tensor([[False, True]]),
                "verifier_pair_stride": torch.tensor([[2]]),
            }
        )
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]
        losses = criterion(outputs, targets)
        self.assertEqual(float(losses["stage_b_v7_phrase_positive_count"]), 1.0)
        self.assertEqual(float(losses["stage_b_v7_phrase_tn_negative_count"]), 1.0)
        self.assertEqual(float(losses["stage_b_v7_phrase_distractor_count"]), 2.0)

    def test_explicit_tn_rank_compares_the_same_candidate(self):
        criterion = StageBV7Criterion(
            patch_criterion=_DummyPatchCriterion(),
            tn_pair_rank_loss_coef=1.0,
            tn_pair_rank_margin=0.3,
        )
        outputs = self._base_outputs(num_slots=2)
        predicate_logits = torch.tensor(
            [[[1.0, 2.0], [8.0, -3.0], [7.0, -4.0]]], requires_grad=True
        )
        outputs.update(
            {
                "stage_b_v7_predicate_logits": predicate_logits,
                "stage_b_v7_final_logits": predicate_logits,
                "phrase_to_token_mask": torch.tensor(
                    [[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=torch.bool
                ),
                "canonical_to_token_mask": torch.zeros((1, 2, 4), dtype=torch.bool),
                "content_to_token_mask": torch.tensor(
                    [[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=torch.bool
                ),
                "attr_neg_to_token_mask": torch.tensor(
                    [[[0, 0, 0, 0], [0, 0, 1, 0]]], dtype=torch.bool
                ),
                "is_tn": torch.tensor([[False, True]]),
                "verifier_pair_stride": torch.tensor([[2]]),
            }
        )
        targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])}]
        losses = criterion(outputs, targets)

        self.assertIn("loss_verifier_tn_pair_rank", criterion.weight_dict)
        self.assertEqual(float(losses["stage_b_v7_tn_pair_count"]), 1.0)
        self.assertLess(float(losses["stage_b_v7_tn_pair_score_gap"]), 0.0)
        losses["loss_verifier_tn_pair_rank"].backward()
        self.assertLess(float(predicate_logits.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(predicate_logits.grad[0, 0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
