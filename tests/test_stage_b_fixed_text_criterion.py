import unittest

import torch

from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
    candidate_max_iou,
)


class StageBFixedTextCriterionTest(unittest.TestCase):
    def test_candidate_max_iou_uses_all_target_boxes(self):
        candidates = torch.tensor(
            [[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]]]
        )
        targets = [
            {
                "boxes": torch.tensor(
                    [[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]]
                )
            }
        ]
        ious = candidate_max_iou(candidates, targets)
        self.assertTrue(torch.allclose(ious[0, :2], torch.ones(2), atol=1e-4))
        self.assertEqual(float(ious[0, 2]), 0.0)

    def test_multi_positive_listwise_gradient_directions(self):
        criterion = StageBFixedTextCriterion(
            local_tn_rank_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
        )
        logits = torch.tensor([[0.1, -0.2, 1.2, -0.3, 5.0]], requires_grad=True)
        ious = torch.tensor([[0.8, 0.6, 0.1, 0.2, 0.4]])
        losses = criterion(logits, ious)
        losses["loss_fixed_text_listwise"].backward()

        self.assertEqual(float(losses["fixed_text_valid_listwise_count"]), 1.0)
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 2]), 0.0)
        self.assertGreater(float(logits.grad[0, 3]), 0.0)
        self.assertEqual(float(logits.grad[0, 4]), 0.0)

    def test_missing_positive_or_negative_returns_graph_connected_zero(self):
        criterion = StageBFixedTextCriterion(
            local_tn_rank_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
        )
        logits = torch.tensor([[0.2, -0.1], [0.4, 0.3]], requires_grad=True)
        ious = torch.tensor([[0.2, 0.1], [0.8, 0.7]])
        loss = criterion(logits, ious)["loss_fixed_text_listwise"]
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_local_tn_rank_only_compares_same_positive_queries(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
        )
        positive = torch.tensor([[0.0, 0.5, -1.0, 2.0]], requires_grad=True)
        local_tn = torch.tensor([[1.0, -0.5, 4.0, 6.0]], requires_grad=True)
        ious = torch.tensor([[0.9, 0.6, 0.1, 0.4]])
        losses = criterion(positive, ious, local_tn_logits=local_tn)
        losses["loss_fixed_text_local_tn_rank"].backward()

        self.assertEqual(float(losses["fixed_text_local_pair_query_count"]), 2.0)
        self.assertLess(float(positive.grad[0, 0]), 0.0)
        self.assertLess(float(positive.grad[0, 1]), 0.0)
        self.assertEqual(float(positive.grad[0, 2]), 0.0)
        self.assertEqual(float(positive.grad[0, 3]), 0.0)
        self.assertGreater(float(local_tn.grad[0, 0]), 0.0)
        self.assertGreater(float(local_tn.grad[0, 1]), 0.0)
        self.assertEqual(float(local_tn.grad[0, 2]), 0.0)
        self.assertEqual(float(local_tn.grad[0, 3]), 0.0)

    def test_local_tn_mask_can_disable_clean_second_slot(self):
        criterion = StageBFixedTextCriterion()
        positive = torch.tensor([[0.0, 0.5]], requires_grad=True)
        local_tn = torch.tensor([[3.0, 4.0]], requires_grad=True)
        ious = torch.tensor([[0.9, 0.1]])
        losses = criterion(
            positive,
            ious,
            local_tn_logits=local_tn,
            local_tn_mask=torch.tensor([False]),
        )
        loss = losses["loss_fixed_text_local_tn_rank"]
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(local_tn.grad)
        self.assertEqual(float(local_tn.grad.abs().sum()), 0.0)

    def test_clean_invalid_slot_padding_keeps_all_losses_finite(self):
        criterion = StageBFixedTextCriterion()
        positive = torch.zeros((1, 50), requires_grad=True)
        invalid_tn = torch.full(
            (1, 50), torch.finfo(torch.float32).min, requires_grad=True
        )
        ious = torch.tensor([[0.8] + [0.1] * 49])
        losses = criterion(
            positive,
            ious,
            local_tn_logits=invalid_tn,
            local_tn_mask=torch.tensor([[False]]),
        )
        loss_values = [
            value
            for key, value in losses.items()
            if key.startswith("loss_")
        ]
        self.assertTrue(all(bool(torch.isfinite(value).all().item()) for value in loss_values))
        losses["loss_stage_b_fixed_text"].backward()
        self.assertTrue(bool(torch.isfinite(positive.grad).all().item()))
        self.assertTrue(bool(torch.isfinite(invalid_tn.grad).all().item()))
        self.assertEqual(float(invalid_tn.grad.abs().sum()), 0.0)

    def test_local_anchor_identifies_common_logit_shift_and_gradient_directions(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=1.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            batch_tail_separation_weight=0.0,
            positive_anchor_logit=0.5,
            negative_anchor_logit=-0.5,
        )
        ious = torch.tensor([[0.9, 0.1]])

        def anchor_at_shift(shift):
            positive = torch.tensor([[shift, 0.0]], requires_grad=True)
            local_tn = torch.tensor([[shift, 0.0]], requires_grad=True)
            losses = criterion(positive, ious, local_tn_logits=local_tn)
            loss = losses["loss_fixed_text_local_anchor"]
            return loss, positive, local_tn, losses

        centered, positive, local_tn, losses = anchor_at_shift(0.0)
        shifted_high, _p_high, _n_high, _ = anchor_at_shift(5.0)
        shifted_low, _p_low, _n_low, _ = anchor_at_shift(-5.0)
        self.assertEqual(
            criterion.weight_dict["loss_fixed_text_local_anchor"],
            1.0,
        )
        self.assertIn("loss_fixed_text_local_anchor", losses)
        self.assertGreater(float(shifted_high.detach()), float(centered.detach()))
        self.assertGreater(float(shifted_low.detach()), float(centered.detach()))
        self.assertEqual(float(losses["fixed_text_local_anchor_sample_count"]), 1.0)
        self.assertEqual(float(losses["fixed_text_local_anchor_paired_sample_count"]), 1.0)
        self.assertEqual(float(losses["fixed_text_local_anchor_positive_query_count"]), 1.0)
        self.assertEqual(float(losses["fixed_text_local_anchor_tn_query_count"]), 1.0)

        centered.backward()
        self.assertLess(float(positive.grad[0, 0]), 0.0)
        self.assertGreater(float(local_tn.grad[0, 0]), 0.0)
        self.assertEqual(float(positive.grad[0, 1]), 0.0)
        self.assertEqual(float(local_tn.grad[0, 1]), 0.0)

    def test_clean_positive_keeps_absolute_anchor_without_tn(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=1.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            batch_tail_separation_weight=0.0,
        )
        positive = torch.tensor([[-2.0, 3.0]], requires_grad=True)
        ious = torch.tensor([[0.9, 0.1]])
        losses = criterion(positive, ious)
        losses["loss_stage_b_fixed_text"].backward()
        self.assertEqual(float(losses["fixed_text_local_anchor_sample_count"]), 1.0)
        self.assertEqual(float(losses["fixed_text_local_anchor_paired_sample_count"]), 0.0)
        self.assertEqual(float(losses["fixed_text_local_anchor_positive_query_count"]), 1.0)
        self.assertEqual(float(losses["fixed_text_local_anchor_tn_query_count"]), 0.0)
        self.assertLess(float(positive.grad[0, 0]), 0.0)
        self.assertEqual(float(positive.grad[0, 1]), 0.0)

    def test_global_tn_requires_explicit_verification(self):
        criterion = StageBFixedTextCriterion()
        logits = torch.zeros((1, 2))
        ious = torch.tensor([[0.8, 0.1]])
        with self.assertRaisesRegex(ValueError, "global_tn_verified"):
            criterion(logits, ious, global_tn_logits=torch.zeros_like(logits))

    def test_global_verified_tn_trains_all_candidates_and_tail(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            global_tn_tail_topk=2,
        )
        positive = torch.zeros((2, 4), requires_grad=True)
        global_tn = torch.tensor(
            [[2.0, 1.0, -1.0, -2.0], [8.0, 8.0, 8.0, 8.0]], requires_grad=True
        )
        ious = torch.tensor([[0.8, 0.2, 0.1, 0.4], [0.9, 0.2, 0.1, 0.0]])
        losses = criterion(
            positive,
            ious,
            global_tn_logits=global_tn,
            global_tn_verified=torch.tensor([True, False]),
        )
        global_loss = (
            losses["loss_fixed_text_global_tn_negative"]
            + losses["loss_fixed_text_global_tn_tail"]
        )
        global_loss.backward()

        self.assertEqual(float(losses["fixed_text_global_tn_sample_count"]), 1.0)
        self.assertEqual(float(losses["fixed_text_global_tn_candidate_count"]), 4.0)
        self.assertTrue(bool((global_tn.grad[0] > 0).all().item()))
        self.assertEqual(float(global_tn.grad[1].abs().sum()), 0.0)

    def test_batch_tail_is_disabled_by_default_but_has_correct_gradients(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            batch_tail_separation_weight=1.0,
        )
        positive = torch.tensor([[-1.0, 0.0]], requires_grad=True)
        local_tn = torch.tensor([[1.0, -2.0]], requires_grad=True)
        ious = torch.tensor([[0.9, 0.1]])
        losses = criterion(positive, ious, local_tn_logits=local_tn)
        losses["loss_stage_b_fixed_text"].backward()

        self.assertGreater(float(losses["loss_fixed_text_batch_tail"].detach()), 0.0)
        self.assertLess(float(positive.grad[0, 0]), 0.0)
        self.assertGreater(float(local_tn.grad[0, 0]), 0.0)
        self.assertEqual(
            StageBFixedTextCriterion().weight_dict["loss_fixed_text_batch_tail"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
