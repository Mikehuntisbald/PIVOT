import unittest

import torch

from engine import (
    _V52_CANDIDATE_ABSOLUTE_SAMPLE_CALIBRATOR_CONTRACT,
    _call_fixed_text_criterion_in_logical_batches,
    _select_dense_duty_confidence_loss_logits,
    _select_dense_duty_sample_confidence_logits,
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
)
from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)


def _criterion(**overrides):
    values = {
        "listwise_weight": 0.0,
        "local_tn_rank_weight": 0.0,
        "predicate_tn_rank_weight": 0.0,
        "local_anchor_weight": 0.0,
        "global_tn_negative_weight": 0.0,
        "global_tn_tail_weight": 0.0,
        "batch_tail_separation_weight": 0.0,
        "local_absolute_weight": 0.0,
        "predicate_absolute_weight": 0.0,
        "tail_queue_weight": 0.0,
    }
    values.update(overrides)
    return StageBFixedTextCriterion(**values)


class StageBV52LossPlumbingTest(unittest.TestCase):
    def test_engine_selects_live_candidate_and_exact_sample_logits(self):
        candidate_rank = torch.zeros(2, 3, 2)
        broadcast_global = torch.randn(2, 3, 2, requires_grad=True)
        candidate_absolute = torch.randn(2, 3, 2, requires_grad=True)
        sample_global = torch.randn(2, 2, requires_grad=True)
        outputs = {
            "stage_b_dense_duty_confidence_base_logits": candidate_absolute,
            "stage_b_dense_duty_global_confidence_logits": sample_global,
        }

        positive, negative = _select_dense_duty_confidence_loss_logits(
            outputs=outputs,
            candidate_logits=candidate_rank,
            confidence_logits=broadcast_global,
            head_gradient_contract=(
                _V52_CANDIDATE_ABSOLUTE_SAMPLE_CALIBRATOR_CONTRACT
            ),
        )
        sample_positive, sample_tn = _select_dense_duty_sample_confidence_logits(
            outputs=outputs,
            candidate_logits=candidate_rank,
            head_gradient_contract=(
                _V52_CANDIDATE_ABSOLUTE_SAMPLE_CALIBRATOR_CONTRACT
            ),
        )

        self.assertTrue(torch.equal(positive, candidate_absolute[..., 0]))
        self.assertTrue(torch.equal(negative, candidate_absolute[..., 1]))
        self.assertTrue(torch.equal(sample_positive, sample_global[..., 0]))
        self.assertTrue(torch.equal(sample_tn, sample_global[..., 1]))

        v53_positive, v53_negative = _select_dense_duty_confidence_loss_logits(
            outputs=outputs,
            candidate_logits=candidate_rank,
            confidence_logits=broadcast_global,
            head_gradient_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
        )
        v53_sample_positive, v53_sample_tn = (
            _select_dense_duty_sample_confidence_logits(
                outputs=outputs,
                candidate_logits=candidate_rank,
                head_gradient_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
            )
        )
        self.assertTrue(torch.equal(v53_positive, candidate_absolute[..., 0]))
        self.assertTrue(torch.equal(v53_negative, candidate_absolute[..., 1]))
        self.assertTrue(torch.equal(v53_sample_positive, sample_global[..., 0]))
        self.assertTrue(torch.equal(v53_sample_tn, sample_global[..., 1]))

        legacy_positive, legacy_negative = (
            _select_dense_duty_confidence_loss_logits(
                outputs=outputs,
                candidate_logits=candidate_rank,
                confidence_logits=broadcast_global,
                head_gradient_contract="split_token_veto_global_absolute_v2",
            )
        )
        self.assertTrue(torch.equal(legacy_positive, broadcast_global[..., 0]))
        self.assertTrue(torch.equal(legacy_negative, broadcast_global[..., 1]))

    def test_local_absolute_reaches_nonwinner_candidates_only(self):
        criterion = _criterion(local_absolute_weight=1.0)
        candidate_rank = torch.zeros(1, 3)
        candidate_ious = torch.tensor([[0.9, 0.8, 0.1]])
        positive_absolute = torch.tensor(
            [[0.4, -0.2, 4.0]], requires_grad=True
        )
        tn_absolute = torch.tensor([[-0.3, 0.7, 4.0]], requires_grad=True)
        sample_positive = torch.tensor([1.1], requires_grad=True)
        sample_tn = torch.tensor([0.5], requires_grad=True)

        losses = criterion(
            candidate_rank,
            candidate_ious,
            local_tn_logits=candidate_rank,
            confidence_logits=positive_absolute,
            local_tn_confidence_logits=tn_absolute,
            local_tn_mask=torch.ones(1, 3, dtype=torch.bool),
            sample_positive_confidence_logits=sample_positive,
            sample_tn_confidence_logits=sample_tn,
        )
        losses["loss_fixed_text_local_absolute"].backward()

        self.assertLess(float(positive_absolute.grad[0, 0]), 0.0)
        self.assertLess(float(positive_absolute.grad[0, 1]), 0.0)
        self.assertEqual(float(positive_absolute.grad[0, 2]), 0.0)
        self.assertGreater(float(tn_absolute.grad[0, 0]), 0.0)
        self.assertGreater(float(tn_absolute.grad[0, 1]), 0.0)
        self.assertEqual(float(tn_absolute.grad[0, 2]), 0.0)
        self.assertIsNone(sample_positive.grad)
        self.assertIsNone(sample_tn.grad)

    def test_sample_global_loss_matches_legacy_broadcast_scalar_once(self):
        legacy = _criterion(
            global_tn_negative_weight=1.0,
            global_tn_tail_weight=1.0,
            global_tn_tail_topk=3,
            global_tn_tail_target_logit=0.2,
        )
        split = _criterion(
            global_tn_negative_weight=1.0,
            global_tn_tail_weight=1.0,
            global_tn_tail_topk=3,
            global_tn_tail_target_logit=0.2,
        )
        candidate_rank = torch.zeros(2, 3)
        candidate_ious = torch.ones(2, 3)
        candidate_mask = torch.tensor(
            [[True, True, False], [True, True, True]]
        )
        sample_positive = torch.tensor([0.3, 0.6])
        sample_tn = torch.tensor([0.4, -0.7], requires_grad=True)
        broadcast_positive = sample_positive[:, None].expand(-1, 3)
        broadcast_tn = sample_tn.detach()[:, None].expand(-1, 3).clone()
        verified = torch.ones(2, dtype=torch.bool)

        legacy_losses = legacy(
            candidate_rank,
            candidate_ious,
            candidate_mask,
            local_tn_logits=candidate_rank,
            confidence_logits=broadcast_positive,
            local_tn_confidence_logits=broadcast_tn,
            local_tn_mask=candidate_mask,
            global_tn_logits=candidate_rank,
            global_tn_confidence_logits=broadcast_tn,
            global_tn_verified=verified,
            global_tn_candidate_mask=candidate_mask,
        )
        local_positive = torch.randn(2, 3, requires_grad=True)
        local_tn = torch.randn(2, 3, requires_grad=True)
        split_losses = split(
            candidate_rank,
            candidate_ious,
            candidate_mask,
            local_tn_logits=candidate_rank,
            confidence_logits=local_positive,
            local_tn_confidence_logits=local_tn,
            local_tn_mask=candidate_mask,
            sample_positive_confidence_logits=sample_positive,
            sample_tn_confidence_logits=sample_tn,
            global_tn_logits=candidate_rank,
            global_tn_verified=verified,
            global_tn_candidate_mask=candidate_mask,
        )

        self.assertTrue(
            torch.allclose(
                split_losses["loss_fixed_text_global_tn_negative"],
                legacy_losses["loss_fixed_text_global_tn_negative"],
            )
        )
        self.assertTrue(
            torch.allclose(
                split_losses["loss_fixed_text_global_tn_tail"],
                legacy_losses["loss_fixed_text_global_tn_tail"],
            )
        )
        self.assertTrue(
            torch.allclose(
                split_losses["loss_fixed_text_global_tn_negative"],
                torch.nn.functional.softplus(sample_tn).mean(),
            )
        )
        self.assertTrue(
            torch.allclose(
                split_losses["loss_fixed_text_global_tn_tail"],
                torch.nn.functional.softplus(sample_tn - 0.2).mean(),
            )
        )
        self.assertEqual(
            int(split_losses["fixed_text_global_tn_candidate_count"]), 5
        )

        (
            split_losses["loss_fixed_text_global_tn_negative"]
            + split_losses["loss_fixed_text_global_tn_tail"]
        ).backward()
        self.assertIsNotNone(sample_tn.grad)
        self.assertTrue(bool(sample_tn.grad.ne(0).all().item()))
        self.assertIsNone(local_positive.grad)
        self.assertIsNone(local_tn.grad)

    def test_fpr95_queue_consumes_sample_globals_not_candidate_maxima(self):
        criterion = _criterion(
            tail_queue_weight=1.0,
            tail_queue_size=8,
            tail_queue_min_count=2,
            tail_queue_positive_quantile=0.25,
            tail_queue_negative_quantile=0.75,
            tail_queue_temperature=0.2,
            tail_queue_margin=0.0,
            tail_queue_global_scores=True,
            tail_queue_objective="fpr95",
        )
        candidate_rank = torch.zeros(2, 3)
        candidate_ious = torch.tensor(
            [[0.9, 0.1, 0.1], [0.8, 0.2, 0.1]]
        )
        candidate_mask = torch.ones(2, 3, dtype=torch.bool)
        history_local_positive = torch.full((2, 3), 50.0)
        history_local_tn = torch.full((2, 3), 40.0)
        history_positive = torch.tensor([0.0, 1.0])
        history_tn = torch.tensor([-0.5, 0.5])

        criterion(
            candidate_rank,
            candidate_ious,
            candidate_mask,
            local_tn_logits=candidate_rank,
            confidence_logits=history_local_positive,
            local_tn_confidence_logits=history_local_tn,
            local_tn_mask=candidate_mask,
            sample_positive_confidence_logits=history_positive,
            sample_tn_confidence_logits=history_tn,
            positive_confidence_gate_logits=history_positive,
            global_tn_verified=torch.ones(2, dtype=torch.bool),
        )
        self.assertTrue(
            torch.equal(criterion._pending_tail_payload[:, 0], history_positive)
        )
        self.assertTrue(
            torch.equal(criterion._pending_tail_payload[:, 2], history_tn)
        )
        criterion.commit_tail_queue(True)

        local_positive = torch.full((2, 3), 60.0, requires_grad=True)
        local_tn = torch.full((2, 3), 55.0, requires_grad=True)
        sample_positive = torch.tensor([0.2, 0.8], requires_grad=True)
        sample_tn = torch.tensor([-0.8, 0.4], requires_grad=True)
        losses = criterion(
            candidate_rank,
            candidate_ious,
            candidate_mask,
            local_tn_logits=candidate_rank,
            confidence_logits=local_positive,
            local_tn_confidence_logits=local_tn,
            local_tn_mask=candidate_mask,
            sample_positive_confidence_logits=sample_positive,
            sample_tn_confidence_logits=sample_tn,
            positive_confidence_gate_logits=sample_positive,
            global_tn_verified=torch.ones(2, dtype=torch.bool),
        )
        losses["loss_fixed_text_tail_queue"].backward()

        self.assertIsNotNone(sample_positive.grad)
        self.assertIsNotNone(sample_tn.grad)
        self.assertTrue(bool(sample_tn.grad.gt(0).all().item()))
        self.assertIsNone(local_positive.grad)
        self.assertIsNone(local_tn.grad)
        criterion.commit_tail_queue(False)

    def test_logical_batches_keep_sample_scalar_rows_aligned(self):
        class SampleAwareCriterion:
            def __init__(self):
                self.calls = []

            def __call__(
                self,
                *,
                candidate_logits,
                candidate_ious,
                sample_positive_confidence_logits,
                sample_tn_confidence_logits,
            ):
                self.calls.append(
                    (
                        sample_positive_confidence_logits.detach().clone(),
                        sample_tn_confidence_logits.detach().clone(),
                    )
                )
                return {
                    "loss": candidate_logits.mean(),
                    "count": candidate_ious.new_tensor(
                        float(candidate_logits.shape[0])
                    ),
                }

            def defer_tail_queue_payload(self):
                return None

        criterion = SampleAwareCriterion()
        sample_positive = torch.tensor([0.1, 0.2, 0.3, 0.4])
        sample_tn = torch.tensor([-0.1, -0.2, -0.3, -0.4])
        _, logical_batches = _call_fixed_text_criterion_in_logical_batches(
            criterion,
            logical_batch_size=2,
            candidate_logits=torch.zeros(4, 2),
            candidate_ious=torch.ones(4, 2),
            sample_positive_confidence_logits=sample_positive,
            sample_tn_confidence_logits=sample_tn,
        )

        self.assertEqual(logical_batches, 2)
        self.assertTrue(torch.equal(criterion.calls[0][0], sample_positive[:2]))
        self.assertTrue(torch.equal(criterion.calls[1][0], sample_positive[2:]))
        self.assertTrue(torch.equal(criterion.calls[0][1], sample_tn[:2]))
        self.assertTrue(torch.equal(criterion.calls[1][1], sample_tn[2:]))


if __name__ == "__main__":
    unittest.main()
