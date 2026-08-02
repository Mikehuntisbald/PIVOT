import inspect
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import models.GroundingDINO.stage_b_fixed_text_criterion as criterion_module
from config.ablations import cfg_stageb_v13_calibrated_tail as v13_config
from models.GroundingDINO.groundingdino import build_groundingdino
from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
    fixed_batch_tail_separation_loss,
)


def _anchor_only_criterion(**kwargs):
    options = dict(
        listwise_weight=0.0,
        local_tn_rank_weight=0.0,
        predicate_tn_rank_weight=0.0,
        local_anchor_weight=1.0,
        global_tn_negative_weight=0.0,
        global_tn_tail_weight=0.0,
        batch_tail_separation_weight=0.0,
    )
    options.update(kwargs)
    return StageBFixedTextCriterion(**options)


class StageBV13CalibratedTailTest(unittest.TestCase):
    def _mixed_anchor_inputs(self):
        positive = torch.tensor(
            [[0.0, 1.0, 4.0], [-2.0, 3.0, 4.0]], requires_grad=True
        )
        local_tn = torch.tensor(
            [[-1.0, 2.0, 4.0], [8.0, 8.0, 8.0]], requires_grad=True
        )
        ious = torch.tensor([[0.9, 0.7, 0.1], [0.8, 0.1, 0.4]])
        local_valid = torch.tensor([True, False])
        return positive, local_tn, ious, local_valid

    def test_default_anchor_reduction_is_strictly_legacy(self):
        self.assertFalse(StageBFixedTextCriterion().balance_local_anchor_classes)
        positive, local_tn, ious, local_valid = self._mixed_anchor_inputs()
        default = _anchor_only_criterion()
        explicit_legacy = _anchor_only_criterion(
            balance_local_anchor_classes=False
        )
        default_loss = default(
            positive,
            ious,
            local_tn_logits=local_tn,
            local_tn_mask=local_valid,
        )["loss_fixed_text_local_anchor"]
        explicit_loss = explicit_legacy(
            positive,
            ious,
            local_tn_logits=local_tn,
            local_tn_mask=local_valid,
        )["loss_fixed_text_local_anchor"]

        positive_sample_0 = F.softplus(0.5 - positive[0, :2]).mean()
        positive_sample_1 = F.softplus(0.5 - positive[1, :1]).mean()
        tn_sample_0 = F.softplus(local_tn[0, :2] + 0.5).mean()
        expected_legacy = (
            0.5 * positive_sample_0 + 0.5 * tn_sample_0 + positive_sample_1
        ) / 2.0
        self.assertTrue(torch.equal(default_loss, explicit_loss))
        self.assertTrue(torch.allclose(default_loss, expected_legacy))

    def test_balanced_anchor_reduces_queries_then_samples_then_classes(self):
        positive, local_tn, ious, local_valid = self._mixed_anchor_inputs()
        criterion = _anchor_only_criterion(balance_local_anchor_classes=True)
        loss = criterion(
            positive,
            ious,
            local_tn_logits=local_tn,
            local_tn_mask=local_valid,
        )["loss_fixed_text_local_anchor"]

        positive_sample_0 = F.softplus(0.5 - positive[0, :2]).mean()
        positive_sample_1 = F.softplus(0.5 - positive[1, :1]).mean()
        tn_sample_0 = F.softplus(local_tn[0, :2] + 0.5).mean()
        expected = (
            0.5 * torch.stack((positive_sample_0, positive_sample_1)).mean()
            + 0.5 * tn_sample_0
        )
        self.assertTrue(torch.allclose(loss, expected))

        loss.backward()
        self.assertTrue(bool((positive.grad[0, :2] < 0).all().item()))
        self.assertLess(float(positive.grad[1, 0]), 0.0)
        self.assertTrue(bool((local_tn.grad[0, :2] > 0).all().item()))
        self.assertEqual(float(positive.grad[:, 2].abs().sum()), 0.0)
        self.assertEqual(float(local_tn.grad[1].abs().sum()), 0.0)

    def test_balanced_anchor_single_class_falls_back_without_halving(self):
        criterion = _anchor_only_criterion(balance_local_anchor_classes=True)
        positive = torch.tensor([[0.0, 9.0], [-1.0, 9.0]], requires_grad=True)
        ious = torch.tensor([[0.9, 0.1], [0.8, 0.1]])
        loss = criterion(positive, ious)["loss_fixed_text_local_anchor"]
        expected = torch.stack(
            (F.softplus(0.5 - positive[0, 0]), F.softplus(0.5 - positive[1, 0]))
        ).mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(bool((positive.grad[:, 0] < 0).all().item()))
        self.assertEqual(float(positive.grad[:, 1].abs().sum()), 0.0)

    def test_fixed_tail_local_fp32_math_autocast_and_gradients(self):
        positive = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        negative = torch.tensor([1.0, -2.0, 0.25], requires_grad=True)
        positive_valid = torch.tensor([True, True, False])
        negative_valid = torch.tensor([True, False, True])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = fixed_batch_tail_separation_loss(
                positive,
                positive_valid,
                negative,
                negative_valid,
                positive_quantile=0.0,
                negative_quantile=1.0,
                margin=0.3,
                ddp_global=True,
            )
        expected = F.softplus(torch.tensor(1.0 - (-1.0) + 0.3))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.allclose(loss.detach(), expected))
        loss.backward()
        self.assertLess(float(positive.grad[0]), 0.0)
        self.assertGreater(float(negative.grad[0]), 0.0)
        self.assertEqual(float(positive.grad[2]), 0.0)
        self.assertEqual(float(negative.grad[1]), 0.0)

    def test_default_batch_tail_reduction_is_strictly_local_legacy(self):
        self.assertFalse(StageBFixedTextCriterion().batch_tail_ddp_global)
        options = dict(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            batch_tail_separation_weight=1.0,
            batch_positive_quantile=0.05,
            batch_negative_quantile=0.95,
        )
        default = StageBFixedTextCriterion(**options)
        explicit_legacy = StageBFixedTextCriterion(
            **options, batch_tail_ddp_global=False
        )
        positive = torch.tensor([[-1.0, 4.0], [0.5, 4.0]])
        local_tn = torch.tensor([[1.0, 4.0], [0.25, 4.0]])
        ious = torch.tensor([[0.9, 0.1], [0.8, 0.1]])
        default_loss = default(
            positive, ious, local_tn_logits=local_tn
        )["loss_fixed_text_batch_tail"]
        explicit_loss = explicit_legacy(
            positive, ious, local_tn_logits=local_tn
        )["loss_fixed_text_batch_tail"]
        positive_tail = torch.quantile(positive[:, 0], 0.05)
        negative_tail = torch.quantile(local_tn[:, 0], 0.95)
        expected = F.softplus(negative_tail - positive_tail + 0.3)
        self.assertTrue(torch.equal(default_loss, explicit_loss))
        self.assertTrue(torch.allclose(default_loss, expected))

    def test_fixed_tail_empty_class_is_graph_connected_zero(self):
        positive = torch.tensor([0.2, -0.1], requires_grad=True)
        negative = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = fixed_batch_tail_separation_loss(
            positive,
            torch.tensor([False, False]),
            negative,
            torch.tensor([True, True]),
            positive_quantile=0.05,
            negative_quantile=0.95,
            margin=0.3,
            ddp_global=False,
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(positive.grad)
        self.assertIsNotNone(negative.grad)
        self.assertEqual(float(positive.grad.abs().sum()), 0.0)
        self.assertEqual(float(negative.grad.abs().sum()), 0.0)

    def test_ddp_global_gathers_fixed_payload_even_when_local_rank_is_empty(self):
        positive = torch.tensor([5.0, 6.0], requires_grad=True)
        negative = torch.tensor([7.0, 8.0], requires_grad=True)
        gathered_payloads = []

        def fake_all_gather(payload):
            gathered_payloads.append(payload)
            remote = torch.tensor(
                [[0.2, 1.0, 0.8, 1.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=payload.dtype,
                device=payload.device,
            )
            return payload, remote

        with (
            mock.patch.object(criterion_module.dist, "is_available", return_value=True),
            mock.patch.object(criterion_module.dist, "is_initialized", return_value=True),
            mock.patch.object(criterion_module.dist, "get_world_size", return_value=2),
            mock.patch.object(
                criterion_module.dist_nn_functional,
                "all_gather",
                side_effect=fake_all_gather,
            ) as gather,
        ):
            loss = fixed_batch_tail_separation_loss(
                positive,
                torch.tensor([False, False]),
                negative,
                torch.tensor([False, False]),
                positive_quantile=0.05,
                negative_quantile=0.95,
                margin=0.3,
                ddp_global=True,
            )

        gather.assert_called_once()
        self.assertEqual(tuple(gathered_payloads[0].shape), (2, 4))
        self.assertEqual(gathered_payloads[0][:, [1, 3]].tolist(), [[0.0, 0.0]] * 2)
        self.assertTrue(
            torch.allclose(loss, F.softplus(torch.tensor(0.8 - 0.2 + 0.3)))
        )

    def test_v13_config_and_builder_enable_only_criterion_switches(self):
        self.assertTrue(v13_config.stage_b_v11_balance_local_anchor_classes)
        self.assertTrue(v13_config.stage_b_v11_batch_tail_ddp_global)
        source = inspect.getsource(build_groundingdino)
        self.assertIn("stage_b_v11_balance_local_anchor_classes", source)
        self.assertIn("stage_b_v11_batch_tail_ddp_global", source)
        self.assertEqual(StageBFixedTextCriterion().state_dict(), {})


if __name__ == "__main__":
    unittest.main()
