import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from models.GroundingDINO.stage_b_gdino_score_adapter import (
    StageBGDINOScoreAdapter,
    aggregate_gdino_full_expression_score,
    baseline_preserving_top1_rank_loss,
    detached_recent_q05_trust_surrogate,
    distributed_gather_1d_with_local_grad,
    exact_tpr_operating_threshold,
    fpr95_global_max_surrogate,
    multi_positive_listwise_rank_loss,
)


class _P3ToyGate(nn.Module):
    def __init__(self, initial_weight):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(initial_weight)))

    def forward(self, features):
        return features * self.weight


class _RankToyResidual(nn.Module):
    def __init__(self, initial_weight):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(initial_weight)))

    def forward(self, features):
        return features * self.weight


def _rank_ddp_inputs():
    fix_row = torch.tensor([True, True, True, False, True, False, False, False])
    base = torch.empty(8, 2)
    base[fix_row] = torch.tensor([0.15, 0.50])
    base[~fix_row] = torch.tensor([0.80, 0.20])
    features = torch.tensor(
        [
            [1.0, -0.2],
            [0.5, -0.7],
            [1.5, 0.1],
            [-1.0, 0.5],
            [0.8, -0.4],
            [-0.7, 0.6],
            [-1.4, 0.2],
            [-0.8, 0.9],
        ]
    )
    iou = torch.tensor([[0.75, 0.20]]).expand(8, 2)
    return base, features, iou


def _rank_ddp_gradient_worker(
    rank,
    world_size,
    init_file,
    reference_gradient,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        base, features, iou = _rank_ddp_inputs()
        local_slice = slice(4 * rank, 4 * (rank + 1))
        model = DistributedDataParallel(_RankToyResidual(0.1))
        residual = model(features[local_slice])
        result = baseline_preserving_top1_rank_loss(
            base[local_slice] + residual,
            base[local_slice],
            residual,
            iou[local_slice],
            preserve_margin=0.1,
            residual_weight=0.0,
        )
        result.loss.backward()
        if not torch.allclose(
            model.module.weight.grad,
            torch.tensor(reference_gradient),
            atol=1e-6,
            rtol=0.0,
        ):
            raise AssertionError(
                f"rank {rank} DDP rank-loss gradient differs from the global "
                f"class-normalized reference: {float(model.module.weight.grad)} "
                f"!= {reference_gradient}"
            )
    finally:
        dist.destroy_process_group()


def _p3_ddp_gradient_worker(
    rank,
    world_size,
    init_file,
    reference_gradient,
    reference_loss,
):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        positive_base = torch.tensor([0.30, 0.45, 0.25, 0.55])
        negative_base = torch.tensor([0.50, 0.10, 0.35, 0.20])
        positive_feature = torch.tensor([1.0, -0.5, 0.7, -1.2])
        negative_feature = torch.tensor([-0.4, 1.1, -0.8, 0.3])
        history = torch.tensor([0.22, 0.28, 0.34, 0.41, 0.50])
        local_slice = slice(2 * rank, 2 * (rank + 1))
        model = DistributedDataParallel(_P3ToyGate(-0.08))
        gates = model(
            torch.stack(
                (
                    positive_feature[local_slice],
                    negative_feature[local_slice],
                )
            )
        )
        positive_gate, negative_gate = gates[0], gates[1]
        result = detached_recent_q05_trust_surrogate(
            (positive_base[local_slice] + positive_gate)[:, None],
            (negative_base[local_slice] + negative_gate)[:, None],
            positive_gate,
            negative_gate,
            positive_history=history,
            paired_margin_weight=0.25,
            paired_margin=0.05,
        )
        if not torch.allclose(
            result.loss.detach(), torch.tensor(reference_loss), atol=1e-7, rtol=0.0
        ):
            raise AssertionError(
                f"rank {rank} P3 loss differs from global reference: "
                f"{float(result.loss)} != {reference_loss}"
            )
        result.loss.backward()
        if not torch.allclose(
            model.module.weight.grad,
            torch.tensor(reference_gradient),
            atol=1e-6,
            rtol=0.0,
        ):
            raise AssertionError(
                f"rank {rank} DDP gradient differs from global reference: "
                f"{float(model.module.weight.grad)} != {reference_gradient}"
            )
    finally:
        dist.destroy_process_group()


class StageBGDINOScoreAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_full_expression_score_matches_evaluator_formula(self):
        token_logits = torch.tensor(
            [
                [[-2.0, 0.0, 2.0, 4.0], [1.0, -1.0, 3.0, -3.0]],
                [[0.2, 0.4, 0.6, 0.8], [-0.2, -0.4, -0.6, -0.8]],
            ]
        )
        mask = torch.tensor([[False, True, True, False], [True, False, False, True]])

        observed = aggregate_gdino_full_expression_score(token_logits, mask)
        expected = torch.stack(
            (
                token_logits[0].sigmoid()[:, [1, 2]].mean(dim=-1),
                token_logits[1].sigmoid()[:, [0, 3]].mean(dim=-1),
            )
        )

        self.assertTrue(torch.equal(observed, expected))
        with self.assertRaisesRegex(ValueError, "at least one scored token"):
            aggregate_gdino_full_expression_score(
                token_logits, torch.zeros_like(mask)
            )

    def test_zero_initialization_is_bitwise_base_identity(self):
        adapter = StageBGDINOScoreAdapter(
            hidden_dim=16, adapter_dim=12, gate_hidden_dim=10
        )
        query_hs = torch.randn(3, 9, 16)
        base_score = torch.rand(3, 9)
        candidate_mask = torch.tensor(
            [
                [True] * 9,
                [True] * 7 + [False] * 2,
                [True] + [False] * 8,
            ]
        )
        base_score = base_score.masked_fill(~candidate_mask, -torch.inf)

        output = adapter(query_hs, base_score, candidate_mask)

        self.assertTrue(torch.equal(output["rank_residual"], torch.zeros(3, 9)))
        self.assertTrue(torch.equal(output["confidence_gate"], torch.zeros(3)))
        self.assertTrue(torch.equal(output["rank_score"], base_score))
        self.assertTrue(torch.equal(output["confidence_score"], base_score))
        self.assertTrue(
            torch.equal(
                output["rank_score"].argmax(dim=1), base_score.argmax(dim=1)
            )
        )

    def test_parameter_sets_are_disjoint(self):
        adapter = StageBGDINOScoreAdapter(hidden_dim=8, adapter_dim=6, gate_hidden_dim=5)
        rank_ids = {id(parameter) for parameter in adapter.rank_parameters()}
        gate_ids = {id(parameter) for parameter in adapter.gate_parameters()}

        self.assertTrue(rank_ids)
        self.assertTrue(gate_ids)
        self.assertFalse(rank_ids & gate_ids)
        self.assertEqual(
            rank_ids | gate_ids, {id(parameter) for parameter in adapter.parameters()}
        )

    def test_low_temperature_pool_tracks_the_global_max_query(self):
        adapter = StageBGDINOScoreAdapter(
            hidden_dim=4,
            adapter_dim=3,
            gate_hidden_dim=5,
            gate_pool_temperature=0.01,
            gate_topk=3,
        )
        confidence_feature = torch.zeros(1, 900, 3)
        confidence_feature[0, 0, 0] = 1.0
        base_score = torch.full((1, 900), 0.2)
        base_score[0, 0] = 0.3
        gate_input = adapter._gate_inputs(
            confidence_feature,
            base_score,
            torch.ones_like(base_score, dtype=torch.bool),
        )

        self.assertTrue(torch.isfinite(gate_input).all())
        self.assertGreater(float(gate_input[0, 0]), 0.95)
        score_features = gate_input[0, adapter.adapter_dim :]
        self.assertAlmostEqual(float(score_features[0]), 0.3, places=6)
        self.assertAlmostEqual(float(score_features[1]), 0.3 / 3.0 + 0.4 / 3.0, places=6)

    def test_nonzero_gate_is_uniform_and_cannot_change_ordering(self):
        adapter = StageBGDINOScoreAdapter(hidden_dim=8, adapter_dim=7, gate_hidden_dim=6)
        with torch.no_grad():
            adapter.rank_output.weight.normal_(mean=0.0, std=0.4)
            adapter.rank_output.bias.fill_(0.1)
            adapter.confidence_gate[-1].bias.fill_(0.375)
        query_hs = torch.randn(2, 13, 8)
        base_score = torch.linspace(-1.3, 1.1, 26).reshape(2, 13)

        output = adapter(query_hs, base_score)
        confidence_delta = output["confidence_score"] - output["base_score"]

        self.assertGreater(float(output["rank_residual"].detach().std()), 0.0)
        self.assertTrue(
            torch.allclose(
                confidence_delta,
                output["confidence_gate"][:, None].expand_as(confidence_delta),
                atol=2e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(
            torch.equal(
                output["confidence_score"].argsort(dim=1),
                output["base_score"].argsort(dim=1),
            )
        )

    def test_listwise_loss_updates_only_rank_branch(self):
        adapter = StageBGDINOScoreAdapter(hidden_dim=8, adapter_dim=7, gate_hidden_dim=6)
        query_hs = torch.randn(2, 6, 8, requires_grad=True)
        base_score = torch.randn(2, 6, requires_grad=True)
        positive_mask = torch.tensor(
            [
                [True, True, False, False, False, False],
                [False, True, False, True, False, False],
            ]
        )

        rank_score = adapter(query_hs, base_score)["rank_score"]
        loss = multi_positive_listwise_rank_loss(rank_score, positive_mask)
        loss.backward()

        self.assertIsNone(query_hs.grad)
        self.assertIsNone(base_score.grad)
        self.assertGreater(float(adapter.rank_output.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in adapter.gate_parameters())
        )

    def test_rank_parameter_perturbation_cannot_change_confidence(self):
        adapter = StageBGDINOScoreAdapter(hidden_dim=8, adapter_dim=7, gate_hidden_dim=6)
        with torch.no_grad():
            adapter.confidence_gate[-1].weight.normal_(mean=0.0, std=0.2)
            adapter.confidence_gate[-1].bias.fill_(0.07)
        query_hs = torch.randn(2, 6, 8)
        base_score = torch.randn(2, 6)
        positive_mask = torch.tensor(
            [
                [True, False, False, False, False, False],
                [False, True, False, False, False, False],
            ]
        )
        before = adapter(query_hs, base_score)
        optimizer = torch.optim.SGD(adapter.rank_parameters(), lr=0.5)
        loss = multi_positive_listwise_rank_loss(
            before["rank_score"], positive_mask
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after = adapter(query_hs, base_score)

        self.assertTrue(
            torch.equal(
                before["confidence_gate"].detach(),
                after["confidence_gate"].detach(),
            )
        )
        self.assertTrue(
            torch.equal(
                before["confidence_score"].detach(),
                after["confidence_score"].detach(),
            )
        )
        self.assertFalse(
            torch.equal(before["rank_score"].detach(), after["rank_score"].detach())
        )

    def test_gate_parameter_perturbation_cannot_change_rank(self):
        adapter = StageBGDINOScoreAdapter(hidden_dim=8, adapter_dim=7, gate_hidden_dim=6)
        query_hs = torch.randn(2, 6, 8)
        base_score = torch.randn(2, 6)
        before = adapter(query_hs, base_score)
        with torch.no_grad():
            for parameter in adapter.gate_parameters():
                parameter.add_(torch.randn_like(parameter) * 0.3)
        after = adapter(query_hs, base_score)

        self.assertTrue(
            torch.equal(
                before["rank_score"].detach(), after["rank_score"].detach()
            )
        )
        self.assertFalse(
            torch.equal(
                before["confidence_score"].detach(),
                after["confidence_score"].detach(),
            )
        )

    def test_fpr_loss_updates_only_confidence_branch(self):
        adapter = StageBGDINOScoreAdapter(hidden_dim=8, adapter_dim=7, gate_hidden_dim=6)
        with torch.no_grad():
            adapter.confidence_gate[-1].weight.normal_(mean=0.0, std=0.2)
        positive_hs = torch.randn(20, 5, 8, requires_grad=True)
        negative_hs = torch.randn(7, 5, 8, requires_grad=True)
        positive_base = torch.rand(20, 5, requires_grad=True)
        negative_base = torch.rand(7, 5, requires_grad=True)

        positive = adapter(positive_hs, positive_base)["confidence_score"]
        negative = adapter(negative_hs, negative_base)["confidence_score"]
        loss = fpr95_global_max_surrogate(positive, negative).loss
        loss.backward()

        self.assertIsNone(positive_hs.grad)
        self.assertIsNone(negative_hs.grad)
        self.assertIsNone(positive_base.grad)
        self.assertIsNone(negative_base.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in adapter.rank_parameters())
        )
        self.assertGreater(
            float(adapter.confidence_gate[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in adapter.confidence_norm.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in adapter.confidence_trunk.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        # FPR@95TPR is translation invariant, so a shared scalar bias correctly
        # has zero gradient while feature-dependent gate weights still learn.
        self.assertIsNotNone(adapter.confidence_gate[-1].bias.grad)
        self.assertEqual(
            float(adapter.confidence_gate[-1].bias.grad.abs().sum()), 0.0
        )


class StageBGDINOScoreLossTest(unittest.TestCase):
    def test_two_rank_gather_keeps_only_local_slice_in_autograd(self):
        local = torch.tensor([0.2, 0.4], requires_grad=True)

        def fake_all_gather(outputs, source):
            if source.dtype == torch.int64:
                outputs[0].copy_(torch.tensor([2], dtype=torch.int64))
                outputs[1].copy_(torch.tensor([2], dtype=torch.int64))
            else:
                outputs[0].copy_(torch.tensor([0.2, 0.4]))
                outputs[1].copy_(torch.tensor([0.1, 0.3]))

        module = "models.GroundingDINO.stage_b_gdino_score_adapter.dist"
        with (
            patch(f"{module}.is_available", return_value=True),
            patch(f"{module}.is_initialized", return_value=True),
            patch(f"{module}.get_world_size", return_value=2),
            patch(f"{module}.get_rank", return_value=0),
            patch(f"{module}.all_gather", side_effect=fake_all_gather),
        ):
            gathered, world_size = distributed_gather_1d_with_local_grad(local)

        self.assertEqual(world_size, 2)
        self.assertTrue(
            torch.equal(gathered, torch.tensor([0.2, 0.4, 0.1, 0.3]))
        )
        (gathered * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
        self.assertTrue(torch.equal(local.grad, torch.tensor([1.0, 2.0])))

    def test_global_batch8_queue_forward_and_current_st_gradient(self):
        positive = torch.tensor(
            [[0.1], [0.4], [0.6], [0.8]], requires_grad=True
        )
        negative = torch.tensor(
            [[-0.5], [0.0], [0.2], [0.9]], requires_grad=True
        )
        remote_values = iter(
            (
                torch.tensor([0.2, 0.3, 0.5, 0.7]),
                torch.tensor([-0.2, 0.1, 0.3, 0.8]),
            )
        )

        def fake_gather(local):
            return torch.cat((local, next(remote_values))), 2

        with patch(
            "models.GroundingDINO.stage_b_gdino_score_adapter."
            "distributed_gather_1d_with_local_grad",
            side_effect=fake_gather,
        ):
            result = fpr95_global_max_surrogate(
                positive,
                negative,
                positive_history=torch.tensor([-0.4]),
                temperature=0.2,
            )
        result.loss.backward()

        # Global current N=8 uses the exact minimum at 95% TPR.  Queue history
        # controls the forward threshold, while ST still differentiates the
        # current global order statistic selected on this rank.
        self.assertAlmostEqual(
            float(result.positive_threshold.detach()), -0.4, places=6
        )
        self.assertAlmostEqual(
            float(result.surrogate_threshold.detach()), -0.4, places=6
        )
        self.assertEqual(result.positive_global_score.numel(), 8)
        self.assertLess(float(positive.grad[0, 0]), 0.0)
        self.assertEqual(float(positive.grad[1:].abs().sum()), 0.0)
        self.assertTrue(bool((negative.grad[:, 0] > 0).all().item()))
        expected_first_negative_grad = float(torch.sigmoid(torch.tensor(-0.5))) / 4.0
        self.assertAlmostEqual(
            float(negative.grad[0, 0]), expected_first_negative_grad, places=6
        )

    def test_baseline_correct_identity_has_exact_zero_rank_loss(self):
        base = torch.tensor([[0.8, 0.3, 0.4]])
        residual = torch.zeros_like(base, requires_grad=True)
        iou = torch.tensor([[0.7, 0.4, 0.2]])

        result = baseline_preserving_top1_rank_loss(
            base + residual,
            base,
            residual,
            iou,
            iou_threshold=0.5,
        )

        self.assertEqual(float(result.loss.detach()), 0.0)
        self.assertEqual(float(result.base_correct), 1.0)
        self.assertEqual(float(result.adapted_correct), 1.0)
        self.assertEqual(float(result.correct_regressed), 0.0)

    def test_rank_loss_treats_every_iou_below_acc50_as_negative(self):
        base = torch.tensor([[0.2, 0.8, 0.1]])
        residual = torch.zeros_like(base, requires_grad=True)
        # The hardest negative is IoU 0.4. It must not be ignored by a legacy
        # 0.3 ambiguity band because deployed acc50 counts it as wrong.
        iou = torch.tensor([[0.7, 0.4, 0.2]])

        result = baseline_preserving_top1_rank_loss(
            base + residual,
            base,
            residual,
            iou,
            iou_threshold=0.5,
        )
        result.loss.backward()

        self.assertGreater(float(result.loss.detach()), 0.0)
        self.assertLess(float(residual.grad[0, 0]), 0.0)
        self.assertGreater(float(residual.grad[0, 1]), 0.0)
        self.assertEqual(float(result.base_correct), 0.0)

        fixed = baseline_preserving_top1_rank_loss(
            torch.tensor([[0.9, 0.8, 0.1]]),
            base,
            torch.tensor([[0.7, 0.0, 0.0]]),
            iou,
            iou_threshold=0.5,
        )
        self.assertEqual(float(fixed.wrong_fixed), 1.0)

    def test_fix_rows_are_not_diluted_by_zero_loss_preserve_rows(self):
        wrong_base = torch.tensor([[0.2, 0.8]])
        wrong_iou = torch.tensor([[0.7, 0.2]])
        single_residual = torch.zeros_like(wrong_base, requires_grad=True)
        single = baseline_preserving_top1_rank_loss(
            wrong_base + single_residual,
            wrong_base,
            single_residual,
            wrong_iou,
            residual_weight=0.0,
        )
        single.loss.backward()

        correct_base = torch.tensor([[0.8, 0.2]]).expand(7, 2)
        padded_base = torch.cat((wrong_base, correct_base), dim=0)
        padded_iou = wrong_iou.expand(8, 2)
        padded_residual = torch.zeros_like(padded_base, requires_grad=True)
        padded = baseline_preserving_top1_rank_loss(
            padded_base + padded_residual,
            padded_base,
            padded_residual,
            padded_iou,
            residual_weight=0.0,
        )
        padded.loss.backward()

        self.assertEqual(float(padded.fix_rows), 1.0)
        self.assertEqual(float(padded.preserve_rows), 7.0)
        self.assertEqual(float(padded.preserve_loss.detach()), 0.0)
        self.assertTrue(
            torch.allclose(padded.fix_loss, single.fix_loss, atol=1e-7, rtol=0.0)
        )
        self.assertTrue(
            torch.allclose(padded.margin_loss, single.margin_loss, atol=1e-7, rtol=0.0)
        )
        self.assertTrue(
            torch.allclose(
                padded_residual.grad[0],
                single_residual.grad[0],
                atol=1e-7,
                rtol=0.0,
            )
        )

    def test_preserve_margin_is_allowed_base_gap_shrinkage(self):
        base = torch.tensor([[0.9, 0.1]])
        iou = torch.tensor([[0.8, 0.2]])

        within_slack = torch.tensor([[-0.08, 0.0]], requires_grad=True)
        allowed = baseline_preserving_top1_rank_loss(
            base + within_slack,
            base,
            within_slack,
            iou,
            preserve_margin=0.1,
            residual_weight=0.0,
        )
        self.assertEqual(float(allowed.preserve_loss.detach()), 0.0)

        beyond_slack = torch.tensor([[-0.14, 0.0]], requires_grad=True)
        penalized = baseline_preserving_top1_rank_loss(
            base + beyond_slack,
            base,
            beyond_slack,
            iou,
            preserve_margin=0.1,
            residual_weight=0.0,
        )
        penalized.loss.backward()

        self.assertEqual(float(penalized.fix_rows), 0.0)
        self.assertEqual(float(penalized.preserve_rows), 1.0)
        self.assertAlmostEqual(float(penalized.preserve_loss.detach()), 0.04, places=6)
        self.assertLess(float(beyond_slack.grad[0, 0]), 0.0)
        self.assertGreater(float(beyond_slack.grad[0, 1]), 0.0)

    def test_listwise_gradient_moves_positives_up_and_negatives_down(self):
        score = torch.tensor([[0.1, -0.2, 1.2, -0.3, 9.0]], requires_grad=True)
        positive = torch.tensor([[True, True, False, False, False]])
        eligible = torch.tensor([[True, True, True, True, False]])

        loss = multi_positive_listwise_rank_loss(
            score, positive, eligible_mask=eligible, temperature=0.2
        )
        loss.backward()

        self.assertLess(float(score.grad[0, 0]), 0.0)
        self.assertLess(float(score.grad[0, 1]), 0.0)
        self.assertGreater(float(score.grad[0, 2]), 0.0)
        self.assertGreater(float(score.grad[0, 3]), 0.0)
        self.assertEqual(float(score.grad[0, 4]), 0.0)

    def test_missing_listwise_class_returns_graph_connected_zero(self):
        score = torch.tensor([[0.1, 0.2], [0.4, 0.3]], requires_grad=True)
        positive = torch.tensor([[True, True], [False, False]])
        eligible = torch.tensor([[True, True], [False, False]])

        loss = multi_positive_listwise_rank_loss(
            score, positive, eligible_mask=eligible
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(score.grad)
        self.assertEqual(float(score.grad.abs().sum()), 0.0)

    def test_listwise_masks_are_exact_and_positive_is_eligible(self):
        score = torch.zeros(1, 3)
        positive = torch.tensor([[True, False, False]])
        eligible = torch.tensor([[True, True, False]])
        with self.assertRaisesRegex(ValueError, "exact boolean"):
            multi_positive_listwise_rank_loss(
                score, positive.to(torch.int64), eligible_mask=eligible
            )
        with self.assertRaisesRegex(ValueError, "exact boolean"):
            multi_positive_listwise_rank_loss(
                score, positive, eligible_mask=eligible.to(torch.int64)
            )
        with self.assertRaisesRegex(ValueError, "subset"):
            multi_positive_listwise_rank_loss(
                score,
                positive,
                eligible_mask=torch.tensor([[False, True, True]]),
            )

    def test_fpr95_uses_exact_order_statistic_and_every_negative_max(self):
        # With N=20 and TPR=0.95, the exact >= threshold is the second-smallest
        # positive score, matching the final evaluation gate.
        positive = torch.arange(20, dtype=torch.float32).view(20, 1).requires_grad_()
        negative = torch.tensor(
            [[0.0, -10.0], [1.0, -10.0], [2.0, -10.0]],
            requires_grad=True,
        )

        result = fpr95_global_max_surrogate(
            positive, negative, temperature=0.2
        )
        result.loss.backward()

        self.assertEqual(float(result.positive_threshold.detach()), 1.0)
        self.assertAlmostEqual(float(result.exact_tpr), 0.95, places=6)
        self.assertAlmostEqual(float(result.exact_fpr), 2.0 / 3.0, places=6)
        # Every TN contributes, including the one below the operating threshold;
        # only each row's deployed maximum receives its gradient.
        self.assertTrue(bool((negative.grad[:, 0] > 0).all().item()))
        self.assertEqual(float(negative.grad[:, 1].abs().sum()), 0.0)
        self.assertLess(float(positive.grad[1, 0]), 0.0)
        self.assertEqual(float(positive.grad[[0] + list(range(2, 20))].abs().sum()), 0.0)

    def test_exact_threshold_matches_final_gate_tie_policy(self):
        score = torch.tensor([0.0] * 3 + [1.0] * 17)
        threshold = exact_tpr_operating_threshold(score, target_tpr=0.95)
        actual_tpr = (score >= threshold).float().mean()

        self.assertEqual(float(threshold), 0.0)
        self.assertEqual(float(actual_tpr), 1.0)

    def test_fpr_score_histories_are_detached_queue_inputs(self):
        positive = torch.tensor([[0.8], [0.6]], requires_grad=True)
        negative = torch.tensor([[0.5], [0.2]], requires_grad=True)
        positive_history = torch.tensor([0.1, 0.7], requires_grad=True)
        negative_history = torch.tensor([0.9, -0.4], requires_grad=True)

        result = fpr95_global_max_surrogate(
            positive,
            negative,
            positive_history=positive_history,
            negative_history=negative_history,
            temperature=0.2,
        )
        result.loss.backward()

        self.assertIsNone(positive_history.grad)
        self.assertIsNone(negative_history.grad)
        self.assertIsNotNone(positive.grad)
        self.assertIsNotNone(negative.grad)
        self.assertGreater(float(positive.grad.abs().sum()), 0.0)

    def test_negative_history_does_not_dilute_current_tn_gradients(self):
        positive_a = torch.tensor([[0.8], [0.6]], requires_grad=True)
        negative_a = torch.tensor([[0.5], [0.2]], requires_grad=True)
        first = fpr95_global_max_surrogate(
            positive_a,
            negative_a,
            positive_history=torch.tensor([0.1, 0.7]),
            temperature=0.2,
        )
        first.loss.backward()
        gradient_without_history = negative_a.grad.detach().clone()

        positive_b = positive_a.detach().clone().requires_grad_()
        negative_b = negative_a.detach().clone().requires_grad_()
        second = fpr95_global_max_surrogate(
            positive_b,
            negative_b,
            positive_history=torch.tensor([0.1, 0.7]),
            negative_history=torch.linspace(-3.0, 3.0, 1000),
            temperature=0.2,
        )
        second.loss.backward()

        self.assertTrue(torch.equal(negative_b.grad, gradient_without_history))
        self.assertEqual(float(second.exact_fpr), float(first.exact_fpr))

    def test_queue_threshold_keeps_current_positive_gradient_and_translation_cancels(self):
        positive = torch.tensor([[0.8], [0.6]], requires_grad=True)
        negative = torch.tensor([[0.5], [0.2]], requires_grad=True)
        result = fpr95_global_max_surrogate(
            positive,
            negative,
            positive_history=torch.tensor([-2.0, -1.0, 0.1, 0.7]),
            temperature=0.2,
            paired_margin_weight=0.5,
            paired_margin=0.05,
        )
        result.loss.backward()

        self.assertEqual(float(result.positive_threshold.detach()), -2.0)
        self.assertGreater(float(positive.grad.abs().sum()), 0.0)
        self.assertGreater(float(negative.grad.abs().sum()), 0.0)
        # FPR and paired-margin objectives both depend only on score gaps.  A
        # common translation of all positive/TN confidence scores has no effect.
        total_translation_gradient = positive.grad.sum() + negative.grad.sum()
        self.assertAlmostEqual(float(total_translation_gradient), 0.0, places=6)

    def test_detached_recent_q05_uses_history_only_and_trust_boundary(self):
        positive = torch.tensor([[-4.0], [-3.0], [-2.0]], requires_grad=True)
        negative = torch.tensor([[0.6], [0.1], [-0.3]], requires_grad=True)
        positive_gate = torch.tensor(
            [-0.01, -0.02, -0.03], requires_grad=True
        )
        negative_gate = torch.tensor([0.2, 0.1, -0.1], requires_grad=True)
        history = torch.tensor([0.5, 0.7], requires_grad=True)

        result = detached_recent_q05_trust_surrogate(
            positive,
            negative,
            positive_gate,
            negative_gate,
            positive_history=history,
            temperature=0.2,
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            paired_margin_weight=0.0,
        )
        result.loss.backward()

        # A warm queue alone defines the bank threshold.  Current positives are
        # deliberately far lower and cannot enter it or receive q05 ST gradient.
        self.assertEqual(float(result.positive_threshold), 0.5)
        self.assertEqual(float(result.current_positive_threshold), -4.0)
        self.assertIsNone(history.grad)
        self.assertIsNone(positive.grad)
        self.assertTrue(bool((negative.grad[:, 0] > 0).all().item()))
        self.assertIsNone(negative_gate.grad)
        self.assertLess(float(positive_gate.grad[0]), 0.0)
        self.assertAlmostEqual(
            float(positive_gate.grad[0]), float(positive_gate.grad[1]), places=7
        )
        self.assertAlmostEqual(
            float(positive_gate.grad[2] - positive_gate.grad[1]),
            -1.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(result.positive_trust_loss.detach()), 0.01 / 3.0, places=8
        )
        self.assertAlmostEqual(
            float(result.positive_trust_violation_rate), 1.0 / 3.0, places=6
        )

    def test_detached_recent_q05_pre_warm_threshold_is_detached(self):
        positive = torch.tensor([[0.2], [0.4]], requires_grad=True)
        negative = torch.tensor([[0.3], [0.5]], requires_grad=True)
        positive_gate = torch.tensor([0.0, 0.0], requires_grad=True)
        negative_gate = torch.tensor([0.0, 0.0], requires_grad=True)

        result = detached_recent_q05_trust_surrogate(
            positive,
            negative,
            positive_gate,
            negative_gate,
            positive_history=None,
            paired_margin_weight=0.0,
        )
        result.loss.backward()

        self.assertAlmostEqual(float(result.positive_threshold), 0.2, places=6)
        self.assertAlmostEqual(
            float(result.current_positive_threshold), 0.2, places=6
        )
        self.assertIsNone(positive.grad)
        self.assertTrue(bool((negative.grad > 0).all().item()))
        self.assertTrue(bool((positive_gate.grad < 0).all().item()))
        self.assertAlmostEqual(
            float(positive_gate.grad[0]), float(positive_gate.grad[1]), places=7
        )

    def test_detached_recent_q05_cancels_common_translation_gradient(self):
        common_bias = torch.tensor(0.0, requires_grad=True)
        positive_base = torch.tensor([[0.4], [0.5], [0.6], [0.7]])
        negative_base = torch.tensor([[0.2], [0.3], [0.55], [0.8]])

        result = detached_recent_q05_trust_surrogate(
            positive_base + common_bias,
            negative_base + common_bias,
            common_bias.expand(4),
            common_bias.expand(4),
            positive_history=torch.tensor([0.35, 0.45, 0.55, 0.65]),
            temperature=0.1,
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            paired_margin_weight=0.25,
            paired_margin=0.05,
        )
        result.loss.backward()

        self.assertAlmostEqual(float(common_bias.grad), 0.0, places=6)

    def test_total_trust_binds_positive_protection_to_deployed_score(self):
        positive_base = torch.tensor(
            [[-4.0], [-3.0], [-2.0]], requires_grad=True
        )
        negative_base = torch.tensor(
            [[0.6], [0.1], [-0.3]], requires_grad=True
        )
        positive_gate = torch.tensor(
            [-0.01, -0.02, -0.03], requires_grad=True
        )
        negative_gate = torch.tensor([0.2, 0.1, -0.1], requires_grad=True)
        positive = positive_base.detach() + positive_gate[:, None]
        negative = negative_base.detach() + negative_gate[:, None]
        result = detached_recent_q05_trust_surrogate(
            positive,
            negative,
            positive_gate,
            negative_gate,
            positive_history=torch.tensor([0.5, 0.7]),
            temperature=0.2,
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            paired_margin_weight=0.0,
            positive_score_trust=True,
        )
        result.loss.backward()

        # The deployed max is the only differentiable positive protection
        # source.  The frozen base receives no gradient, while the confidence
        # owner gets a non-zero positive-tail signal through the score itself.
        self.assertIsNone(positive_base.grad)
        self.assertTrue(bool((positive_gate.grad < 0).all().item()))
        self.assertGreater(float(result.positive_score_trust_loss.detach()), 0.0)
        self.assertEqual(
            float(result.positive_trust_loss.detach()),
            float(result.positive_score_trust_loss.detach()),
        )
        self.assertEqual(float(result.positive_trust_violation_rate), 0.0)
        self.assertEqual(float(result.positive_score_trust_violation_rate), 1.0)

    def test_total_trust_directly_differentiates_deployed_score_leaf(self):
        positive_score = torch.tensor(
            [[-1.0], [-1.5], [-2.0]], requires_grad=True
        )
        negative_score = torch.tensor(
            [[0.6], [0.1], [-0.3]], requires_grad=True
        )
        positive_gate = torch.tensor(
            [-0.01, -0.02, -0.03], requires_grad=True
        )
        negative_gate = torch.tensor([0.2, 0.1, -0.1], requires_grad=True)
        result = detached_recent_q05_trust_surrogate(
            positive_score,
            negative_score,
            positive_gate,
            negative_gate,
            positive_history=torch.tensor([0.5, 0.7]),
            temperature=0.2,
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            paired_margin_weight=0.0,
            positive_score_trust=True,
        )
        result.loss.backward()

        self.assertGreater(float(positive_score.grad.abs().sum()), 0.0)
        self.assertIsNone(positive_gate.grad)

    def test_total_trust_common_translation_remains_zero(self):
        common_bias = torch.tensor(0.0, requires_grad=True)
        positive_base = torch.tensor([[0.4], [0.5], [0.6], [0.7]])
        negative_base = torch.tensor([[0.2], [0.3], [0.55], [0.8]])
        result = detached_recent_q05_trust_surrogate(
            positive_base + common_bias,
            negative_base + common_bias,
            common_bias.expand(4),
            common_bias.expand(4),
            positive_history=torch.tensor([0.35, 0.45, 0.55, 0.65]),
            temperature=0.1,
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            paired_margin_weight=0.25,
            paired_margin=0.05,
            positive_score_trust=True,
        )
        result.loss.backward()
        self.assertAlmostEqual(float(common_bias.grad), 0.0, places=6)

    def test_two_rank_simulated_p3_gradient_matches_global_batch(self):
        positive_base = torch.tensor([0.30, 0.45, 0.25, 0.55])
        negative_base = torch.tensor([0.50, 0.10, 0.35, 0.20])
        positive_feature = torch.tensor([1.0, -0.5, 0.7, -1.2])
        negative_feature = torch.tensor([-0.4, 1.1, -0.8, 0.3])
        initial_weight = -0.08
        history = torch.tensor([0.22, 0.28, 0.34, 0.41, 0.50])

        reference_weight = torch.tensor(initial_weight, requires_grad=True)
        reference_positive_gate = positive_feature * reference_weight
        reference_negative_gate = negative_feature * reference_weight
        reference = detached_recent_q05_trust_surrogate(
            (positive_base + reference_positive_gate)[:, None],
            (negative_base + reference_negative_gate)[:, None],
            reference_positive_gate,
            reference_negative_gate,
            positive_history=history,
            paired_margin_weight=0.25,
            paired_margin=0.05,
        )
        reference.loss.backward()

        rank_gradients = []
        for rank, local_slice in enumerate((slice(0, 2), slice(2, 4))):
            remote_slice = slice(2, 4) if rank == 0 else slice(0, 2)
            weight = torch.tensor(initial_weight, requires_grad=True)
            local_positive_gate = positive_feature[local_slice] * weight
            local_negative_gate = negative_feature[local_slice] * weight
            remote_positive_gate = (
                positive_feature[remote_slice] * weight.detach()
            )
            remote_negative_gate = (
                negative_feature[remote_slice] * weight.detach()
            )
            local_values = (
                (positive_base[local_slice] + local_positive_gate)[:, None],
                (negative_base[local_slice] + local_negative_gate)[:, None],
                local_positive_gate,
                local_negative_gate,
            )
            remote_values = iter(
                (
                    positive_base[remote_slice] + remote_positive_gate,
                    negative_base[remote_slice] + remote_negative_gate,
                    remote_positive_gate,
                    remote_negative_gate,
                )
            )

            def fake_gather(local):
                remote = next(remote_values)
                parts = (local, remote) if rank == 0 else (remote, local)
                return torch.cat(parts), 2

            with patch(
                "models.GroundingDINO.stage_b_gdino_score_adapter."
                "distributed_gather_1d_with_local_grad",
                side_effect=fake_gather,
            ):
                result = detached_recent_q05_trust_surrogate(
                    *local_values,
                    positive_history=history,
                    paired_margin_weight=0.25,
                    paired_margin=0.05,
                )
            result.loss.backward()
            rank_gradients.append(weight.grad.detach())

        simulated_ddp_gradient = torch.stack(rank_gradients).mean()
        self.assertAlmostEqual(
            float(simulated_ddp_gradient),
            float(reference_weight.grad),
            places=6,
        )

    def test_two_rank_gloo_ddp_p3_gradient_matches_global_batch(self):
        positive_base = torch.tensor([0.30, 0.45, 0.25, 0.55])
        negative_base = torch.tensor([0.50, 0.10, 0.35, 0.20])
        positive_feature = torch.tensor([1.0, -0.5, 0.7, -1.2])
        negative_feature = torch.tensor([-0.4, 1.1, -0.8, 0.3])
        history = torch.tensor([0.22, 0.28, 0.34, 0.41, 0.50])
        model = _P3ToyGate(-0.08)
        gates = model(torch.stack((positive_feature, negative_feature)))
        reference = detached_recent_q05_trust_surrogate(
            (positive_base + gates[0])[:, None],
            (negative_base + gates[1])[:, None],
            gates[0],
            gates[1],
            positive_history=history,
            paired_margin_weight=0.25,
            paired_margin=0.05,
        )
        reference.loss.backward()

        with tempfile.TemporaryDirectory() as temporary_directory:
            mp.spawn(
                _p3_ddp_gradient_worker,
                args=(
                    2,
                    f"{temporary_directory}/gloo_init",
                    float(model.weight.grad),
                    float(reference.loss.detach()),
                ),
                nprocs=2,
                join=True,
            )

    def test_two_rank_gloo_rank_gradient_matches_global_class_normalization(self):
        base, features, iou = _rank_ddp_inputs()
        model = _RankToyResidual(0.1)
        residual = model(features)
        reference = baseline_preserving_top1_rank_loss(
            base + residual,
            base,
            residual,
            iou,
            preserve_margin=0.1,
            residual_weight=0.0,
        )
        reference.loss.backward()

        self.assertEqual(float(reference.fix_rows), 4.0)
        self.assertEqual(float(reference.preserve_rows), 4.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            mp.spawn(
                _rank_ddp_gradient_worker,
                args=(
                    2,
                    f"{temporary_directory}/rank_gloo_init",
                    float(model.weight.grad),
                ),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
