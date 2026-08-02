import unittest
import types

import torch
from torch import nn

from engine import (
    _build_stage_b_legacy_gate_pair_captions,
    _set_stage_b_legacy_global_gate_training_mode,
    _split_stage_b_legacy_gate_batch,
)

from models.GroundingDINO.stage_b_legacy_global_gate import (
    LegacyStageBGlobalGate,
    LegacyStageBGlobalGateCriterion,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_stageb_tn_val as tn_eval


class LegacyStageBGlobalGateTest(unittest.TestCase):
    def test_zero_init_is_bitwise_identity_and_preserves_ranking(self):
        torch.manual_seed(4)
        gate = LegacyStageBGlobalGate(hidden_dim=16, gate_hidden_dim=8)
        query_hs = torch.randn(2, 11, 16)
        legacy_score = torch.randn(2, 11, 3)

        output = gate(query_hs, legacy_score)

        self.assertTrue(torch.equal(output["gate_bias"], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(output["confidence"], legacy_score))
        self.assertTrue(
            torch.equal(
                output["confidence"].argmax(dim=1),
                legacy_score.argmax(dim=1),
            )
        )

    def test_one_bias_is_broadcast_to_every_query(self):
        torch.manual_seed(5)
        gate = LegacyStageBGlobalGate(hidden_dim=8, gate_hidden_dim=4)
        with torch.no_grad():
            gate.gate[-1].bias.fill_(0.375)
        query_hs = torch.randn(2, 7, 8)
        legacy_score = torch.randn(2, 7, 2)

        output = gate(query_hs, legacy_score)
        expected = output["gate_bias"][:, None, :].expand_as(legacy_score)
        observed = output["confidence"] - legacy_score

        self.assertTrue(torch.allclose(observed, expected, atol=2e-7, rtol=0.0))
        self.assertTrue(
            torch.equal(
                output["confidence"].argmax(dim=1),
                legacy_score.argmax(dim=1),
            )
        )

    def test_confidence_gradient_cannot_reach_frozen_inputs(self):
        torch.manual_seed(6)
        gate = LegacyStageBGlobalGate(hidden_dim=8, gate_hidden_dim=4)
        query_hs = torch.randn(2, 7, 8, requires_grad=True)
        legacy_score = torch.randn(2, 7, 2, requires_grad=True)

        gate(query_hs, legacy_score)["confidence"].sum().backward()

        self.assertIsNone(query_hs.grad)
        self.assertIsNone(legacy_score.grad)
        self.assertGreater(float(gate.gate[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(gate.gate[-1].bias.grad.abs().sum()), 0.0)

    def test_training_mode_keeps_frozen_base_deterministic(self):
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.9))
                self.stage_b_legacy_global_gate = LegacyStageBGlobalGate(4, 4)

        model = DummyModel()
        model.train()
        _set_stage_b_legacy_global_gate_training_mode(model)

        self.assertFalse(model.training)
        self.assertFalse(model.base.training)
        self.assertTrue(model.stage_b_legacy_global_gate.training)
        first = model.base(torch.ones(2, 4))
        second = model.base(torch.ones(2, 4))
        self.assertTrue(torch.equal(first, second))

    def test_pair_extraction_is_fail_closed_and_keeps_expressions_separate(self):
        target = {
            "cap_list": ["red car", "blue car"],
            "is_tn": torch.tensor([False, True]),
            "verifier_pair_stride": torch.tensor([2]),
            "verifier_num_patch_slots": torch.tensor([1]),
            "proposalset_proxy_verified": torch.tensor([True]),
        }
        positive, negative = _build_stage_b_legacy_gate_pair_captions([target])
        self.assertEqual(positive, ["red car ."])
        self.assertEqual(negative, ["blue car ."])

        target["proposalset_proxy_verified"] = torch.tensor([False])
        with self.assertRaisesRegex(RuntimeError, "proposalset_proxy_verified=True"):
            _build_stage_b_legacy_gate_pair_captions([target])
        target["proposalset_proxy_verified"] = "false"
        with self.assertRaisesRegex(RuntimeError, "proposalset_proxy_verified=True"):
            _build_stage_b_legacy_gate_pair_captions([target])

    def test_paired_output_split_preserves_positive_and_tn_halves(self):
        paired = {
            "score": torch.arange(12).view(4, 3),
            "aux": [{"box": torch.arange(16).view(4, 4)}],
        }
        positive = _split_stage_b_legacy_gate_batch(paired, 2, False)
        negative = _split_stage_b_legacy_gate_batch(paired, 2, True)
        self.assertTrue(torch.equal(positive["score"], paired["score"][:2]))
        self.assertTrue(torch.equal(negative["score"], paired["score"][2:]))
        self.assertTrue(torch.equal(positive["aux"][0]["box"], paired["aux"][0]["box"][:2]))
        self.assertTrue(torch.equal(negative["aux"][0]["box"], paired["aux"][0]["box"][2:]))

    def test_ref_and_tn_evaluators_consume_gated_confidence(self):
        cfg = types.SimpleNamespace(
            stage_b_v11_fixed_text=False,
            stage_b_v7=False,
        )
        confidence = torch.tensor([[[0.2], [0.8], [0.1]]])
        outputs = {"stage_b_legacy_global_confidence": confidence}

        self.assertTrue(torch.equal(ref_eval._slot_scores(outputs, cfg, 1.0), confidence))
        self.assertTrue(torch.equal(tn_eval._slot_scores(outputs, cfg, 1.0), confidence))


class LegacyStageBGlobalGateCriterionTest(unittest.TestCase):
    @staticmethod
    def _outputs(score: torch.Tensor):
        return {"stage_b_legacy_global_confidence": score}

    def test_losses_use_deployed_global_max_and_backpropagate(self):
        positive = torch.tensor(
            [[[0.1], [0.8], [0.2]], [[0.7], [0.3], [0.4]]],
            requires_grad=True,
        )
        negative = torch.tensor(
            [[[0.6], [0.2], [0.1]], [[0.5], [0.9], [0.4]]],
            requires_grad=True,
        )
        outputs = self._outputs(positive)
        outputs["stage_b_legacy_global_tn_outputs"] = self._outputs(negative)
        criterion = LegacyStageBGlobalGateCriterion(tail_fraction=0.5)
        targets = [
            {"proposalset_proxy_verified": torch.tensor([True])},
            {"proposalset_proxy_verified": torch.tensor([True])},
        ]

        losses = criterion(outputs, targets)
        total = sum(losses[key] for key in criterion.weight_dict)
        total.backward()

        self.assertAlmostEqual(
            float(losses["legacy_gate_positive_global_max"]), 0.75, places=6
        )
        self.assertAlmostEqual(
            float(losses["legacy_gate_tn_global_max"]), 0.75, places=6
        )
        self.assertGreater(float(positive.grad.abs().sum()), 0.0)
        self.assertGreater(float(negative.grad.abs().sum()), 0.0)
        self.assertEqual(float(positive.grad[0, [0, 2]].abs().sum()), 0.0)
        self.assertEqual(float(positive.grad[1, [1, 2]].abs().sum()), 0.0)
        self.assertEqual(float(negative.grad[0, [1, 2]].abs().sum()), 0.0)
        self.assertEqual(float(negative.grad[1, [0, 2]].abs().sum()), 0.0)

    def test_fpr95_uses_exact_order_statistic_and_every_negative_row(self):
        # For N=20, ceil(.95*N)=19 and the evaluator's >= threshold is the
        # second-smallest positive score.
        positive = torch.arange(20, dtype=torch.float32).view(20, 1, 1)
        positive.requires_grad_()
        negative_max = torch.tensor([0.0, 1.0, 2.0] + [0.0] * 17)
        negative = torch.stack(
            (negative_max, torch.full_like(negative_max, -10.0)), dim=1
        ).unsqueeze(-1)
        negative.requires_grad_()
        outputs = self._outputs(positive)
        outputs["stage_b_legacy_global_tn_outputs"] = self._outputs(negative)
        criterion = LegacyStageBGlobalGateCriterion(
            tail_objective="fpr95",
            tail_margin=0.0,
            loss_temperature=0.2,
            require_proposalset_proxy_verified=False,
        )

        losses = criterion(outputs)
        losses["loss_legacy_gate_tail"].backward()

        self.assertEqual(float(losses["legacy_gate_fpr95_threshold"]), 1.0)
        self.assertAlmostEqual(
            float(losses["legacy_gate_batch_exact_tpr"]), 0.95, places=6
        )
        self.assertAlmostEqual(
            float(losses["legacy_gate_batch_exact_fpr95"]), 2.0 / 20.0, places=6
        )
        # All TN rows contribute to the smooth FPR denominator; only each
        # row's deployed maximum is differentiable through max reduction.
        self.assertTrue(bool((negative.grad[:, 0, 0] > 0).all().item()))
        self.assertEqual(float(negative.grad[:, 1, 0].abs().sum()), 0.0)
        self.assertLess(float(positive.grad[1, 0, 0]), 0.0)
        other_positive = [0] + list(range(2, 20))
        self.assertEqual(float(positive.grad[other_positive].abs().sum()), 0.0)

    def test_fpr95_threshold_uses_greater_equal_tie_policy(self):
        positive = torch.tensor([0.0] * 3 + [1.0] * 17).view(20, 1, 1)
        negative = torch.zeros(20, 1, 1)
        outputs = self._outputs(positive)
        outputs["stage_b_legacy_global_tn_outputs"] = self._outputs(negative)
        criterion = LegacyStageBGlobalGateCriterion(
            tail_objective="fpr95",
            require_proposalset_proxy_verified=False,
        )

        losses = criterion(outputs)

        self.assertEqual(float(losses["legacy_gate_fpr95_threshold"]), 0.0)
        self.assertEqual(float(losses["legacy_gate_batch_exact_tpr"]), 1.0)
        self.assertEqual(float(losses["legacy_gate_batch_exact_fpr95"]), 1.0)

    def test_cvar_remains_default_and_invalid_objective_is_rejected(self):
        criterion = LegacyStageBGlobalGateCriterion()
        self.assertEqual(criterion.tail_objective, "cvar")
        with self.assertRaisesRegex(ValueError, "tail_objective"):
            LegacyStageBGlobalGateCriterion(tail_objective="negative_q95")

    def test_unverified_tn_fails_closed(self):
        outputs = self._outputs(torch.zeros(1, 2, 1))
        outputs["stage_b_legacy_global_tn_outputs"] = self._outputs(
            torch.zeros(1, 2, 1)
        )
        criterion = LegacyStageBGlobalGateCriterion(
            require_proposalset_proxy_verified=True
        )

        with self.assertRaisesRegex(RuntimeError, "non-proxy TN pair"):
            criterion(
                outputs,
                [{"proposalset_proxy_verified": torch.tensor([False])}],
            )

        with self.assertRaisesRegex(RuntimeError, "non-proxy TN pair"):
            criterion(outputs, [{"proposalset_proxy_verified": "false"}])


if __name__ == "__main__":
    unittest.main()
