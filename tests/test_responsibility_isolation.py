import unittest

import torch

from tools.responsibility_isolation import (
    RESPONSIBILITY_OWNERSHIP_ISOLATED,
    RESPONSIBILITY_OWNERSHIP_SHARED,
    FrozenCandidateResponsibilityHeads,
    assert_isolated_responsibility_gradients,
    responsibility_gradient_report,
    responsibility_ownership_report,
)


def _inputs(feature_dim=256):
    generator = torch.Generator().manual_seed(20260821)
    features = torch.randn(2, 4, feature_dim, generator=generator)
    native = torch.tensor(
        [[0.31, 0.27, 0.04, -0.10], [0.14, 0.03, 0.09, 0.20]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True]],
        dtype=torch.bool,
    )
    return features, native, mask


class FrozenCandidateResponsibilityHeadsTest(unittest.TestCase):
    def test_zero_init_shapes_finiteness_mask_and_input_detach(self):
        module = FrozenCandidateResponsibilityHeads(
            ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED
        )
        features, native, mask = _inputs()
        # A compact fp16 candidate cache is accepted by fp32 heads without
        # restoring an autograd path to the frozen source model.
        features = features.half()
        features.requires_grad_(True)
        native.requires_grad_(True)
        output = module(features, native, mask)

        self.assertEqual(tuple(output["rank_score"].shape), (2, 4))
        self.assertEqual(tuple(output["confidence_score"].shape), (2, 4))
        self.assertTrue(torch.equal(output["candidate_mask"], mask))
        self.assertTrue(
            torch.equal(output["rank_residual"], torch.zeros_like(native))
        )
        self.assertTrue(torch.equal(output["rank_score"][mask], native.detach()[mask]))
        self.assertTrue(
            torch.equal(
                output["confidence_score"][mask],
                torch.zeros_like(native.detach()[mask]),
            )
        )
        self.assertTrue(torch.isfinite(output["rank_score"]).all())
        self.assertTrue(torch.isfinite(output["confidence_score"]).all())
        for row, row_mask in zip(output["rank_score"], mask):
            if bool((~row_mask).any()):
                self.assertLess(
                    float(row[~row_mask].detach().max()),
                    float(row[row_mask].detach().min()),
                )

        input_gradients = torch.autograd.grad(
            output["rank_score"][mask].sum()
            + output["confidence_score"][mask].sum(),
            (features, native),
            allow_unused=True,
        )
        self.assertEqual(input_gradients, (None, None))

    def test_input_contract_fails_closed(self):
        module = FrozenCandidateResponsibilityHeads(feature_dim=4, hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            module(
                torch.randn(1, 3, 4),
                torch.randn(1, 3),
                torch.zeros(1, 3, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            module(
                torch.full((1, 3, 4), torch.nan),
                torch.randn(1, 3),
                torch.ones(1, 3, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "boolean"):
            module(torch.randn(1, 3, 4), torch.randn(1, 3), torch.ones(1, 3))

    def test_shared_layout_has_a_measurable_conflict_path(self):
        torch.manual_seed(7)
        module = FrozenCandidateResponsibilityHeads(
            feature_dim=8,
            hidden_dim=12,
            ownership=RESPONSIBILITY_OWNERSHIP_SHARED,
            rank_residual_limit=1.0,
        )
        with torch.no_grad():
            module.rank_head.weight.fill_(0.2)
            module.confidence_head.weight.copy_(module.rank_head.weight)
        features, native, mask = _inputs(feature_dim=8)
        output = module(features, native, mask)
        rank_loss = output["rank_score"][mask].sum()
        confidence_loss = -output["confidence_score"][mask].sum()
        report = responsibility_gradient_report(
            module, rank_loss, confidence_loss
        )

        self.assertFalse(report["structurally_isolated"])
        self.assertTrue(report["gradient_finite"])
        self.assertGreater(report["shared_tensor_count"], 0)
        self.assertTrue(
            all(
                name.startswith(("rank_trunk.", "confidence_trunk."))
                for name in report["jointly_connected_parameter_names"]
            )
        )
        self.assertEqual(
            report["jointly_connected_parameter_names"],
            report["shared_parameter_names"],
        )
        self.assertTrue(report["joint_gradient_cosine_defined"])
        self.assertLess(report["joint_gradient_cosine"], 0.0)

    def test_isolated_layout_has_no_bidirectional_cross_task_gradient(self):
        module = FrozenCandidateResponsibilityHeads(
            feature_dim=8,
            hidden_dim=12,
            ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED,
        )
        features, native, mask = _inputs(feature_dim=8)
        output = module(features, native, mask)
        rank_loss = output["rank_score"][mask].square().mean()
        confidence_loss = output["confidence_score"][mask].square().mean()
        report = assert_isolated_responsibility_gradients(
            module, rank_loss, confidence_loss
        )

        self.assertTrue(report["structurally_isolated"])
        self.assertTrue(report["gradient_finite"])
        self.assertEqual(report["shared_parameter_names"], ())
        self.assertEqual(report["jointly_connected_parameter_names"], ())
        self.assertEqual(
            report["rank_loss_to_confidence_only_parameter_names"], ()
        )
        self.assertEqual(
            report["confidence_loss_to_rank_only_parameter_names"], ()
        )

    def test_confidence_owned_mutation_cannot_change_isolated_rank_output(self):
        module = FrozenCandidateResponsibilityHeads(
            feature_dim=8,
            hidden_dim=12,
            ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED,
        ).eval()
        features, native, mask = _inputs(feature_dim=8)
        before = module(features, native, mask)
        rank_before = before["rank_score"].detach().clone()
        confidence_before = before["confidence_score"].detach().clone()

        with torch.no_grad():
            for index, (_, parameter) in enumerate(
                module.named_task_parameters("confidence"), start=1
            ):
                parameter.add_(0.01 * index)
        after = module(features, native, mask)
        self.assertTrue(torch.equal(after["rank_score"], rank_before))
        self.assertFalse(
            torch.equal(after["confidence_score"], confidence_before)
        )

    def test_rank_branch_can_change_candidate_ordering(self):
        torch.manual_seed(19)
        module = FrozenCandidateResponsibilityHeads(
            feature_dim=8,
            hidden_dim=16,
            ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED,
            rank_residual_limit=0.25,
        )
        features = torch.randn(1, 2, 8)
        native = torch.tensor([[0.01, 0.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        self.assertEqual(int(module(features, native, mask)["rank_score"].argmax()), 0)

        optimizer = torch.optim.SGD(module.task_parameters("rank"), lr=10.0)
        output = module(features, native, mask)
        loss = output["rank_score"][0, 0] - output["rank_score"][0, 1]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertEqual(int(module(features, native, mask)["rank_score"].argmax()), 1)


class ResponsibilityOwnershipAuditTest(unittest.TestCase):
    def test_isolated_towers_start_equal_without_sharing_storage(self):
        torch.manual_seed(91)
        module = FrozenCandidateResponsibilityHeads(
            feature_dim=4,
            hidden_dim=8,
            ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED,
        )
        rank = tuple(module.rank_trunk.parameters())
        confidence = tuple(module.confidence_trunk.parameters())
        self.assertEqual(len(rank), len(confidence))
        for rank_parameter, confidence_parameter in zip(rank, confidence):
            self.assertTrue(torch.equal(rank_parameter, confidence_parameter))
            self.assertNotEqual(rank_parameter.data_ptr(), confidence_parameter.data_ptr())

    def test_reports_are_deterministic_and_match_topology(self):
        shared = FrozenCandidateResponsibilityHeads(
            feature_dim=4,
            hidden_dim=8,
            ownership=RESPONSIBILITY_OWNERSHIP_SHARED,
        )
        isolated = FrozenCandidateResponsibilityHeads(
            feature_dim=4,
            hidden_dim=8,
            ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED,
        )
        self.assertEqual(
            responsibility_ownership_report(shared),
            responsibility_ownership_report(shared),
        )
        shared_report = responsibility_ownership_report(shared)
        isolated_report = responsibility_ownership_report(isolated)
        self.assertGreater(shared_report["shared_tensor_count"], 0)
        self.assertEqual(isolated_report["shared_tensor_count"], 0)
        self.assertEqual(
            isolated_report["all_trainable_element_count"],
            shared_report["all_trainable_element_count"],
        )
        self.assertEqual(
            isolated_report["all_trainable_tensor_count"],
            shared_report["all_trainable_tensor_count"],
        )

    def test_shared_and_isolated_start_functionally_and_capacity_matched(self):
        torch.manual_seed(20260821)
        shared = FrozenCandidateResponsibilityHeads(
            feature_dim=8,
            hidden_dim=12,
            ownership=RESPONSIBILITY_OWNERSHIP_SHARED,
        )
        torch.manual_seed(20260821)
        isolated = FrozenCandidateResponsibilityHeads(
            feature_dim=8,
            hidden_dim=12,
            ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED,
        )
        self.assertEqual(
            tuple(shared.state_dict()), tuple(isolated.state_dict())
        )
        for name in shared.state_dict():
            self.assertTrue(
                torch.equal(shared.state_dict()[name], isolated.state_dict()[name])
            )
        features, native, mask = _inputs(feature_dim=8)
        shared_output = shared(features, native, mask)
        isolated_output = isolated(features, native, mask)
        for key in ("rank_residual", "rank_score", "confidence_score"):
            self.assertTrue(torch.equal(shared_output[key], isolated_output[key]))


if __name__ == "__main__":
    unittest.main()
