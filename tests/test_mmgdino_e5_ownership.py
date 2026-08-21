import unittest

import torch

from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
    task_gradient_connection_report,
)


class MMGDinoE5OwnershipTests(unittest.TestCase):
    def _inputs(self):
        generator = torch.Generator().manual_seed(123)
        features = torch.randn(2, 7, 256, generator=generator)
        native = torch.rand(2, 7, generator=generator)
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[0, -1] = False
        return features, native, mask

    def test_formal_parameter_and_mac_counts(self):
        expected = {
            OWNERSHIP_SHARED_128: (50308, 49536, 128),
            OWNERSHIP_SHARED_WIDE: (100362, 99424, 210),
            OWNERSHIP_ISOLATED_128: (100358, 98816, 128),
        }
        for ownership, values in expected.items():
            with self.subTest(ownership=ownership):
                report = MMGDinoE5ResponsibilityOwners(
                    ownership=ownership
                ).architecture_report()
                self.assertEqual(report.trainable_parameters, values[0])
                self.assertEqual(report.macs_per_query_both_outputs, values[1])
                self.assertEqual(report.rank_representation_dim, values[2])

    def test_all_arms_start_with_native_rank_and_zero_confidence(self):
        features, native, mask = self._inputs()
        for ownership in (
            OWNERSHIP_SHARED_128,
            OWNERSHIP_SHARED_WIDE,
            OWNERSHIP_ISOLATED_128,
        ):
            with self.subTest(ownership=ownership):
                module = MMGDinoE5ResponsibilityOwners(ownership=ownership)
                output = module(features, native, mask)
                self.assertTrue(torch.equal(output["rank_residual"], torch.zeros_like(native)))
                self.assertTrue(torch.equal(output["rank_score"][mask], native[mask]))
                self.assertTrue(torch.equal(output["confidence_score"], torch.zeros_like(native)))
                for row, row_mask in zip(output["rank_score"], mask):
                    if bool((~row_mask).any().item()):
                        self.assertTrue(bool((row[row_mask].min() > row[~row_mask].max()).item()))

    def test_frozen_inputs_have_no_autograd_connection(self):
        features, native, mask = self._inputs()
        features.requires_grad_(True)
        native.requires_grad_(True)
        module = MMGDinoE5ResponsibilityOwners(ownership=OWNERSHIP_SHARED_128)
        output = module(features, native, mask)
        (output["rank_score"][mask].sum() + output["confidence_score"][mask].sum()).backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(native.grad)

    def test_shared_and_isolated_gradient_topology(self):
        features, native, mask = self._inputs()
        for ownership, isolated in (
            (OWNERSHIP_SHARED_128, False),
            (OWNERSHIP_SHARED_WIDE, False),
            (OWNERSHIP_ISOLATED_128, True),
        ):
            with self.subTest(ownership=ownership):
                module = MMGDinoE5ResponsibilityOwners(ownership=ownership)
                # Nonzero heads are needed for the task losses to reach trunks.
                torch.nn.init.normal_(module.rank_head.weight, std=0.01)
                torch.nn.init.normal_(module.confidence_head.weight, std=0.01)
                output = module(features, native, mask)
                report = task_gradient_connection_report(
                    module,
                    output["rank_score"][mask].square().mean(),
                    output["confidence_score"][mask].square().mean(),
                )
                self.assertEqual(report["structurally_isolated"], isolated)
                self.assertEqual(bool(report["cross_task_parameter_names"]), not isolated)

    def test_isolated_confidence_mutation_cannot_change_rank(self):
        features, native, mask = self._inputs()
        module = MMGDinoE5ResponsibilityOwners(ownership=OWNERSHIP_ISOLATED_128)
        torch.nn.init.normal_(module.rank_head.weight, std=0.01)
        before = module(features, native, mask)["rank_score"].detach().clone()
        with torch.no_grad():
            for value in module.task_parameters("confidence"):
                value.add_(torch.randn_like(value))
        after = module(features, native, mask)["rank_score"].detach()
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
