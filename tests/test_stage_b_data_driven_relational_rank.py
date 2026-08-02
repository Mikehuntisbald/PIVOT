import inspect
import unittest

import torch

from models.GroundingDINO.stage_b_data_driven_score import (
    RelationalRankAdapter,
    StageBDataDrivenScoreHeads,
)


class StageBDataDrivenRelationalRankTest(unittest.TestCase):
    def _heads(self, *, seed=42):
        return StageBDataDrivenScoreHeads(
            hidden_dim=8,
            rank_dim=8,
            rank_architecture="relational_v1",
            rank_num_heads=2,
            rank_image_levels=2,
            rank_image_pool_size=2,
            rank_box_fourier_bands=2,
            rank_ffn_dim=16,
            rank_dropout=0.0,
            head_init_seed=seed,
            confidence_dim=5,
            gate_hidden_dim=6,
        )

    def _inputs(self, *, scorer_batch=2):
        torch.manual_seed(9)
        image_batch = 2
        query = torch.randn(scorer_batch, 3, 8)
        text = torch.randn(scorer_batch, 4, 8)
        token_mask = torch.tensor(
            [[True, True, True, False]] * scorer_batch, dtype=torch.bool
        )
        boxes = torch.rand(image_batch, 3, 4)
        features = [
            torch.randn(image_batch, 8, 4, 4),
            torch.randn(image_batch, 8, 2, 2),
        ]
        masks = [
            torch.zeros(image_batch, 4, 4, dtype=torch.bool),
            torch.zeros(image_batch, 2, 2, dtype=torch.bool),
        ]
        owners = (
            torch.tensor([0, 1])
            if scorer_batch == 2
            else torch.tensor([0, 0, 1, 1])
        )
        return query, text, token_mask, boxes, features, masks, owners

    def _forward(self, heads, inputs):
        query, text, token_mask, boxes, features, masks, owners = inputs
        return heads(
            query,
            text,
            token_mask,
            query_boxes=boxes,
            image_features=features,
            image_masks=masks,
            image_owner_indices=owners,
        )

    def test_single_and_paired_output_shapes_are_finite(self):
        heads = self._heads().eval()
        single = self._forward(heads, self._inputs())
        paired = self._forward(heads, self._inputs(scorer_batch=4))
        self.assertEqual(tuple(single["text_rank_score"].shape), (2, 3))
        self.assertEqual(tuple(single["text_rank_token_logits"].shape), (2, 3, 4))
        self.assertEqual(tuple(paired["text_rank_score"].shape), (4, 3))
        self.assertTrue(torch.isfinite(single["text_rank_score"]).all())
        self.assertTrue(torch.isfinite(paired["text_rank_score"]).all())

    def test_backward_detaches_every_frozen_input(self):
        heads = self._heads().train()
        values = list(self._inputs())
        for index in (0, 1, 3):
            values[index].requires_grad_(True)
        for feature in values[4]:
            feature.requires_grad_(True)
        result = self._forward(heads, tuple(values))
        result["text_rank_score"].square().mean().backward()
        self.assertIsNone(values[0].grad)
        self.assertIsNone(values[1].grad)
        self.assertIsNone(values[3].grad)
        self.assertTrue(all(feature.grad is None for feature in values[4]))
        rank_grads = [
            parameter.grad for parameter in heads.rank_parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(rank_grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in rank_grads))
        self.assertTrue(
            all(parameter.grad is None for parameter in heads.confidence_parameters())
        )

    def test_masked_text_and_image_values_do_not_change_rank(self):
        heads = self._heads().eval()
        inputs = self._inputs()
        query, text, token_mask, boxes, features, masks, owners = inputs
        masks[0][:, 0, 0] = True
        masks[1][:, 0, 0] = True
        baseline = self._forward(heads, inputs)["text_rank_score"]
        changed_text = text.clone()
        changed_text[:, 3] = 1e6
        changed_features = [feature.clone() for feature in features]
        changed_features[0][:, :, 0, 0] = 1e6
        changed_features[1][:, :, 0, 0] = -1e6
        changed = self._forward(
            heads,
            (
                query,
                changed_text,
                token_mask,
                boxes,
                changed_features,
                masks,
                owners,
            ),
        )["text_rank_score"]
        self.assertTrue(torch.equal(baseline, changed))

    def test_image_memory_and_rank_are_invariant_to_poisoned_outer_padding(self):
        heads = self._heads().eval()
        rank = heads.rank_branch
        torch.manual_seed(17)
        features = [torch.randn(1, 8, 5, 4), torch.randn(1, 8, 3, 2)]
        masks = [
            torch.tensor(
                [[[False, False, False, False],
                  [False, True, False, False],
                  [False, False, False, False],
                  [False, False, True, False],
                  [False, False, False, False]]],
                dtype=torch.bool,
            ),
            torch.tensor(
                [[[False, False], [False, True], [False, False]]],
                dtype=torch.bool,
            ),
        ]
        baseline_memory, baseline_mask = rank._pool_image_memory(features, masks)

        padded_features = []
        padded_masks = []
        for level, (feature, mask) in enumerate(zip(features, masks)):
            top, bottom, left, right = (level + 1, 2, 2 - level, 3)
            padded = torch.full(
                (
                    1,
                    8,
                    int(feature.shape[-2]) + top + bottom,
                    int(feature.shape[-1]) + left + right,
                ),
                1e6 if level == 0 else -1e6,
            )
            padded[:, :, top : top + feature.shape[-2], left : left + feature.shape[-1]] = feature
            padded_mask = torch.ones(padded.shape[0], *padded.shape[-2:], dtype=torch.bool)
            padded_mask[
                :, top : top + mask.shape[-2], left : left + mask.shape[-1]
            ] = mask
            padded_features.append(padded)
            padded_masks.append(padded_mask)

        padded_memory, padded_mask = rank._pool_image_memory(
            padded_features, padded_masks
        )
        self.assertTrue(torch.equal(baseline_mask, padded_mask))
        self.assertTrue(torch.allclose(baseline_memory, padded_memory, atol=1e-6, rtol=0.0))

        query = torch.randn(1, 3, 8)
        text = torch.randn(1, 4, 8)
        token_mask = torch.tensor([[True, True, True, False]])
        boxes = torch.rand(1, 3, 4)
        owners = torch.zeros(1, dtype=torch.long)
        baseline = heads(
            query,
            text,
            token_mask,
            query_boxes=boxes,
            image_features=features,
            image_masks=masks,
            image_owner_indices=owners,
        )["text_rank_score"]
        padded = heads(
            query,
            text,
            token_mask,
            query_boxes=boxes,
            image_features=padded_features,
            image_masks=padded_masks,
            image_owner_indices=owners,
        )["text_rank_score"]
        self.assertTrue(torch.allclose(baseline, padded, atol=1e-6, rtol=0.0))

    def test_box_text_and_image_modalities_affect_rank(self):
        heads = self._heads().eval()
        inputs = self._inputs()
        baseline = self._forward(heads, inputs)["text_rank_score"]
        for index, coordinate in ((1, (0, 0, 0)), (3, (0, 0, 0))):
            changed = list(inputs)
            changed[index] = changed[index].clone()
            changed[index][coordinate].add_(0.5)
            self.assertFalse(
                torch.allclose(
                    baseline, self._forward(heads, tuple(changed))["text_rank_score"]
                )
            )
        changed = list(inputs)
        changed[4] = [feature.clone() for feature in changed[4]]
        changed[4][-1][:, 0, 1, 1].add_(3.0)
        self.assertFalse(
            torch.allclose(
                baseline, self._forward(heads, tuple(changed))["text_rank_score"]
            )
        )

    def test_rng_isolation_preserves_confidence_across_rank_architectures(self):
        absolute = StageBDataDrivenScoreHeads(
            hidden_dim=8,
            rank_dim=8,
            rank_architecture="absolute_token",
            head_init_seed=73,
            confidence_dim=5,
            gate_hidden_dim=6,
        )
        relational = self._heads(seed=73)
        for module_name in ("confidence_branch", "confidence_gate"):
            left = getattr(absolute, module_name).state_dict()
            right = getattr(relational, module_name).state_dict()
            self.assertEqual(left.keys(), right.keys())
            self.assertTrue(all(torch.equal(left[key], right[key]) for key in left))
        repeated = self._heads(seed=73)
        self.assertTrue(
            all(
                torch.equal(value, repeated.state_dict()[key])
                for key, value in relational.state_dict().items()
            )
        )
        different = self._heads(seed=74)
        self.assertTrue(
            any(
                not torch.equal(value, different.rank_branch.state_dict()[key])
                for key, value in relational.rank_branch.state_dict().items()
            )
        )
        self.assertEqual(int(absolute._contract_version), 1)
        self.assertEqual(int(relational._contract_version), 3)

    def test_unknown_image_pool_policy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "image_pool_policy"):
            RelationalRankAdapter(
                hidden_dim=8,
                rank_dim=8,
                num_heads=2,
                image_levels=2,
                image_pool_size=2,
                image_pool_policy="padded_canvas",
                box_fourier_bands=2,
                ffn_dim=16,
            )

    def test_unknown_image_level_policy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "image_level_policy"):
            RelationalRankAdapter(
                hidden_dim=8,
                rank_dim=8,
                num_heads=2,
                image_level_policy="first",
                image_levels=2,
                image_pool_size=2,
                box_fourier_bands=2,
                ffn_dim=16,
            )

    def test_relational_surface_has_no_teacher_input_or_state(self):
        signature = set(inspect.signature(RelationalRankAdapter.forward).parameters)
        forbidden = ("teacher", "r100", "p50", "stage_a", "gdino_base", "u0")
        self.assertFalse(any(token in name for name in signature for token in forbidden))
        names = set(self._heads().state_dict())
        self.assertFalse(any(token in name.lower() for name in names for token in forbidden))


if __name__ == "__main__":
    unittest.main()
