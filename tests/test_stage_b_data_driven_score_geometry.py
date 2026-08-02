import unittest

import torch

from models.GroundingDINO.stage_b_data_driven_score import (
    groundingdino_raw_dot_phrase_geometry,
)
from models.GroundingDINO.stage_b_gdino_score_adapter import (
    aggregate_gdino_full_expression_score,
)
from models.GroundingDINO.utils import ContrastiveEmbed


class StageBDataDrivenScoreGeometryTest(unittest.TestCase):
    def setUp(self):
        self.query = torch.tensor(
            [
                [[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]],
                [[-1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
            ]
        )
        self.text = torch.tensor(
            [
                [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [9.0, 9.0, 9.0]],
                [[0.5, 1.0, 0.0], [1.0, -1.0, 1.0], [-2.0, 0.5, 1.0]],
            ]
        )
        self.text_mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        self.expression_mask = torch.tensor(
            [[False, True, False], [True, False, True]], dtype=torch.bool
        )

    def test_matches_contrastive_embed_and_authoritative_aggregation(self):
        query_before = self.query.clone()
        text_before = self.text.clone()
        observed = groundingdino_raw_dot_phrase_geometry(
            self.query,
            self.text,
            self.text_mask,
            self.expression_mask,
            max_text_len=5,
        )

        expected_logits = ContrastiveEmbed(max_text_len=5)(
            self.query,
            {
                "encoded_text": self.text,
                "text_token_mask": self.text_mask,
            },
        )
        expected_mask = torch.zeros(2, 5, dtype=torch.bool)
        expected_mask[:, :3] = self.expression_mask
        expected_score = aggregate_gdino_full_expression_score(
            expected_logits, expected_mask
        )

        self.assertTrue(torch.equal(observed["token_logits"], expected_logits))
        self.assertTrue(
            torch.equal(observed["expression_token_mask"], expected_mask)
        )
        self.assertTrue(torch.equal(observed["score"], expected_score))
        self.assertTrue(torch.equal(self.query, query_before))
        self.assertTrue(torch.equal(self.text, text_before))

    def test_padding_and_phrase_mask_do_not_score_unselected_tokens(self):
        observed = groundingdino_raw_dot_phrase_geometry(
            self.query,
            self.text,
            self.text_mask,
            self.expression_mask,
            max_text_len=6,
        )
        logits = observed["token_logits"]
        mask = observed["expression_token_mask"]

        self.assertEqual(tuple(logits.shape), (2, 2, 6))
        self.assertEqual(tuple(mask.shape), (2, 6))
        self.assertTrue(torch.isneginf(logits[0, :, 2:]).all().item())
        self.assertTrue(torch.isneginf(logits[1, :, 3:]).all().item())
        self.assertFalse(mask[:, 3:].any().item())
        manual = torch.stack(
            (
                logits[0, :, 1].float().sigmoid(),
                logits[1, :, [0, 2]].float().sigmoid().mean(dim=-1),
            )
        )
        self.assertTrue(torch.equal(observed["score"], manual))

    def test_mixed_dtypes_follow_cpu_autocast_contrastive_geometry(self):
        mixed_text = self.text.to(dtype=torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            observed = groundingdino_raw_dot_phrase_geometry(
                self.query,
                mixed_text,
                self.text_mask,
                self.expression_mask,
                max_text_len=5,
            )
            expected_logits = ContrastiveEmbed(max_text_len=5)(
                self.query,
                {
                    "encoded_text": mixed_text,
                    "text_token_mask": self.text_mask,
                },
            )
        expected_mask = torch.zeros(2, 5, dtype=torch.bool)
        expected_mask[:, :3] = self.expression_mask
        expected_score = aggregate_gdino_full_expression_score(
            expected_logits, expected_mask
        )

        self.assertTrue(torch.equal(observed["token_logits"], expected_logits))
        self.assertTrue(torch.equal(observed["score"], expected_score))

        with self.assertRaisesRegex(ValueError, "require autocast"):
            groundingdino_raw_dot_phrase_geometry(
                self.query,
                mixed_text,
                self.text_mask,
                self.expression_mask,
                max_text_len=5,
            )

    def test_rejects_invalid_geometry_inputs(self):
        valid_args = (
            self.query,
            self.text,
            self.text_mask,
            self.expression_mask,
        )
        with self.assertRaisesRegex(TypeError, "must be tensors"):
            groundingdino_raw_dot_phrase_geometry(
                [], self.text, self.text_mask, self.expression_mask, max_text_len=5
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            groundingdino_raw_dot_phrase_geometry(
                self.query[0],
                self.text,
                self.text_mask,
                self.expression_mask,
                max_text_len=5,
            )
        with self.assertRaisesRegex(TypeError, "floating point"):
            groundingdino_raw_dot_phrase_geometry(
                self.query.long(),
                self.text,
                self.text_mask,
                self.expression_mask,
                max_text_len=5,
            )
        with self.assertRaisesRegex(ValueError, "require autocast"):
            groundingdino_raw_dot_phrase_geometry(
                self.query.double(),
                self.text,
                self.text_mask,
                self.expression_mask,
                max_text_len=5,
            )
        with self.assertRaisesRegex(ValueError, "batches/dimensions"):
            groundingdino_raw_dot_phrase_geometry(
                self.query,
                self.text[:, :, :2],
                self.text_mask,
                self.expression_mask,
                max_text_len=5,
            )
        with self.assertRaisesRegex(TypeError, "integer"):
            groundingdino_raw_dot_phrase_geometry(*valid_args, max_text_len=True)
        with self.assertRaisesRegex(ValueError, "shorter"):
            groundingdino_raw_dot_phrase_geometry(*valid_args, max_text_len=2)
        nonfinite_query = self.query.clone()
        nonfinite_query[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite values"):
            groundingdino_raw_dot_phrase_geometry(
                nonfinite_query,
                self.text,
                self.text_mask,
                self.expression_mask,
                max_text_len=5,
            )

    def test_rejects_invalid_masks(self):
        with self.assertRaisesRegex(TypeError, "boolean tensor"):
            groundingdino_raw_dot_phrase_geometry(
                self.query,
                self.text,
                self.text_mask.long(),
                self.expression_mask,
                max_text_len=5,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            groundingdino_raw_dot_phrase_geometry(
                self.query,
                self.text,
                self.text_mask[:, :2],
                self.expression_mask,
                max_text_len=5,
            )
        empty_text = self.text_mask.clone()
        empty_text[0] = False
        with self.assertRaisesRegex(ValueError, "valid token"):
            groundingdino_raw_dot_phrase_geometry(
                self.query,
                self.text,
                empty_text,
                self.expression_mask & empty_text,
                max_text_len=5,
            )
        empty_expression = self.expression_mask.clone()
        empty_expression[0] = False
        with self.assertRaisesRegex(ValueError, "scored token"):
            groundingdino_raw_dot_phrase_geometry(
                self.query,
                self.text,
                self.text_mask,
                empty_expression,
                max_text_len=5,
            )
        outside_text = self.expression_mask.clone()
        outside_text[0, 2] = True
        with self.assertRaisesRegex(ValueError, "subset"):
            groundingdino_raw_dot_phrase_geometry(
                self.query,
                self.text,
                self.text_mask,
                outside_text,
                max_text_len=5,
            )


if __name__ == "__main__":
    unittest.main()
