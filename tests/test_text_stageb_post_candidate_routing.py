import types
import unittest
from unittest import mock

import torch

from tools import eval_text_groundingdino_refcoco_tn as text_eval


def _post_candidate_cfg(**overrides):
    values = {
        "stage_b_v11_fixed_text": True,
        "stage_b_v7": False,
        "stage_b_v15_decoupled_confidence": True,
        "stage_b_gdino_score_adapter": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class TextStageBPostCandidateRoutingTest(unittest.TestCase):
    def test_ref_selects_rank_and_tn_selects_confidence(self):
        cfg = _post_candidate_cfg()
        rank = torch.tensor([[[0.1], [0.9], [0.2]]])
        confidence = torch.tensor([[[0.6], [0.4], [0.5]]])
        outputs = {
            "pred_boxes": torch.zeros((1, 3, 4)),
            "stage_b_v15_dense_rank_score": rank,
            "stage_b_v7_final_score": confidence,
            # A misleading base output must never be used by this branch.
            "pred_logits": torch.full((1, 3, 4), 99.0),
        }
        self.assertTrue(
            torch.equal(
                text_eval._single_post_candidate_slot_scores(
                    outputs, cfg, branch="rank"
                ),
                rank[..., 0],
            )
        )
        self.assertTrue(
            torch.equal(
                text_eval._single_post_candidate_slot_scores(
                    outputs, cfg, branch="confidence"
                ),
                confidence[..., 0],
            )
        )

    def test_missing_post_candidate_outputs_fail_without_pred_logits_fallback(self):
        cfg = _post_candidate_cfg()
        misleading = {
            "pred_boxes": torch.zeros((1, 3, 4)),
            "pred_logits": torch.full((1, 3, 4), 99.0),
        }
        with self.assertRaisesRegex(KeyError, "dense_rank_score"):
            text_eval._single_post_candidate_slot_scores(
                misleading, cfg, branch="rank"
            )
        with self.assertRaisesRegex(KeyError, "post-candidate eval requires"):
            text_eval._single_post_candidate_slot_scores(
                misleading, cfg, branch="confidence"
            )

    def test_post_candidate_batch_routing_reuses_specialized_forwards(self):
        cfg = _post_candidate_cfg()
        batch = (object(), [{"caption": "red object"}])
        targets = [{"boxes": torch.zeros((1, 4))}]
        rank_outputs = {
            "pred_boxes": torch.zeros((1, 2, 4)),
            "stage_b_v15_dense_rank_score": torch.tensor([[[0.2], [0.8]]]),
        }
        neg_outputs = {
            "pred_boxes": torch.zeros((1, 2, 4)),
            "stage_b_v7_final_score": torch.tensor([[[0.1], [0.3]]]),
        }
        pos_outputs = {
            "pred_boxes": torch.zeros((1, 2, 4)),
            "stage_b_v7_final_score": torch.tensor([[[0.7], [0.6]]]),
        }
        direct_model = mock.Mock(side_effect=AssertionError("direct model path used"))
        with mock.patch.object(
            text_eval,
            "_stage_b_ref_forward",
            return_value=(rank_outputs, targets),
        ) as ref_forward:
            outputs, observed_targets, rank_scores = text_eval._forward_ref_batch(
                cfg, direct_model, batch, torch.device("cpu"), amp=False
            )
        ref_forward.assert_called_once()
        self.assertIs(outputs, rank_outputs)
        self.assertIs(observed_targets, targets)
        self.assertTrue(torch.equal(rank_scores, torch.tensor([[0.2, 0.8]])))

        with mock.patch.object(
            text_eval,
            "_stage_b_tn_forward_pair",
            return_value=(
                neg_outputs,
                pos_outputs,
                targets,
                torch.tensor([True]),
            ),
        ) as tn_forward:
            result = text_eval._forward_tn_batch(
                cfg, direct_model, batch, torch.device("cpu"), amp=False
            )
        tn_forward.assert_called_once()
        self.assertIs(result[0], neg_outputs)
        self.assertIs(result[1], pos_outputs)
        self.assertTrue(torch.equal(result[4], torch.tensor([[0.1, 0.3]])))
        self.assertTrue(torch.equal(result[6], torch.tensor([[0.7, 0.6]])))
        self.assertEqual(result[3].tolist(), [True])
        self.assertEqual(result[5].tolist(), [True])
        self.assertEqual(result[7].tolist(), [True])

    def test_adapter_config_keeps_direct_ref_path_even_if_v11_flag_is_present(self):
        cfg = _post_candidate_cfg(stage_b_gdino_score_adapter=True)
        batch = (torch.zeros((1, 3, 2, 2)), [{"caption": "red object"}])
        outputs = {"pred_boxes": torch.zeros((1, 2, 4))}
        model = mock.Mock(return_value=outputs)
        with mock.patch.object(text_eval, "_stage_b_ref_forward") as ref_forward:
            observed, targets, rank_scores = text_eval._forward_ref_batch(
                cfg, model, batch, torch.device("cpu"), amp=False
            )
        ref_forward.assert_not_called()
        model.assert_called_once()
        self.assertIs(observed, outputs)
        self.assertEqual(targets, batch[1])
        self.assertIsNone(rank_scores)

    def test_post_candidate_slot_shape_must_match_boxes(self):
        cfg = _post_candidate_cfg()
        with self.assertRaisesRegex(RuntimeError, "exactly one expression slot"):
            text_eval._single_post_candidate_slot_scores(
                {
                    "pred_boxes": torch.zeros((1, 2, 4)),
                    "stage_b_v15_dense_rank_score": torch.zeros((1, 2, 2)),
                },
                cfg,
                branch="rank",
            )
        with self.assertRaisesRegex(RuntimeError, "does not align"):
            text_eval._single_post_candidate_slot_scores(
                {
                    "pred_boxes": torch.zeros((1, 3, 4)),
                    "stage_b_v15_dense_rank_score": torch.zeros((1, 2, 1)),
                },
                cfg,
                branch="rank",
            )


if __name__ == "__main__":
    unittest.main()
