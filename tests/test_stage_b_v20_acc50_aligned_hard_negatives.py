import ast
import runpy
import unittest
from pathlib import Path


class StageBV20Acc50AlignedHardNegativesConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.v18_path = (
            root / "config/ablations/cfg_stageb_v18_strong_fpr_tail.py"
        )
        cls.v20_path = (
            root
            / "config/ablations/cfg_stageb_v20_acc50_aligned_hard_negatives.py"
        )
        cls.v18 = runpy.run_path(str(cls.v18_path))
        cls.v20 = runpy.run_path(str(cls.v20_path))

    def test_inherits_v18_and_only_changes_the_rank_negative_boundary(self):
        source = self.v20_path.read_text(encoding="utf-8")
        self.assertIn(
            "from config.ablations.cfg_stageb_v18_strong_fpr_tail import *",
            source,
        )
        self.assertEqual(self.v18["stage_b_v11_negative_iou_threshold"], 0.3)
        self.assertEqual(self.v20["stage_b_v11_negative_iou_threshold"], 0.499)
        self.assertTrue(self.v20["stage_b_v20_acc50_aligned_hard_negatives"])
        self.assertNotIn(
            "stage_b_v20_acc50_aligned_hard_negatives", self.v18
        )
        assigned_names = {
            target.id
            for node in ast.parse(source).body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertEqual(
            assigned_names,
            {
                "stage_b_v11_negative_iou_threshold",
                "stage_b_v20_acc50_aligned_hard_negatives",
            },
        )

    def test_preserves_v18_score_contract_and_training_hyperparameters(self):
        expected = {
            "stage_b_v15_patch_rank_fusion": True,
            "stage_b_v15_patch_rank_weight": 1.0,
            "stage_b_v15_exclude_canonical_from_score": False,
            "stage_b_v16_confidence_output_mode": "gate_only",
            "stage_b_v14_validity_head": True,
            "stage_b_v15_decoupled_confidence": True,
            "stage_b_v15_separate_grad_clip": True,
            "only_train_keywords": ["stage_b_fixed_text_scorer"],
            "only_train_exclude_keywords": [
                "stage_b_fixed_text_scorer.confidence_decoder"
            ],
            "batch_size": 56,
            "stage_b_v11_expression_microbatch": 16,
            "clip_max_norm": 0.1,
            "lr": 2e-5,
            "lr_linear_proj_mult": 2e-6,
            "stage_b_v15_validity_lr": 5e-4,
            "stage_b_v14_tail_queue_weight": 1.0,
            "stage_b_v15_tail_queue_pair_weight": 1.0,
            "stage_b_v15_tail_queue_positive_trust_weight": 1.0,
        }
        for key, expected_value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(self.v18[key], expected_value)
                self.assertEqual(self.v20[key], self.v18[key])


if __name__ == "__main__":
    unittest.main()
