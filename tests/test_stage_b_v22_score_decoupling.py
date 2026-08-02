import runpy
import unittest
from pathlib import Path

import torch

from models.GroundingDINO.stage_b_fixed_text_scorer import (
    FixedBoxFullTextScorer,
    normalize_stage_b_score_ownership,
    select_stage_b_rank_confidence_logits,
)
from util.stage_b_task_gradients import (
    branch_isolation_report,
    gradient_conflict_report,
    weighted_stage_b_task_losses,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "config" / "ablations"


def _config(name):
    return runpy.run_path(str(CONFIG_ROOT / name))


class _MinimalSourceDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Linear(4, 4)
        layer.use_text_cross_attention = True
        self.layers = torch.nn.ModuleList([layer])
        self.d_model = 4
        self.num_layers = 1
        self.return_intermediate = True
        self.norm = torch.nn.LayerNorm(4)
        self.ref_point_head = torch.nn.Linear(4, 4)
        self.bbox_embed = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.class_embed = torch.nn.ModuleList([torch.nn.Linear(4, 1)])


class StageBTaskGradientTest(unittest.TestCase):
    def test_shared_gradient_conflict_reports_cosine_norms_and_fraction(self):
        shared = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        rank_loss = shared.sum()
        confidence_loss = -shared.sum()
        report = gradient_conflict_report(
            rank_loss, confidence_loss, [("shared", shared)]
        )
        self.assertAlmostEqual(report["cosine"], -1.0, places=7)
        self.assertTrue(report["cosine_defined"])
        self.assertAlmostEqual(report["rank_norm"], 2**0.5, places=7)
        self.assertAlmostEqual(report["confidence_norm"], 2**0.5, places=7)
        self.assertEqual(report["element_conflict_fraction"], 1.0)
        self.assertEqual(report["tensor_conflict_fraction"], 1.0)
        self.assertEqual(report["shared_parameter_names"], ("shared",))

    def test_shared_gradient_diagnostic_fails_on_independent_parameters(self):
        rank = torch.nn.Parameter(torch.tensor(1.0))
        confidence = torch.nn.Parameter(torch.tensor(2.0))
        with self.assertRaisesRegex(RuntimeError, "No shared trainable parameters"):
            gradient_conflict_report(
                rank.square(),
                confidence.square(),
                [("rank", rank), ("confidence", confidence)],
            )

    def test_independent_branches_pass_bidirectional_isolation(self):
        rank = torch.nn.Parameter(torch.tensor(1.0))
        confidence = torch.nn.Parameter(torch.tensor(2.0))
        report = branch_isolation_report(
            rank.square(),
            confidence.square(),
            [("rank", rank)],
            [("confidence", confidence)],
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["rank_parameter_count"], 1)
        self.assertEqual(report["confidence_parameter_count"], 1)

    def test_branch_isolation_fails_on_structural_cross_gradient(self):
        rank = torch.nn.Parameter(torch.tensor(1.0))
        confidence = torch.nn.Parameter(torch.tensor(2.0))
        confidence_loss = (rank + confidence).square()
        with self.assertRaisesRegex(RuntimeError, "branch isolation failed"):
            branch_isolation_report(
                rank.square(),
                confidence_loss,
                [("rank", rank)],
                [("confidence", confidence)],
            )

    def test_weighted_task_groups_use_only_active_contract_terms(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        losses = {
            "loss_fixed_text_listwise": parameter,
            "loss_fixed_text_local_absolute": -parameter,
            "ignored": 100.0 * parameter,
        }
        rank, confidence = weighted_stage_b_task_losses(
            losses,
            {
                "loss_fixed_text_listwise": 2.0,
                "loss_fixed_text_local_absolute": 3.0,
                "ignored": 10.0,
            },
        )
        self.assertEqual(float(rank.detach()), 2.0)
        self.assertEqual(float(confidence.detach()), -3.0)


class StageBScoreOwnershipConfigTest(unittest.TestCase):
    def test_ownership_labels_are_closed_and_normalized(self):
        self.assertEqual(
            normalize_stage_b_score_ownership("shared-trunk-two-heads"),
            "shared_trunk_two_heads",
        )
        with self.assertRaisesRegex(ValueError, "score_ownership"):
            normalize_stage_b_score_ownership("not-a-contract")

    def test_score_tensor_routing_matches_s0_s2_contracts(self):
        phrase = torch.randn(2, 3, 2, requires_grad=True)
        validity = torch.randn(2, 3, 2, requires_grad=True)
        outputs = {
            "stage_b_v11_final_phrase_logits": phrase,
            "stage_b_v14_final_validity_logits": validity,
        }
        s0_rank, s0_confidence = select_stage_b_rank_confidence_logits(
            outputs, score_ownership="shared_score"
        )
        self.assertIs(s0_rank, phrase)
        self.assertIs(s0_confidence, phrase)
        for ownership in (
            "shared_trunk_two_heads",
            "independent_decoders_joint",
            "independent_decoders_two_phase",
        ):
            with self.subTest(ownership=ownership):
                rank, confidence = select_stage_b_rank_confidence_logits(
                    outputs, score_ownership=ownership
                )
                self.assertIs(rank, phrase)
                self.assertIs(confidence, validity)

    def test_s2_constructor_keeps_independent_confidence_decoder_frozen(self):
        scorer = FixedBoxFullTextScorer(
            _MinimalSourceDecoder(),
            num_layers=1,
            max_text_len=8,
            use_validity_head=True,
            decouple_validity_from_ranking=True,
            score_ownership="independent_decoders_joint",
        )
        self.assertTrue(any(parameter.requires_grad for parameter in scorer.decoder.parameters()))
        self.assertTrue(
            any(parameter.requires_grad for parameter in scorer.validity_head.parameters())
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in scorer.confidence_decoder.parameters()
            )
        )
        self.assertIn("_score_contract_ownership", scorer.state_dict())

    def test_constructor_rejects_mislabeled_historical_layout(self):
        with self.assertRaisesRegex(ValueError, "shared_trunk_two_heads"):
            FixedBoxFullTextScorer(
                _MinimalSourceDecoder(),
                num_layers=1,
                max_text_len=8,
                use_validity_head=True,
                decouple_validity_from_ranking=True,
                score_ownership="shared_trunk_two_heads",
            )

    def test_s0_s3_parameter_ownership_is_explicit(self):
        expected = {
            "cfg_stageb_v22_s0_shared_score.py": (
                "S0",
                "shared_score",
                "joint",
                False,
                False,
                5_626_240,
            ),
            "cfg_stageb_v22_s1_shared_trunk_two_heads.py": (
                "S1",
                "shared_trunk_two_heads",
                "joint",
                True,
                False,
                5_692_289,
            ),
            "cfg_stageb_v22_s2_independent_joint.py": (
                "S2",
                "independent_decoders_joint",
                "joint",
                True,
                True,
                5_692_289,
            ),
            "cfg_stageb_v22_s3_rank_phase.py": (
                "S3-rank",
                "independent_decoders_two_phase",
                "rank",
                True,
                True,
                5_626_240,
            ),
            "cfg_stageb_v22_s3_isolation_probe.py": (
                "S3-isolation-probe",
                "independent_decoders_two_phase",
                "isolation_probe",
                True,
                True,
                5_692_289,
            ),
            "cfg_stageb_v22_s3_confidence_phase.py": (
                "S3-confidence",
                "independent_decoders_two_phase",
                "confidence",
                True,
                True,
                66_049,
            ),
        }
        for filename, contract in expected.items():
            with self.subTest(filename=filename):
                config = _config(filename)
                actual = (
                    config["stage_b_v22_table_id"],
                    config["stage_b_v22_score_ownership"],
                    config["stage_b_v22_train_phase"],
                    config["stage_b_v14_validity_head"],
                    config["stage_b_v15_decoupled_confidence"],
                    config["stage_b_v11_trainable_params_min"],
                )
                self.assertEqual(actual, contract)
                self.assertEqual(
                    config["stage_b_v11_trainable_params_min"],
                    config["stage_b_v11_trainable_params_max"],
                )

    def test_s2_clean_block_differs_only_by_gate_specific_trust_term(self):
        reference = _config(
            "cfg_stageb_v21_edit_token_supervision_acc50_hardneg.py"
        )
        s2 = _config("cfg_stageb_v22_s2_independent_joint.py")
        objective_keys = (
            "stage_b_v11_listwise_weight",
            "stage_b_v11_local_tn_rank_weight",
            "stage_b_v11_predicate_tn_rank_weight",
            "stage_b_v11_local_anchor_weight",
            "stage_b_v14_local_absolute_weight",
            "stage_b_v11_global_tn_negative_weight",
            "stage_b_v11_global_tn_tail_weight",
            "stage_b_v14_tail_queue_weight",
            "stage_b_v15_tail_queue_pair_weight",
            "stage_b_v21_token_objective",
            "stage_b_v21_token_weight",
            "stage_b_v21_token_positive_weight",
            "stage_b_v21_token_shared_weight",
            "stage_b_v21_token_edit_weight",
        )
        for key in objective_keys:
            with self.subTest(key=key):
                self.assertEqual(s2[key], reference[key])
        self.assertEqual(
            s2["stage_b_v22_objective_fidelity"],
            "common_objective_ownership_ablation",
        )
        self.assertEqual(s2["stage_b_v15_tail_queue_positive_trust_weight"], 0.0)
        full = _config("cfg_stageb_v22_s2_independent_joint_full.py")
        for key in (*objective_keys, "stage_b_v15_tail_queue_positive_trust_weight"):
            with self.subTest(full_objective_key=key):
                self.assertEqual(full[key], reference[key])
        self.assertEqual(
            full["stage_b_v22_objective_fidelity"],
            "full_v19_base_plus_gate_objective",
        )
        self.assertIn(
            "stage_b_fixed_text_scorer.confidence_decoder",
            s2["only_train_exclude_keywords"],
        )

    def test_s0_s1_disclose_only_gate_specific_objective_mismatch(self):
        reference = _config(
            "cfg_stageb_v21_edit_token_supervision_acc50_hardneg.py"
        )
        for filename in (
            "cfg_stageb_v22_s0_shared_score.py",
            "cfg_stageb_v22_s1_shared_trunk_two_heads.py",
        ):
            with self.subTest(filename=filename):
                config = _config(filename)
                self.assertEqual(
                    config["stage_b_v15_tail_queue_positive_trust_weight"], 0.0
                )
                self.assertEqual(
                    config["stage_b_v22_missing_gate_objective"],
                    "positive_residual_trust_and_translation",
                )
                for key in (
                    "stage_b_v14_tail_queue_weight",
                    "stage_b_v15_tail_queue_pair_weight",
                    "stage_b_v11_global_tn_negative_weight",
                    "stage_b_v11_global_tn_tail_weight",
                    "stage_b_v21_token_objective",
                    "stage_b_v21_token_weight",
                ):
                    self.assertEqual(config[key], reference[key])

    def test_s3_splits_the_fixed_total_schedule_and_trainable_branches(self):
        reference = _config("cfg_stageb_v22_s2_independent_joint.py")
        rank = _config("cfg_stageb_v22_s3_rank_phase.py")
        confidence = _config("cfg_stageb_v22_s3_confidence_phase.py")
        self.assertEqual(rank["epochs"] + confidence["epochs"], reference["epochs"])
        self.assertEqual(
            rank["only_train_keywords"], ["stage_b_fixed_text_scorer.decoder"]
        )
        self.assertEqual(
            confidence["only_train_keywords"],
            ["stage_b_fixed_text_scorer.validity_head"],
        )
        for key in (
            "stage_b_v11_listwise_weight",
            "stage_b_v11_local_tn_rank_weight",
            "stage_b_v11_predicate_tn_rank_weight",
            "stage_b_v14_local_absolute_weight",
            "stage_b_v14_tail_queue_weight",
            "stage_b_v21_token_weight",
        ):
            with self.subTest(key=key):
                self.assertEqual(rank[key], reference[key])
                self.assertEqual(confidence[key], reference[key])

    def test_all_table_d_rows_lock_acc50_and_predicate_pair_rank(self):
        for filename in (
            "cfg_stageb_v22_s0_shared_score.py",
            "cfg_stageb_v22_s1_shared_trunk_two_heads.py",
            "cfg_stageb_v22_s2_independent_joint.py",
            "cfg_stageb_v22_s3_rank_phase.py",
            "cfg_stageb_v22_s3_confidence_phase.py",
            "cfg_stageb_v22_s3_isolation_probe.py",
        ):
            with self.subTest(filename=filename):
                config = _config(filename)
                self.assertEqual(
                    config["stage_b_v11_negative_iou_threshold"], 0.499
                )
                self.assertEqual(
                    config["stage_b_v11_predicate_tn_rank_weight"], 1.0
                )
                self.assertEqual(
                    config["stage_b_v22_full_predicate_tn_rank_weight"], 1.0
                )


if __name__ == "__main__":
    unittest.main()
