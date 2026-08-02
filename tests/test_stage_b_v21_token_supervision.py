import math
import json
import runpy
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import torch

from engine import (
    _build_stage_b_v21_certified_edit_traces,
    _preserve_stage_b_v21_trace_metadata,
)
from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)
from datasets.patch_episode import (
    PatchEpisodeJsonlDataset,
    _validate_single_edit_token_provenance,
)


def _token_criterion(objective, **overrides):
    options = {
        "listwise_weight": 0.0,
        "local_tn_rank_weight": 0.0,
        "predicate_tn_rank_weight": 0.0,
        "local_anchor_weight": 0.0,
        "global_tn_negative_weight": 0.0,
        "global_tn_tail_weight": 0.0,
        "batch_tail_separation_weight": 0.0,
        "local_absolute_weight": 0.0,
        "predicate_absolute_weight": 0.0,
        "tail_queue_weight": 0.0,
        "token_objective": objective,
        "token_weight": 1.0,
        "token_positive_weight": 1.0,
        "token_shared_weight": 1.0,
        "token_edit_weight": 1.0,
        "token_focal_alpha": 0.25,
        "token_focal_gamma": 0.0,
    }
    options.update(overrides)
    return StageBFixedTextCriterion(**options)


def _token_inputs(
    *,
    candidate_ious=None,
    predicate_pair_valid=True,
    token_supervision_valid=True,
    direct_trace_valid=True,
):
    candidate_logits = torch.zeros((1, 2), requires_grad=True)
    local_tn_logits = torch.zeros((1, 2), requires_grad=True)
    token_logits = torch.zeros((1, 2, 2, 4), requires_grad=True)
    inputs = {
        "candidate_logits": candidate_logits,
        "candidate_ious": (
            torch.tensor([[0.9, 0.2]])
            if candidate_ious is None
            else candidate_ious
        ),
        "local_tn_logits": local_tn_logits,
        "local_tn_mask": torch.tensor([True]),
        "predicate_pair_valid": torch.tensor([predicate_pair_valid]),
        "token_logits": token_logits,
        "score_token_mask": torch.tensor(
            [[[True, True, True, False], [True, True, True, False]]]
        ),
        "predicate_token_mask": torch.tensor(
            [[[True, False, False, False], [True, False, False, False]]]
        ),
        "expression_valid_mask": torch.tensor([[True, True]]),
        "token_supervision_valid": torch.tensor([token_supervision_valid]),
        "token_positive_mask": torch.tensor(
            [[[True, True, True, False], [False, False, False, False]]]
        ),
        "token_shared_mask": torch.tensor(
            [[[False, False, False, False], [False, True, True, False]]]
        ),
        "token_changed_mask": torch.tensor(
            [[[False, False, False, False], [True, False, False, False]]]
        ),
        "token_direct_trace_valid": torch.tensor([direct_trace_valid]),
    }
    if not direct_trace_valid:
        for key in (
            "token_positive_mask",
            "token_shared_mask",
            "token_changed_mask",
        ):
            inputs[key].zero_()
    return inputs


class StageBV21TokenSupervisionTest(unittest.TestCase):
    def test_off_preserves_legacy_call_contract(self):
        criterion = StageBFixedTextCriterion(token_objective="off")
        result = criterion(
            candidate_logits=torch.zeros((1, 2), requires_grad=True),
            candidate_ious=torch.tensor([[0.9, 0.1]]),
        )
        self.assertEqual(float(result["loss_fixed_text_token"].detach()), 0.0)
        self.assertEqual(float(result["fixed_text_token_sample_count"]), 0.0)
        self.assertEqual(criterion.weight_dict["loss_fixed_text_token"], 0.0)

    def test_edit_bce_supervises_positive_shared_and_changed_groups(self):
        criterion = _token_criterion("edit_bce")
        inputs = _token_inputs()
        result = criterion(**inputs)
        result["loss_stage_b_fixed_text"].backward()

        grad = inputs["token_logits"].grad
        self.assertLess(float(grad[0, 0, 0, :3].max()), 0.0)
        self.assertGreater(float(grad[0, 0, 1, 0]), 0.0)
        self.assertLess(float(grad[0, 0, 1, 1:3].max()), 0.0)
        self.assertEqual(float(grad[0, 1].abs().sum()), 0.0)
        self.assertEqual(float(grad[..., 3].abs().sum()), 0.0)
        self.assertAlmostEqual(
            float(result["loss_fixed_text_token"].detach()),
            math.log(2.0),
            places=6,
        )
        self.assertEqual(float(result["fixed_text_token_positive_count"]), 3.0)
        self.assertEqual(float(result["fixed_text_token_shared_count"]), 2.0)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 1.0)

    def test_carrier_scope_adds_only_changed_token_on_base_logit_winner(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        inputs = _token_inputs()
        # The deployed global logit is broadcast over candidates and cannot
        # identify its carrier. The candidate-specific base logit must do so.
        global_logits = torch.tensor([[3.0, 3.0]], requires_grad=True)
        carrier_logits = torch.tensor([[0.0, 2.0]], requires_grad=True)
        inputs.update(
            {
                "global_tn_confidence_logits": global_logits,
                "global_tn_verified": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([True]),
                "global_tn_candidate_mask": torch.tensor([[True, True]]),
                "token_edit_carrier_logits": carrier_logits,
            }
        )
        result = criterion(**inputs)
        result["loss_stage_b_fixed_text"].backward()

        grad = inputs["token_logits"].grad
        self.assertGreater(float(grad[0, 1, 1, 0]), 0.0)
        self.assertEqual(float(grad[0, 1, 1, 1:].abs().sum()), 0.0)
        self.assertEqual(float(grad[0, 1, 0].abs().sum()), 0.0)
        self.assertEqual(float(result["fixed_text_token_shared_count"]), 2.0)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 2.0)
        self.assertEqual(
            float(result["fixed_text_token_edit_query_count"]), 2.0
        )
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_selected_count"]), 1.0
        )
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_added_count"]), 1.0
        )
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_target_overlap_count"]),
            0.0,
        )
        self.assertEqual(float(global_logits.grad.abs().sum()), 0.0)
        self.assertIsNone(carrier_logits.grad)

    def test_role_complete_carrier_supervises_all_roles_on_deployed_winners(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_role_complete_"
                "confidence_base_argmax_v3"
            ),
        )
        inputs = _token_inputs()
        carrier_logits = torch.tensor(
            [[[0.0, 0.0], [2.0, 3.0]]], requires_grad=True
        )
        inputs.update(
            {
                "global_tn_verified": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([True]),
                "global_tn_candidate_mask": torch.tensor([[True, True]]),
                "token_role_carrier_logits": carrier_logits,
            }
        )
        result = criterion(**inputs)
        result["loss_stage_b_fixed_text"].backward()

        grad = inputs["token_logits"].grad
        self.assertLess(float(grad[0, 1, 0, :3].max()), 0.0)
        self.assertGreater(float(grad[0, 1, 1, 0]), 0.0)
        self.assertLess(float(grad[0, 1, 1, 1:3].max()), 0.0)
        self.assertEqual(float(result["fixed_text_token_positive_count"]), 6.0)
        self.assertEqual(float(result["fixed_text_token_shared_count"]), 4.0)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 2.0)
        self.assertEqual(
            float(result["fixed_text_token_role_carrier_pair_selected_count"]),
            1.0,
        )
        self.assertEqual(
            float(result["fixed_text_token_role_carrier_positive_added_count"]),
            1.0,
        )
        self.assertEqual(
            float(result["fixed_text_token_role_carrier_tn_added_count"]), 1.0
        )
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_added_count"]), 1.0
        )
        self.assertIsNone(carrier_logits.grad)

    def test_role_complete_carrier_is_pair_complete_and_fails_closed(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_role_complete_"
                "confidence_base_argmax_v3"
            ),
        )
        inputs = _token_inputs()
        inputs.update(
            {
                "global_tn_verified": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([True]),
                "global_tn_candidate_mask": torch.tensor([[True, True]]),
                "token_role_carrier_logits": torch.tensor(
                    [[[float("nan"), 0.0], [float("nan"), 2.0]]]
                ),
            }
        )
        result = criterion(**inputs)
        self.assertEqual(float(result["fixed_text_token_positive_count"]), 3.0)
        self.assertEqual(float(result["fixed_text_token_shared_count"]), 2.0)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 1.0)
        self.assertEqual(
            float(result["fixed_text_token_role_carrier_pair_selected_count"]),
            0.0,
        )

    def test_role_complete_carrier_requires_verified_train_eligible_trace(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_role_complete_"
                "confidence_base_argmax_v3"
            ),
        )
        for verified, eligible, direct_trace_valid in (
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            with self.subTest(
                verified=verified,
                eligible=eligible,
                direct_trace_valid=direct_trace_valid,
            ):
                inputs = _token_inputs(direct_trace_valid=direct_trace_valid)
                inputs.update(
                    {
                        "global_tn_verified": torch.tensor([verified]),
                        "confidence_tn_train_eligible": torch.tensor([eligible]),
                        "global_tn_candidate_mask": torch.tensor([[True, True]]),
                        "token_role_carrier_logits": torch.tensor(
                            [[[0.0, 0.0], [2.0, 2.0]]]
                        ),
                    }
                )
                result = criterion(**inputs)
                self.assertEqual(
                    float(
                        result[
                            "fixed_text_token_role_carrier_pair_selected_count"
                        ]
                    ),
                    0.0,
                )

    def test_role_complete_carrier_requires_full_pair_selector(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_role_complete_"
                "confidence_base_argmax_v3"
            ),
        )
        with self.assertRaisesRegex(ValueError, "token_role_carrier_logits"):
            criterion(**_token_inputs())

    def test_carrier_scope_does_not_duplicate_target_query(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        inputs = _token_inputs()
        inputs.update(
            {
                "global_tn_verified": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([True]),
                "global_tn_candidate_mask": torch.tensor([[True, True]]),
                "token_edit_carrier_logits": torch.tensor([[2.0, 0.0]]),
            }
        )
        result = criterion(**inputs)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 1.0)
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_selected_count"]), 1.0
        )
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_added_count"]), 0.0
        )
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_target_overlap_count"]),
            1.0,
        )

    def test_carrier_scope_requires_query_specific_selector(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        with self.assertRaisesRegex(ValueError, "token_edit_carrier_logits"):
            criterion(**_token_inputs())

    def test_carrier_scope_rejects_unverified_ablation_only_row(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        inputs = _token_inputs()
        inputs.update(
            {
                "global_tn_confidence_logits": torch.zeros((1, 2)),
                "global_tn_verified": torch.tensor([False]),
                "confidence_ablation_eligible": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([True]),
                "global_tn_candidate_mask": torch.tensor([[True, True]]),
                "token_edit_carrier_logits": torch.tensor([[0.0, 2.0]]),
            }
        )
        result = criterion(**inputs)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 1.0)
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_selected_count"]), 0.0
        )
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_added_count"]), 0.0
        )

    def test_carrier_scope_masks_inadmitted_and_nonfinite_candidates(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        for carrier_logits, candidate_mask in (
            (torch.tensor([[0.0, 5.0]]), torch.tensor([[True, False]])),
            (torch.tensor([[0.0, float("nan")]]), torch.tensor([[True, True]])),
        ):
            with self.subTest(carrier_logits=carrier_logits):
                inputs = _token_inputs()
                inputs.update(
                    {
                        "global_tn_verified": torch.tensor([True]),
                        "confidence_tn_train_eligible": torch.tensor([True]),
                        "global_tn_candidate_mask": candidate_mask,
                        "token_edit_carrier_logits": carrier_logits,
                    }
                )
                result = criterion(**inputs)
                self.assertEqual(float(result["fixed_text_token_edit_count"]), 1.0)
                self.assertEqual(
                    float(result["fixed_text_token_edit_carrier_added_count"]),
                    0.0,
                )
                self.assertEqual(
                    float(result["fixed_text_token_edit_carrier_selected_count"]),
                    1.0,
                )
                self.assertEqual(
                    float(
                        result[
                            "fixed_text_token_edit_carrier_target_overlap_count"
                        ]
                    ),
                    1.0,
                )

    def test_carrier_scope_fails_closed_without_valid_candidate(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        inputs = _token_inputs()
        inputs.update(
            {
                "global_tn_verified": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([True]),
                "global_tn_candidate_mask": torch.tensor([[False, False]]),
                "token_edit_carrier_logits": torch.tensor([[1.0, 2.0]]),
            }
        )
        result = criterion(**inputs)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 1.0)
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_selected_count"]), 0.0
        )

    def test_carrier_scope_requires_tn_train_eligibility(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        inputs = _token_inputs()
        inputs.update(
            {
                "global_tn_verified": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([False]),
                "global_tn_candidate_mask": torch.tensor([[True, True]]),
                "token_edit_carrier_logits": torch.tensor([[0.0, 2.0]]),
            }
        )
        result = criterion(**inputs)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 1.0)
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_selected_count"]), 0.0
        )

    def test_carrier_scope_requires_valid_direct_trace(self):
        criterion = _token_criterion(
            "edit_bce",
            token_edit_query_scope=(
                "target_iou_union_detached_final_confidence_base_argmax_v2"
            ),
        )
        inputs = _token_inputs(direct_trace_valid=False)
        inputs.update(
            {
                "global_tn_verified": torch.tensor([True]),
                "confidence_tn_train_eligible": torch.tensor([True]),
                "global_tn_candidate_mask": torch.tensor([[True, True]]),
                "token_edit_carrier_logits": torch.tensor([[0.0, 2.0]]),
            }
        )
        result = criterion(**inputs)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 0.0)
        self.assertEqual(
            float(result["fixed_text_token_edit_carrier_selected_count"]), 0.0
        )

    def test_explicit_default_query_scope_matches_implicit_default(self):
        implicit = _token_criterion("edit_bce")(**_token_inputs())
        explicit = _token_criterion(
            "edit_bce", token_edit_query_scope="target_iou_v1"
        )(**_token_inputs())
        for key in implicit:
            self.assertTrue(torch.equal(implicit[key], explicit[key]), key)

    def test_flat_role_weights_are_per_token_not_group_normalized(self):
        flat = _token_criterion(
            "edit_bce", token_shared_weight=0.25
        )(**_token_inputs())
        grouped = _token_criterion(
            "edit_bce_group_balanced", token_shared_weight=0.25
        )(**_token_inputs())
        expected_flat = math.log(2.0) * (3.0 + 2.0 * 0.25 + 1.0) / 6.0
        self.assertAlmostEqual(
            float(flat["loss_fixed_text_token"].detach()), expected_flat, places=6
        )
        self.assertAlmostEqual(
            float(grouped["loss_fixed_text_token"].detach()),
            math.log(2.0),
            places=6,
        )

    def test_label_scheme_and_loss_family_form_a_targetlocal_factorial(self):
        losses = {}
        for objective in (
            "targetlocal_allneg_focal",
            "targetlocal_allneg_bce",
            "edit_focal",
            "edit_bce",
        ):
            criterion = _token_criterion(
                objective,
                token_focal_alpha=0.25,
                token_focal_gamma=2.0,
            )
            inputs = _token_inputs()
            with torch.no_grad():
                # Shared TN context is deliberately confident. All-negative
                # labels penalize it; edit-aware labels preserve it.
                inputs["token_logits"][0, 0, 1, 1:3] = 1.0
            losses[objective] = float(
                criterion(**inputs)["loss_fixed_text_token"].detach()
            )
        self.assertGreater(
            losses["targetlocal_allneg_bce"],
            losses["targetlocal_allneg_focal"],
        )
        self.assertGreater(losses["edit_bce"], losses["edit_focal"])
        self.assertNotEqual(
            losses["targetlocal_allneg_focal"], losses["edit_focal"]
        )
        self.assertNotEqual(
            losses["targetlocal_allneg_bce"], losses["edit_bce"]
        )

    def test_gdino_loss_form_uses_sum_over_positive_query_normalizer(self):
        flat = _token_criterion(
            "allquery_allneg_focal",
            token_focal_alpha=0.25,
            token_focal_gamma=2.0,
        )(**_token_inputs())
        gdino = _token_criterion(
            "gdino_allquery_allneg_focal",
            token_focal_alpha=0.25,
            token_focal_gamma=2.0,
        )(**_token_inputs())
        self.assertGreater(
            float(gdino["loss_fixed_text_token"].detach()),
            float(flat["loss_fixed_text_token"].detach()),
        )
        self.assertEqual(float(gdino["fixed_text_token_target_query_count"]), 1.0)

    def test_edit_groups_require_a_valid_direct_trace(self):
        criterion = _token_criterion("edit_bce")
        inputs = _token_inputs(direct_trace_valid=False)
        result = criterion(**inputs)
        result["loss_fixed_text_token"].backward()

        grad = inputs["token_logits"].grad
        self.assertNotEqual(float(grad[:, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(grad[:, :, 1].abs().sum()), 0.0)
        self.assertEqual(float(result["fixed_text_token_shared_count"]), 0.0)
        self.assertEqual(float(result["fixed_text_token_edit_count"]), 0.0)

    def test_predicate_diff_validity_does_not_gate_direct_trace_roles(self):
        criterion = _token_criterion("edit_bce")
        inputs = _token_inputs(predicate_pair_valid=False)
        result = criterion(**inputs)
        result["loss_fixed_text_token"].backward()

        self.assertNotEqual(
            float(inputs["token_logits"].grad[:, :, 1].abs().sum()), 0.0
        )
        self.assertEqual(
            float(result["fixed_text_token_direct_trace_valid_count"]), 1.0
        )

    def test_edit_groups_fail_closed_without_certified_provenance(self):
        criterion = _token_criterion("edit_bce")
        inputs = _token_inputs(token_supervision_valid=False)
        result = criterion(**inputs)
        result["loss_fixed_text_token"].backward()

        grad = inputs["token_logits"].grad
        self.assertNotEqual(float(grad[:, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(grad[:, :, 1].abs().sum()), 0.0)
        self.assertEqual(
            float(result["fixed_text_token_provenance_valid_count"]), 0.0
        )

    def test_legacy_token_diff_fallback_is_explicit(self):
        inputs = _token_inputs()
        inputs.pop("token_supervision_valid")
        for key in (
            "token_positive_mask",
            "token_shared_mask",
            "token_changed_mask",
            "token_direct_trace_valid",
        ):
            inputs.pop(key)
        strict = _token_criterion("edit_bce")
        with self.assertRaisesRegex(ValueError, "provenance"):
            strict(**inputs)

        legacy = _token_criterion(
            "edit_bce", allow_legacy_token_diff_fallback=True
        )
        result = legacy(**inputs)
        result["loss_fixed_text_token"].backward()
        self.assertNotEqual(
            float(inputs["token_logits"].grad[:, :, 1].abs().sum()), 0.0
        )

    def test_targetlocal_all_negative_focal_does_not_touch_other_queries(self):
        criterion = _token_criterion("targetlocal_allneg_focal")
        inputs = _token_inputs()
        result = criterion(**inputs)
        result["loss_fixed_text_token"].backward()

        grad = inputs["token_logits"].grad
        self.assertLess(float(grad[0, 0, 0, :3].max()), 0.0)
        self.assertGreater(float(grad[0, 0, 1, :3].min()), 0.0)
        self.assertEqual(float(grad[0, 1].abs().sum()), 0.0)
        self.assertEqual(
            float(result["fixed_text_token_all_negative_count"]), 3.0
        )

    def test_allquery_all_negative_focal_matches_detector_style_targets(self):
        criterion = _token_criterion("allquery_allneg_focal")
        inputs = _token_inputs()
        result = criterion(**inputs)
        result["loss_fixed_text_token"].backward()

        grad = inputs["token_logits"].grad
        self.assertLess(float(grad[0, 0, 0, :3].max()), 0.0)
        self.assertGreater(float(grad[0, 1, 0, :3].min()), 0.0)
        self.assertGreater(float(grad[0, :, 1, :3].min()), 0.0)
        self.assertEqual(float(result["fixed_text_token_positive_count"]), 6.0)
        self.assertEqual(
            float(result["fixed_text_token_all_negative_count"]), 6.0
        )

    def test_token_loss_has_no_gradient_into_rank_or_confidence_scalars(self):
        criterion = _token_criterion("edit_bce")
        inputs = _token_inputs()
        confidence = torch.zeros((1, 2), requires_grad=True)
        local_confidence = torch.zeros((1, 2), requires_grad=True)
        inputs["confidence_logits"] = confidence
        inputs["local_tn_confidence_logits"] = local_confidence
        result = criterion(**inputs)
        result["loss_stage_b_fixed_text"].backward()

        self.assertNotEqual(float(inputs["token_logits"].grad.abs().sum()), 0.0)
        self.assertEqual(float(inputs["candidate_logits"].grad.abs().sum()), 0.0)
        self.assertEqual(float(inputs["local_tn_logits"].grad.abs().sum()), 0.0)
        self.assertEqual(float(confidence.grad.abs().sum()), 0.0)
        self.assertEqual(float(local_confidence.grad.abs().sum()), 0.0)

    def test_raw_word_veto_gate_anchors_positive_and_changed_word_signs(self):
        criterion = _token_criterion(
            "edit_bce",
            token_weight=0.0,
            raw_veto_gate_weight=1.0,
            raw_veto_positive_margin=0.1,
            raw_veto_tn_margin=0.1,
        )
        inputs = _token_inputs()
        inputs["confidence_tn_train_eligible"] = torch.tensor([True])
        inputs["score_word_group_ids"] = torch.tensor(
            [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
        )
        inputs["token_changed_mask"] = torch.tensor(
            [[[False, False, False, False], [True, True, False, False]]]
        )
        inputs["token_shared_mask"] = torch.tensor(
            [[[False, False, False, False], [False, False, True, False]]]
        )
        residual = torch.zeros((1, 2, 2, 4), requires_grad=True)
        inputs["token_residual_logits"] = residual

        result = criterion(**inputs)
        result["loss_stage_b_fixed_text"].backward()

        self.assertAlmostEqual(
            float(result["loss_fixed_text_raw_veto_gate"].detach()), 0.1, places=6
        )
        self.assertGreater(float(residual.grad[0, 0, 0, :3].sum()), 0.0)
        self.assertLess(float(residual.grad[0, 0, 1, :2].sum()), 0.0)
        self.assertEqual(float(residual.grad[0, 1].abs().sum()), 0.0)
        self.assertEqual(
            float(result["fixed_text_raw_veto_positive_query_count"]), 1.0
        )
        self.assertEqual(float(result["fixed_text_raw_veto_tn_query_count"]), 1.0)
        self.assertEqual(
            float(result["fixed_text_raw_veto_positive_violation_rate"]), 1.0
        )
        self.assertEqual(
            float(result["fixed_text_raw_veto_tn_violation_rate"]), 1.0
        )

    def test_raw_word_veto_all_admitted_tn_and_positive_carrier_scope(self):
        criterion = _token_criterion(
            "edit_bce",
            token_weight=0.0,
            raw_veto_gate_weight=1.0,
            raw_veto_positive_margin=0.1,
            raw_veto_tn_margin=0.15,
            raw_veto_query_scope="tn_all_admitted_positive_carrier_v2",
        )
        inputs = _token_inputs()
        inputs["confidence_tn_train_eligible"] = torch.tensor([True])
        inputs["local_tn_mask"] = torch.ones((1, 2), dtype=torch.bool)
        inputs["positive_reference_base_logits"] = torch.tensor([[0.0, 2.0]])
        inputs["score_word_group_ids"] = torch.tensor(
            [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
        )
        inputs["token_changed_mask"] = torch.tensor(
            [[[False, False, False, False], [True, True, False, False]]]
        )
        inputs["token_shared_mask"] = torch.tensor(
            [[[False, False, False, False], [False, False, True, False]]]
        )
        residual = torch.zeros((1, 2, 2, 4), requires_grad=True)
        inputs["token_residual_logits"] = residual

        result = criterion(**inputs)
        result["loss_stage_b_fixed_text"].backward()

        self.assertEqual(
            float(result["fixed_text_raw_veto_positive_query_count"]), 2.0
        )
        self.assertEqual(float(result["fixed_text_raw_veto_tn_query_count"]), 2.0)
        self.assertGreater(float(residual.grad[0, :, 0, :3].sum()), 0.0)
        self.assertLess(float(residual.grad[0, :, 1, :2].sum()), 0.0)

    def test_raw_word_veto_balances_all_tn_queries_with_tn_rank_carrier(self):
        criterion = _token_criterion(
            "edit_bce",
            token_weight=0.0,
            raw_veto_gate_weight=1.0,
            raw_veto_positive_margin=0.1,
            raw_veto_tn_margin=0.15,
            raw_veto_query_scope=(
                "tn_all_admitted_carrier_balanced_positive_carrier_v3"
            ),
            raw_veto_tn_carrier_balance=0.5,
            raw_veto_gate_offset=0.02,
            raw_veto_gate_scale=0.03,
        )
        inputs = _token_inputs()
        inputs["confidence_tn_train_eligible"] = torch.tensor([True])
        inputs["local_tn_mask"] = torch.ones((1, 2), dtype=torch.bool)
        inputs["confidence_veto_carrier_indices"] = torch.tensor([[0, 1]])
        inputs["score_word_group_ids"] = torch.tensor(
            [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
        )
        inputs["token_changed_mask"] = torch.tensor(
            [[[False, False, False, False], [True, True, False, False]]]
        )
        inputs["token_shared_mask"] = torch.tensor(
            [[[False, False, False, False], [False, False, True, False]]]
        )
        residual = torch.zeros((1, 2, 2, 4), requires_grad=True)
        inputs["token_residual_logits"] = residual

        result = criterion(**inputs)
        result["loss_stage_b_fixed_text"].backward()

        self.assertAlmostEqual(
            float(result["loss_fixed_text_raw_veto_gate"].detach()),
            0.125,
            places=6,
        )
        noncarrier_grad = float(residual.grad[0, 0, 1, :2].abs().sum())
        carrier_grad = float(residual.grad[0, 1, 1, :2].abs().sum())
        self.assertGreater(carrier_grad, 2.9 * noncarrier_grad)
        self.assertEqual(
            float(result["fixed_text_raw_veto_tn_carrier_sample_count"]), 1.0
        )
        self.assertEqual(
            float(result["fixed_text_raw_veto_tn_carrier_source_mean"]), 0.0
        )
        self.assertEqual(
            float(result["fixed_text_raw_veto_tn_carrier_violation_rate"]), 1.0
        )

    def test_carrier_balanced_raw_veto_requires_scorer_carrier_indices(self):
        criterion = _token_criterion(
            "edit_bce",
            token_weight=0.0,
            raw_veto_gate_weight=1.0,
            raw_veto_query_scope=(
                "tn_all_admitted_carrier_balanced_positive_carrier_v3"
            ),
            raw_veto_tn_carrier_balance=0.5,
            raw_veto_gate_offset=0.02,
            raw_veto_gate_scale=0.03,
        )
        inputs = _token_inputs()
        inputs["confidence_tn_train_eligible"] = torch.tensor([True])
        inputs["score_word_group_ids"] = torch.tensor(
            [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
        )
        inputs["token_residual_logits"] = torch.zeros((1, 2, 2, 4))

        with self.assertRaisesRegex(
            ValueError, "confidence_veto_carrier_indices"
        ):
            criterion(**inputs)

    def test_carrier_balancing_leaves_positive_loss_unchanged(self):
        def run(scope):
            options = {
                "token_weight": 0.0,
                "raw_veto_gate_weight": 1.0,
                "raw_veto_positive_margin": 0.1,
                "raw_veto_tn_margin": 0.15,
                "raw_veto_query_scope": scope,
            }
            if scope.endswith("_v3"):
                options.update(
                    raw_veto_tn_carrier_balance=0.5,
                    raw_veto_gate_offset=0.02,
                    raw_veto_gate_scale=0.03,
                )
            criterion = _token_criterion("edit_bce", **options)
            inputs = _token_inputs()
            inputs["confidence_tn_train_eligible"] = torch.tensor([True])
            inputs["local_tn_mask"] = torch.ones((1, 2), dtype=torch.bool)
            inputs["score_word_group_ids"] = torch.tensor(
                [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
            )
            inputs["token_changed_mask"] = torch.tensor(
                [[[False, False, False, False], [True, True, False, False]]]
            )
            inputs["token_shared_mask"] = torch.tensor(
                [[[False, False, False, False], [False, False, True, False]]]
            )
            residual = torch.zeros((1, 2, 2, 4), requires_grad=True)
            with torch.no_grad():
                residual[0, 1, 0, :3] = 0.2
            inputs["token_residual_logits"] = residual
            if scope.endswith("_v3"):
                inputs["confidence_veto_carrier_indices"] = torch.tensor(
                    [[1, 1]]
                )
            else:
                inputs["positive_reference_base_logits"] = torch.tensor(
                    [[0.0, 2.0]]
                )
            return criterion(**inputs)

        legacy = run("tn_all_admitted_positive_carrier_v2")
        balanced = run(
            "tn_all_admitted_carrier_balanced_positive_carrier_v3"
        )
        self.assertAlmostEqual(
            float(legacy["fixed_text_raw_veto_positive_loss_mean"]),
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(balanced["fixed_text_raw_veto_positive_loss_mean"]),
            float(legacy["fixed_text_raw_veto_positive_loss_mean"]),
            places=6,
        )

    def test_paired_carrier_separation_is_independent_of_v8_raw_loss(self):
        common = {
            "token_weight": 0.0,
            "raw_veto_gate_weight": 1.0,
            "raw_veto_positive_margin": 0.1,
            "raw_veto_tn_margin": 0.15,
            "raw_veto_tn_carrier_balance": 0.25,
            "raw_veto_gate_offset": 0.02,
            "raw_veto_gate_scale": 0.03,
        }

        def run(
            scope,
            pair_weight=0.0,
            pair_gradient_contract="bidirectional_v1",
        ):
            criterion = _token_criterion(
                "edit_bce",
                raw_veto_query_scope=scope,
                raw_veto_carrier_pair_weight=pair_weight,
                raw_veto_carrier_pair_margin=0.25,
                raw_veto_carrier_pair_gradient_contract=(
                    pair_gradient_contract
                ),
                **common,
            )
            inputs = _token_inputs()
            inputs["confidence_tn_train_eligible"] = torch.tensor([True])
            inputs["local_tn_mask"] = torch.ones((1, 2), dtype=torch.bool)
            inputs["local_tn_confidence_logits"] = torch.zeros((1, 2))
            inputs["confidence_veto_carrier_indices"] = torch.tensor([[0, 1]])
            inputs["score_word_group_ids"] = torch.tensor(
                [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
            )
            inputs["token_changed_mask"] = torch.tensor(
                [[[False, False, False, False], [True, True, False, False]]]
            )
            inputs["token_shared_mask"] = torch.tensor(
                [[[False, False, False, False], [False, False, True, False]]]
            )
            residual = torch.zeros((1, 2, 2, 4), requires_grad=True)
            with torch.no_grad():
                residual[0, 0, 0, :3] = -0.05
                residual[0, 1, 1, :2] = 0.05
            inputs["token_residual_logits"] = residual
            return criterion, residual, criterion(**inputs)

        _v8_criterion, v8_residual, v8 = run(
            "tn_all_admitted_carrier_balanced_positive_carrier_v3"
        )
        v9_criterion, v9_residual, v9 = run(
            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
            pair_weight=0.25,
        )
        _v42_criterion, v42_residual, v42 = run(
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
            pair_weight=0.25,
            pair_gradient_contract="tn_only_positive_detached_v2",
        )

        self.assertAlmostEqual(
            float(v9["loss_fixed_text_raw_veto_gate"].detach()),
            float(v8["loss_fixed_text_raw_veto_gate"].detach()),
            places=7,
        )
        self.assertAlmostEqual(
            float(v9["fixed_text_raw_veto_positive_loss_mean"]),
            float(v8["fixed_text_raw_veto_positive_loss_mean"]),
            places=7,
        )
        self.assertAlmostEqual(
            float(v9["fixed_text_raw_veto_tn_balanced_loss_mean"]),
            float(v8["fixed_text_raw_veto_tn_balanced_loss_mean"]),
            places=7,
        )
        v8_raw_grad = torch.autograd.grad(
            v8["loss_fixed_text_raw_veto_gate"], v8_residual
        )[0]
        v9_raw_grad = torch.autograd.grad(
            v9["loss_fixed_text_raw_veto_gate"],
            v9_residual,
            retain_graph=True,
        )[0]
        self.assertTrue(torch.equal(v9_raw_grad, v8_raw_grad))
        self.assertEqual(
            v9_criterion.weight_dict["loss_fixed_text_raw_veto_carrier_pair"],
            0.25,
        )
        self.assertAlmostEqual(
            float(v9["loss_fixed_text_raw_veto_carrier_pair"].detach()),
            0.15,
            places=6,
        )
        self.assertTrue(
            torch.equal(
                v42["loss_fixed_text_raw_veto_carrier_pair"],
                v9["loss_fixed_text_raw_veto_carrier_pair"],
            )
        )
        self.assertEqual(
            float(v9["fixed_text_raw_veto_carrier_pair_sample_count"]), 1.0
        )
        self.assertAlmostEqual(
            float(v9["fixed_text_raw_veto_carrier_pair_gap_mean"]),
            0.1,
            places=6,
        )
        self.assertAlmostEqual(
            float(v9["fixed_text_raw_veto_carrier_pair_hinge_mean"]),
            0.15,
            places=6,
        )
        self.assertEqual(
            float(v9["fixed_text_raw_veto_carrier_pair_violation_rate"]), 1.0
        )

        v9["loss_fixed_text_raw_veto_carrier_pair"].backward()
        self.assertGreater(float(v9_residual.grad[0, 0, 0, :3].sum()), 0.0)
        self.assertLess(float(v9_residual.grad[0, 1, 1, :2].sum()), 0.0)

        v42["loss_fixed_text_raw_veto_carrier_pair"].backward()
        self.assertEqual(float(v42_residual.grad[0, :, 0].abs().sum()), 0.0)
        self.assertLess(float(v42_residual.grad[0, 1, 1, :2].sum()), 0.0)
        self.assertEqual(float(v42_residual.grad[0, 0, 1].abs().sum()), 0.0)

    def test_tail_weighted_carrier_focuses_high_global_score_changed_word(self):
        criterion = _token_criterion(
            "edit_bce",
            token_weight=0.0,
            raw_veto_gate_weight=1.0,
            raw_veto_positive_margin=0.1,
            raw_veto_tn_margin=0.15,
            raw_veto_query_scope=(
                "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6"
            ),
            raw_veto_tn_carrier_balance=0.99,
            raw_veto_carrier_pair_weight=0.0,
            raw_veto_gate_offset=0.02,
            raw_veto_gate_scale=0.03,
            raw_veto_tail_quantile=0.5,
            raw_veto_tail_temperature=0.1,
            raw_veto_tail_min_count=256,
        )
        base = _token_inputs()
        inputs = {}
        for name, value in base.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
                repeats = (2,) + (1,) * (value.dim() - 1)
                inputs[name] = value.detach().repeat(repeats)
            else:
                inputs[name] = value
        inputs["candidate_logits"] = torch.zeros((2, 2), requires_grad=True)
        inputs["local_tn_logits"] = torch.zeros((2, 2), requires_grad=True)
        inputs["token_logits"] = torch.zeros((2, 2, 2, 4), requires_grad=True)
        residual = torch.zeros((2, 2, 2, 4), requires_grad=True)
        inputs["token_residual_logits"] = residual
        inputs["local_tn_mask"] = torch.ones((2, 2), dtype=torch.bool)
        inputs["confidence_tn_train_eligible"] = torch.tensor([True, True])
        inputs["confidence_veto_carrier_indices"] = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        inputs["score_word_group_ids"] = torch.tensor(
            [
                [[0, 0, 1, -1], [0, 0, 1, -1]],
                [[0, 0, 1, -1], [0, 0, 1, -1]],
            ],
            dtype=torch.long,
        )
        inputs["token_changed_mask"] = torch.tensor(
            [
                [[False, False, False, False], [True, True, False, False]],
                [[False, False, False, False], [True, True, False, False]],
            ]
        )
        inputs["token_shared_mask"] = torch.tensor(
            [
                [[False, False, False, False], [False, False, True, False]],
                [[False, False, False, False], [False, False, True, False]],
            ]
        )
        inputs["local_tn_confidence_logits"] = torch.tensor(
            [[6.0, 6.0], [0.0, 0.0]]
        )

        result = criterion(**inputs)
        result["loss_fixed_text_raw_veto_gate"].backward()
        high_tail_grad = residual.grad[0, 1, 1, :2].abs().sum()
        low_tail_grad = residual.grad[1, 1, 1, :2].abs().sum()
        self.assertGreater(float(high_tail_grad), 20.0 * float(low_tail_grad))
        self.assertEqual(float(residual.grad[..., 2:].abs().sum()), 0.0)
        self.assertGreater(
            float(result["fixed_text_raw_veto_tn_tail_weight_mean"]), 0.0
        )
        self.assertLess(
            float(result["fixed_text_raw_veto_tn_tail_effective_sample_count"]),
            1.1,
        )

    def test_tail_weighted_pair_reuses_tn_tail_weights_one_to_one(self):
        common = {
            "token_weight": 0.0,
            "raw_veto_gate_weight": 1.0,
            "raw_veto_positive_margin": 0.1,
            "raw_veto_tn_margin": 0.15,
            "raw_veto_tn_carrier_balance": 0.99,
            "raw_veto_carrier_pair_weight": 1.0,
            "raw_veto_carrier_pair_margin": 0.25,
            "raw_veto_gate_offset": 0.02,
            "raw_veto_gate_scale": 0.03,
            "raw_veto_tail_quantile": 0.5,
            "raw_veto_tail_temperature": 0.1,
            "raw_veto_tail_min_count": 256,
        }

        def run(scope):
            criterion = _token_criterion(
                "edit_bce", raw_veto_query_scope=scope, **common
            )
            base = _token_inputs()
            inputs = {}
            for name, value in base.items():
                if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
                    repeats = (2,) + (1,) * (value.dim() - 1)
                    inputs[name] = value.detach().repeat(repeats)
                else:
                    inputs[name] = value
            inputs["candidate_logits"] = torch.zeros((2, 2), requires_grad=True)
            inputs["local_tn_logits"] = torch.zeros((2, 2), requires_grad=True)
            inputs["token_logits"] = torch.zeros(
                (2, 2, 2, 4), requires_grad=True
            )
            residual = torch.zeros((2, 2, 2, 4), requires_grad=True)
            with torch.no_grad():
                residual[:, 0, 0, :2] = -0.05
                residual[:, 0, 0, 2] = -0.2
                residual[0, 1, 1, :2] = 0.05
                residual[1, 1, 1, :2] = -0.15
            inputs["token_residual_logits"] = residual
            inputs["local_tn_mask"] = torch.ones((2, 2), dtype=torch.bool)
            inputs["confidence_tn_train_eligible"] = torch.tensor([True, True])
            inputs["confidence_veto_carrier_indices"] = torch.tensor(
                [[0, 1], [0, 1]], dtype=torch.long
            )
            inputs["score_word_group_ids"] = torch.tensor(
                [
                    [[0, 0, 1, -1], [0, 0, 1, -1]],
                    [[0, 0, 1, -1], [0, 0, 1, -1]],
                ],
                dtype=torch.long,
            )
            inputs["token_changed_mask"] = torch.tensor(
                [
                    [[False, False, False, False], [True, True, False, False]],
                    [[False, False, False, False], [True, True, False, False]],
                ]
            )
            inputs["token_shared_mask"] = torch.tensor(
                [
                    [[False, False, False, False], [False, False, True, False]],
                    [[False, False, False, False], [False, False, True, False]],
                ]
            )
            tail_scores = torch.tensor(
                [[6.0, 6.0], [0.0, 0.0]], requires_grad=True
            )
            inputs["local_tn_confidence_logits"] = tail_scores
            return residual, tail_scores, criterion(**inputs)

        v17_residual, _v17_tail_scores, v17 = run(
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6"
        )
        v18_residual, v18_tail_scores, v18 = run(
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
        )

        self.assertTrue(
            torch.equal(
                v17["loss_fixed_text_raw_veto_gate"],
                v18["loss_fixed_text_raw_veto_gate"],
            )
        )
        self.assertEqual(
            v17["fixed_text_raw_veto_tn_balanced_loss_mean"],
            v18["fixed_text_raw_veto_tn_balanced_loss_mean"],
        )
        self.assertNotIn("fixed_text_raw_veto_tail_pair_hinge_mean", v17)
        self.assertAlmostEqual(
            float(v17["loss_fixed_text_raw_veto_carrier_pair"].detach()),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(v18["loss_fixed_text_raw_veto_carrier_pair"].detach()),
            0.15,
            places=5,
        )
        self.assertAlmostEqual(
            float(v18["fixed_text_raw_veto_tail_pair_gap_mean"]),
            0.1,
            places=5,
        )
        self.assertAlmostEqual(
            float(v18["fixed_text_raw_veto_tail_pair_hinge_mean"]),
            0.15,
            places=5,
        )
        self.assertAlmostEqual(
            float(v18["fixed_text_raw_veto_tail_pair_violation_rate"]),
            1.0,
            places=6,
        )
        self.assertLess(
            float(v18["fixed_text_raw_veto_tail_pair_effective_sample_count"]),
            1.1,
        )

        v18["loss_fixed_text_raw_veto_carrier_pair"].backward()
        high_positive_grad = v18_residual.grad[0, 0, 0, :2]
        high_tn_grad = v18_residual.grad[0, 1, 1, :2]
        low_pair_grad = v18_residual.grad[1, :, :, :2].abs().sum()
        self.assertGreater(float(high_positive_grad.sum()), 0.0)
        self.assertLess(float(high_tn_grad.sum()), 0.0)
        self.assertGreater(
            float(high_positive_grad.abs().sum()), 1e10 * float(low_pair_grad)
        )
        self.assertEqual(float(v18_residual.grad[..., 2:].abs().sum()), 0.0)
        self.assertIsNone(v18_tail_scores.grad)
        self.assertIsNone(v17_residual.grad)

    def test_tail_weighted_pair_requires_trace_and_same_sample_carriers(self):
        criterion = _token_criterion(
            "edit_bce",
            token_weight=0.0,
            raw_veto_gate_weight=1.0,
            raw_veto_query_scope=(
                "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
            ),
            raw_veto_tn_carrier_balance=0.25,
            raw_veto_carrier_pair_weight=1.0,
            raw_veto_gate_offset=0.02,
            raw_veto_gate_scale=0.03,
        )
        invalid_trace = _token_inputs(direct_trace_valid=False)
        invalid_trace["confidence_tn_train_eligible"] = torch.tensor([True])
        invalid_trace["confidence_veto_carrier_indices"] = torch.tensor(
            [[0, 1]], dtype=torch.long
        )
        invalid_trace["score_word_group_ids"] = torch.tensor(
            [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
        )
        invalid_trace["token_residual_logits"] = torch.zeros(
            (1, 2, 2, 4), requires_grad=True
        )
        invalid_trace["local_tn_confidence_logits"] = torch.zeros((1, 2))
        result = criterion(**invalid_trace)
        self.assertEqual(
            float(result["fixed_text_raw_veto_carrier_pair_sample_count"]), 0.0
        )
        self.assertEqual(
            float(result["loss_fixed_text_raw_veto_carrier_pair"].detach()), 0.0
        )

        missing_positive_carrier = _token_inputs()
        missing_positive_carrier["confidence_tn_train_eligible"] = torch.tensor(
            [True]
        )
        missing_positive_carrier["confidence_veto_carrier_indices"] = torch.tensor(
            [[-1, 1]], dtype=torch.long
        )
        missing_positive_carrier["score_word_group_ids"] = torch.tensor(
            [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
        )
        missing_positive_carrier["token_changed_mask"] = torch.tensor(
            [[[False, False, False, False], [True, True, False, False]]]
        )
        missing_positive_carrier["token_shared_mask"] = torch.tensor(
            [[[False, False, False, False], [False, False, True, False]]]
        )
        missing_positive_carrier["token_residual_logits"] = torch.zeros(
            (1, 2, 2, 4), requires_grad=True
        )
        missing_positive_carrier["local_tn_confidence_logits"] = torch.zeros(
            (1, 2)
        )
        with self.assertRaisesRegex(
            RuntimeError, "positive row has no confidence carrier"
        ):
            criterion(**missing_positive_carrier)

    def test_dual_carrier_balance_focuses_positive_inference_carrier(self):
        criterion = _token_criterion(
            "edit_bce",
            token_weight=0.0,
            raw_veto_gate_weight=1.0,
            raw_veto_positive_margin=0.1,
            raw_veto_tn_margin=0.15,
            raw_veto_query_scope=(
                "tn_all_admitted_dual_carrier_balanced_paired_v5"
            ),
            raw_veto_tn_carrier_balance=0.25,
            raw_veto_positive_carrier_balance=0.25,
            raw_veto_carrier_pair_weight=0.25,
            raw_veto_carrier_pair_margin=0.25,
            raw_veto_gate_offset=0.02,
            raw_veto_gate_scale=0.03,
        )
        inputs = _token_inputs()
        inputs["confidence_tn_train_eligible"] = torch.tensor([True])
        inputs["local_tn_mask"] = torch.ones((1, 2), dtype=torch.bool)
        inputs["confidence_veto_carrier_indices"] = torch.tensor([[1, 1]])
        inputs["score_word_group_ids"] = torch.tensor(
            [[[0, 0, 1, -1], [0, 0, 1, -1]]], dtype=torch.long
        )
        inputs["token_changed_mask"] = torch.tensor(
            [[[False, False, False, False], [True, True, False, False]]]
        )
        inputs["token_shared_mask"] = torch.tensor(
            [[[False, False, False, False], [False, False, True, False]]]
        )
        residual = torch.zeros((1, 2, 2, 4), requires_grad=True)
        with torch.no_grad():
            residual[0, 1, 0, :3] = 0.2
        inputs["token_residual_logits"] = residual

        result = criterion(**inputs)
        self.assertAlmostEqual(
            float(result["fixed_text_raw_veto_positive_all_hinge_mean"]),
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(result["fixed_text_raw_veto_positive_carrier_hinge_mean"]),
            0.3,
            places=6,
        )
        self.assertAlmostEqual(
            float(result["fixed_text_raw_veto_positive_loss_mean"]),
            0.225,
            places=6,
        )
        self.assertAlmostEqual(
            float(result["loss_fixed_text_raw_veto_gate"].detach()),
            0.1875,
            places=6,
        )
        raw_grad = torch.autograd.grad(
            result["loss_fixed_text_raw_veto_gate"], residual
        )[0]
        noncarrier_grad = float(raw_grad[0, 0, 0].abs().sum())
        carrier_grad = float(raw_grad[0, 1, 0].abs().sum())
        self.assertGreater(carrier_grad, 1.6 * noncarrier_grad)

    def test_carrier_pair_weight_requires_paired_scope(self):
        with self.assertRaisesRegex(ValueError, "paired carrier query scope"):
            _token_criterion(
                "edit_bce",
                raw_veto_carrier_pair_weight=0.25,
                raw_veto_query_scope=(
                    "tn_all_admitted_carrier_balanced_positive_carrier_v3"
                ),
                raw_veto_tn_carrier_balance=0.25,
                raw_veto_gate_offset=0.02,
                raw_veto_gate_scale=0.03,
            )

        for value in (float("nan"), float("inf"), -0.1):
            with self.subTest(pair_weight=value):
                with self.assertRaisesRegex(
                    ValueError, "raw_veto_carrier_pair_weight"
                ):
                    _token_criterion(
                        "edit_bce", raw_veto_carrier_pair_weight=value
                    )

        with self.assertRaisesRegex(
            ValueError, "TN-only carrier-pair gradients require"
        ):
            _token_criterion(
                "edit_bce",
                raw_veto_carrier_pair_gradient_contract=(
                    "tn_only_positive_detached_v2"
                ),
            )

        with self.assertRaisesRegex(
            ValueError, "raw_veto_carrier_pair_gradient_contract"
        ):
            _token_criterion(
                "edit_bce",
                raw_veto_carrier_pair_gradient_contract="unknown",
            )

        with self.assertRaisesRegex(ValueError, "dual-carrier v5 query scope"):
            _token_criterion(
                "edit_bce",
                raw_veto_query_scope=(
                    "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4"
                ),
                raw_veto_tn_carrier_balance=0.25,
                raw_veto_positive_carrier_balance=0.25,
                raw_veto_gate_offset=0.02,
                raw_veto_gate_scale=0.03,
            )

    def test_invalid_objective_and_missing_tensor_contracts_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "token_objective"):
            _token_criterion("unknown")
        criterion = _token_criterion("edit_bce")
        with self.assertRaisesRegex(ValueError, "token_logits"):
            criterion(
                candidate_logits=torch.zeros((1, 1)),
                candidate_ious=torch.ones((1, 1)),
            )


class StageBV21WiringAndConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_v19_and_v20_companion_configs_enable_identical_token_objective(self):
        v19_path = (
            self.root / "config/ablations/cfg_stageb_v21_edit_token_supervision.py"
        )
        v20_path = self.root / (
            "config/ablations/"
            "cfg_stageb_v21_edit_token_supervision_acc50_hardneg.py"
        )
        self.assertIn(
            "cfg_stageb_v19_full_text_base_plus_gate import *",
            v19_path.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "cfg_stageb_v19_full_text_base_plus_gate import *",
            v20_path.read_text(encoding="utf-8"),
        )
        v19 = runpy.run_path(str(v19_path))
        v20 = runpy.run_path(str(v20_path))
        for config in (v19, v20):
            self.assertEqual(config["stage_b_v21_token_objective"], "edit_bce")
            self.assertEqual(config["stage_b_v21_token_weight"], 1.0)
            self.assertEqual(config["stage_b_v21_token_positive_weight"], 1.0)
            self.assertEqual(config["stage_b_v21_token_shared_weight"], 0.25)
            self.assertEqual(config["stage_b_v21_token_edit_weight"], 1.0)
            self.assertEqual(config["stage_b_v21_token_focal_alpha"], 0.25)
            self.assertFalse(
                config["stage_b_v21_allow_legacy_token_diff_fallback"]
            )
        self.assertEqual(v19["stage_b_v16_confidence_output_mode"], "base_plus_gate")
        self.assertEqual(v20["stage_b_v16_confidence_output_mode"], "base_plus_gate")
        self.assertTrue(v20["stage_b_v19_explicit_confidence_output_contract"])
        self.assertEqual(v20["stage_b_v11_negative_iou_threshold"], 0.499)

    def test_builder_and_engine_forward_all_token_contract_tensors(self):
        builder = (
            self.root / "models/GroundingDINO/groundingdino.py"
        ).read_text(encoding="utf-8")
        engine = (self.root / "engine.py").read_text(encoding="utf-8")
        for name in (
            "stage_b_v21_token_objective",
            "stage_b_v21_token_weight",
            "stage_b_v21_token_positive_weight",
            "stage_b_v21_token_shared_weight",
            "stage_b_v21_token_edit_weight",
            "stage_b_v21_token_focal_alpha",
            "stage_b_v21_token_focal_gamma",
            "stage_b_v21_allow_legacy_token_diff_fallback",
            "stage_b_dense_duty_raw_veto_carrier_pair_weight",
            "stage_b_dense_duty_raw_veto_carrier_pair_margin",
        ):
            self.assertIn(name, builder)
        for value in (
            'token_logits=outputs["stage_b_v11_final_token_logits"]',
            'score_token_mask=outputs["stage_b_v15_score_token_mask"]',
            '"stage_b_v11_predicate_token_mask"',
            '"stage_b_v11_expression_valid_mask"',
            '"stage_b_v21_token_supervision_valid"',
            '"stage_b_data_driven_trace"',
            '"stage_b_v21_positive_token_mask"',
            '"stage_b_v21_shared_token_mask"',
            '"stage_b_v21_changed_token_mask"',
            '"stage_b_v21_direct_trace_valid"',
        ):
            self.assertIn(value, engine)

    def test_patch_only_trace_copy_and_provenance_gate_are_fail_closed(self):
        trace = {
            "category": "color",
            "replace_from": "red",
            "replace_to": "blue",
            "replace_span": [0, 1],
        }
        destination = {}
        _preserve_stage_b_v21_trace_metadata(
            {"stage_b_data_driven_trace": trace}, destination
        )
        trace["replace_to"] = "green"
        self.assertEqual(
            destination["stage_b_data_driven_trace"]["replace_to"], "blue"
        )

        certified = {
            "stage_b_v21_token_supervision_valid": torch.tensor([True]),
            "stage_b_data_driven_trace": destination[
                "stage_b_data_driven_trace"
            ],
        }
        uncertified = {
            "stage_b_v21_token_supervision_valid": torch.tensor([False]),
            "stage_b_data_driven_trace": destination[
                "stage_b_data_driven_trace"
            ],
        }
        traces = _build_stage_b_v21_certified_edit_traces(
            [certified, uncertified]
        )
        self.assertEqual(traces[0]["replace_to"], "blue")
        self.assertIsNone(traces[1])
        with self.assertRaisesRegex(RuntimeError, "lost its direct edit trace"):
            _build_stage_b_v21_certified_edit_traces(
                [
                    {
                        "stage_b_v21_token_supervision_valid": torch.tensor(
                            [True]
                        )
                    }
                ]
            )
        with self.assertRaisesRegex(TypeError, "must remain a mapping"):
            _preserve_stage_b_v21_trace_metadata(
                {"stage_b_data_driven_trace": "forged"}, {}
            )

    def test_actual_single_edit_datasetinfo_is_fail_closed_and_valid(self):
        config_path = self.root / "config/datasets_stageb_v21_single_edit_train.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(len(config["train"]), 4)
        tn_sources = [
            source
            for source in config["train"]
            if source.get("require_single_edit_token_provenance", False)
        ]
        self.assertEqual(len(tn_sources), 1)
        tn_source = tn_sources[0]
        self.assertEqual(tn_source["source"], "sam3_tn_pair")
        self.assertFalse(tn_source["require_global_tn_verified"])
        self.assertTrue(tn_source["build_text_token_masks"])
        self.assertEqual(tn_source["paper_table_b_id"], "D3")
        self.assertTrue(
            tn_source["anno"].endswith("/d3_proposal_covered_train.jsonl")
        )

        actual_path = self.root / (
            "data/ablations/stageb_tn_table_b_equal_exposure_20260717/"
            "d3_proposal_covered_train.jsonl"
        )
        row_count = 0
        with actual_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                _validate_single_edit_token_provenance(
                    row, context=f"{actual_path}:{line_number}"
                )
                row_count += 1
        self.assertGreater(row_count, 0)

    def test_row_authored_boolean_cannot_bypass_provenance_validation(self):
        row = {
            "stage_b_v21_token_supervision_valid": True,
            "tn_edits": [
                {
                    "category": "color",
                    "replace_from": "red",
                    "replace_to": "blue",
                    "replace_span": [1, 2],
                }
            ],
            "replace_category": ["wrong"],
            "replace_from": ["red"],
            "replace_to": ["blue"],
            "replace_span": [[1, 2]],
        }
        with self.assertRaisesRegex(ValueError, "replace_category"):
            _validate_single_edit_token_provenance(row, context="unit-test")

    def test_runtime_flag_is_granted_only_by_datasetinfo_contract(self):
        row = {
            "stage_b_v21_token_supervision_valid": True,
            "sample_id": "unit:single-edit",
            "class_id": 1,
            "image_id": 1,
            "sent": "red car",
            "try_tn": "blue car",
            "target_bbox_used": [0, 0, 10, 10],
            "tn_edits": [
                {
                    "category": "color",
                    "replace_from": "red",
                    "replace_to": "blue",
                    "replace_span": [0, 1],
                }
            ],
            "replace_category": ["color"],
            "replace_from": ["red"],
            "replace_to": ["blue"],
            "replace_span": [[0, 1]],
        }

        def normalize(require_contract):
            dataset = PatchEpisodeJsonlDataset.__new__(PatchEpisodeJsonlDataset)
            dataset.anno = "unit.jsonl"
            dataset.root = "/tmp"
            dataset.source = "sam3_tn_pair"
            dataset._fixed_stagea_exact_rows = {}
            dataset.cfg = SimpleNamespace(
                require_single_edit_token_provenance=require_contract,
                require_fixed_stagea_topk_exact_verified=False,
                require_global_tn_verified=False,
                require_proposalset_proxy_verified=False,
                patch_text_aug_max_words=0,
                build_text_token_masks=True,
                sam3_tn_image_root="/tmp",
                sam3_tn_bbox_key="target_bbox_used",
            )
            with redirect_stdout(StringIO()):
                return dataset._normalize_sam3_tn_pair_metas([row])[0]

        self.assertFalse(normalize(False)["stage_b_v21_token_supervision_valid"])
        self.assertTrue(normalize(True)["stage_b_v21_token_supervision_valid"])


if __name__ == "__main__":
    unittest.main()
