import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tools import eval_refcoco_stageb as ref_eval
from util.misc import NestedTensor


def _patch_cfg(**overrides):
    values = {
        "stage_b_v11_fixed_text": True,
        "stage_b_gdino_score_adapter": False,
        "stage_b_v15_decoupled_confidence": True,
        "stage_b_v15_patch_rank_fusion": True,
        "stage_b_v15_patch_rank_weight": 1.0,
        "stage_b_v11_candidate_topk": 50,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _external_cfg(**overrides):
    values = {
        "stage_b_gdino_score_adapter": True,
        "stage_b_v11_fixed_text": False,
        "stage_b_v7": False,
        "num_queries": 900,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _patch_outputs():
    pred_boxes = torch.tensor(
        [[[0.20, 0.20, 0.10, 0.10], [0.70, 0.70, 0.20, 0.20], [0.5, 0.5, 0.1, 0.1]]],
        dtype=torch.float32,
    )
    candidate_idx = torch.tensor([[0, 1]], dtype=torch.int64)
    patch_logits = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    fused_logits = patch_logits.unsqueeze(-1)
    expression_valid = torch.ones((1, 1), dtype=torch.bool)
    scatter_idx = candidate_idx.unsqueeze(-1)
    dense_logits = torch.full(
        (1, 3, 1), torch.finfo(torch.float32).min, dtype=torch.float32
    )
    dense_logits.scatter_(1, scatter_idx, fused_logits)
    dense_score = torch.zeros((1, 3, 1), dtype=torch.float32)
    dense_score.scatter_(1, scatter_idx, fused_logits.sigmoid())
    candidate_mask = torch.zeros((1, 3, 1), dtype=torch.bool)
    candidate_mask.scatter_(1, scatter_idx, True)
    return {
        "pred_boxes": pred_boxes,
        "stage_b_v11_final_phrase_logits": fused_logits,
        "stage_b_v15_candidate_patch_logits": patch_logits,
        "stage_b_v11_candidate_idx": candidate_idx,
        "stage_b_v11_expression_valid_mask": expression_valid,
        "stage_b_v11_candidate_mask": candidate_mask,
        "stage_b_v15_dense_rank_logits": dense_logits,
        "stage_b_v15_dense_rank_score": dense_score,
    }


def _external_outputs():
    boxes = torch.full((1, 900, 4), 0.05, dtype=torch.float32)
    boxes[..., :2] = 0.95
    boxes[0, 0] = torch.tensor([0.20, 0.20, 0.10, 0.10])
    boxes[0, 1] = torch.tensor([0.70, 0.70, 0.20, 0.20])
    rank = torch.full((1, 900), -5.0, dtype=torch.float32)
    rank[0, 0] = 0.25
    rank[0, 1] = 3.0
    base = torch.full((1, 900), 0.01, dtype=torch.float32)
    base[0, 0] = 0.9
    base[0, 1] = 0.8
    return {
        "pred_boxes": boxes,
        "stage_b_gdino_base_score": base,
        "stage_b_gdino_rank_score": rank,
        "stage_b_gdino_confidence_score": torch.full_like(rank, 100.0),
    }


def _runtime_settings():
    settings = {
        "transfer_modes": ["nearest_iou", "max_score_iou_power"],
        "iou_powers": [1.0, 2.0],
        "patch_weights": [1.0],
        "text_weights": [1.0],
        "candidate_topk": 2,
        "contract_patch_rank_weight": 1.0,
        "external_query_count": 900,
    }
    settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
        settings
    )
    return settings


class _AdapterModel(nn.Module):
    def __init__(self, *, include_rank=True):
        super().__init__()
        self.stage_b_gdino_score_adapter = object()
        self.include_rank = include_rank
        self.calls = []

    def forward(self, samples, *, captions):
        self.calls.append(list(captions))
        batch_size = len(captions)
        outputs = {
            "pred_boxes": torch.full((batch_size, 900, 4), 0.2),
            "stage_b_gdino_base_score": torch.zeros(batch_size, 900),
            "stage_b_gdino_confidence_score": torch.zeros(batch_size, 900),
        }
        if self.include_rank:
            outputs["stage_b_gdino_rank_score"] = torch.ones(batch_size, 900)
        return outputs


class ExternalGDINORankTransferTest(unittest.TestCase):
    def test_ref_split_seeds_are_stable_across_subset_and_reordering(self):
        full = dict(
            (name, seed)
            for name, _spec, seed in ref_eval._requested_ref_split_specs(
                ["all"], 42
            )
        )
        subset = dict(
            (name, seed)
            for name, _spec, seed in ref_eval._requested_ref_split_specs(
                ["refcocog_val", "refcoco_testB"], 42
            )
        )
        reordered = dict(
            (name, seed)
            for name, _spec, seed in ref_eval._requested_ref_split_specs(
                ["refcoco_testB", "refcocog_val"], 42
            )
        )
        self.assertEqual(subset, reordered)
        self.assertEqual(subset["refcocog_val"], full["refcocog_val"])
        self.assertEqual(subset["refcoco_testB"], full["refcoco_testB"])
        self.assertEqual(full["refcocog_val"], 600042)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ref_eval._requested_ref_split_specs(
                ["refcoco_val", "refcoco_val"], 42
            )
        with self.assertRaisesRegex(KeyError, "Unknown"):
            ref_eval._requested_ref_split_specs(["not_a_ref_split"], 42)

    def test_exact_transfer_formulas_and_signed_zero_iou_policy(self):
        pair_iou = torch.tensor(
            [[[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32
        )
        rank_score = torch.tensor([[-2.0, -1.0, -100.0]], dtype=torch.float32)
        transferred = ref_eval._transfer_external_rank_scores_from_iou(
            pair_iou,
            rank_score,
            transfer_modes=["nearest_iou", "max_score_iou_power"],
            iou_powers=[1.0, 2.0],
            expected_query_count=3,
        )
        self.assertTrue(
            torch.equal(
                transferred[("nearest_iou", None)],
                torch.tensor([[-2.0, -2.0]]),
            )
        )
        # The zero-IoU -100 query is excluded instead of creating a synthetic 0.
        self.assertTrue(
            torch.equal(
                transferred[("max_score_iou_power", 1.0)],
                torch.tensor([[-0.5, -2.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                transferred[("max_score_iou_power", 2.0)],
                torch.tensor([[-0.25, -2.0]]),
            )
        )

    def test_top_query_nearest_candidate_preserves_external_top1_mapping(self):
        pair_iou = torch.tensor(
            [
                [
                    [0.8, 0.1, 0.2, 0.0],
                    [0.8, 0.7, 0.2, 0.0],
                    [0.1, 0.6, 0.9, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        rank_score = torch.tensor([[2.0, 9.0, 4.0, -3.0]])
        transferred = ref_eval._transfer_external_rank_scores_from_iou(
            pair_iou,
            rank_score,
            transfer_modes=["top_query_nearest_candidate"],
            iou_powers=[],
            expected_query_count=4,
        )[("top_query_nearest_candidate", None)]

        # Query 0 has an IoU tie and is assigned to first candidate 0. Query 3
        # has an all-zero tie and follows the same deterministic first-index rule.
        self.assertTrue(torch.equal(transferred, torch.tensor([[2.0, 9.0, 4.0]])))
        external_top_query = int(rank_score[0].argmax().item())
        expected_candidate = int(pair_iou[0, :, external_top_query].argmax().item())
        self.assertEqual(int(transferred[0].argmax().item()), expected_candidate)

        tied_rank = torch.tensor([[9.0, 1.0, 9.0, -3.0]])
        tied = ref_eval._transfer_external_rank_scores_from_iou(
            pair_iou,
            tied_rank,
            transfer_modes=["top_query_nearest_candidate"],
            iou_powers=[],
            expected_query_count=4,
        )[("top_query_nearest_candidate", None)]
        first_top_query = int(tied_rank[0].argmax().item())
        first_top_candidate = int(
            pair_iou[0, :, first_top_query].argmax().item()
        )
        patch_zero_fused = 0.0 * torch.tensor([[100.0, -100.0, 50.0]]) + tied
        self.assertEqual(
            int(patch_zero_fused[0].argmax().item()), first_top_candidate
        )

    def test_fixed_candidates_and_boxes_are_invariant_across_grid(self):
        patch_outputs = _patch_outputs()
        external_outputs = _external_outputs()
        snapshots = {key: value.clone() for key, value in patch_outputs.items()}
        (
            transferred,
            patch_score,
            candidate_idx,
            candidate_boxes,
            _pair_iou,
            _external_boxes,
        ) = (
            ref_eval._diagnostic_external_rank_candidate_scores(
                patch_outputs,
                external_outputs,
                settings=_runtime_settings(),
            )
        )
        self.assertEqual(len(transferred), 3)
        self.assertTrue(torch.equal(candidate_idx, torch.tensor([[0, 1]])))
        self.assertTrue(torch.equal(patch_score, torch.tensor([[2.0, 1.0]])))
        expected_all_boxes = ref_eval._normalized_cxcywh_to_xyxy(
            patch_outputs["pred_boxes"], name="expected"
        )
        self.assertTrue(torch.equal(candidate_boxes, expected_all_boxes[:, :2]))
        for key, before in snapshots.items():
            self.assertTrue(torch.equal(patch_outputs[key], before), key)

    def test_external_box_mode_keeps_candidate_winner_but_uses_text_box(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = [
            "max_score_iou_power",
            "max_score_iou_power_external_box",
        ]
        settings["iou_powers"] = [1.0]
        settings["patch_weights"] = [0.0]
        settings["text_weights"] = [1.0]
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        external = _external_outputs()
        external["pred_boxes"] = external["pred_boxes"].clone()
        external["pred_boxes"][0, 1] = torch.tensor(
            [0.70, 0.70, 0.40, 0.40]
        )
        accumulator = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            settings
        )
        accumulator.update(
            _patch_outputs(),
            external,
            [{"boxes": torch.tensor([[0.70, 0.70, 0.40, 0.40]])}],
        )
        rows = {row["diagnostic_transfer_mode"]: row for row in accumulator.results()}
        self.assertEqual(rows["max_score_iou_power"]["acc50"], 0.0)
        self.assertEqual(
            rows["max_score_iou_power_external_box"]["acc50"], 1.0
        )
        self.assertEqual(
            rows["max_score_iou_power"]["candidate_oracle_recall50"], 0.0
        )
        self.assertEqual(
            rows["max_score_iou_power_external_box"][
                "candidate_oracle_recall50"
            ],
            0.0,
        )

    def test_top_query_external_box_preserves_global_text_query_geometry(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = [
            "top_query_nearest_candidate",
            "top_query_nearest_candidate_external_box",
        ]
        settings["iou_powers"] = []
        settings["patch_weights"] = [0.0]
        settings["text_weights"] = [1.0]
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        external = _external_outputs()
        external["pred_boxes"] = external["pred_boxes"].clone()
        external["pred_boxes"][0, 1] = torch.tensor(
            [0.70, 0.70, 0.40, 0.40]
        )
        accumulator = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            settings
        )
        accumulator.update(
            _patch_outputs(),
            external,
            [{"boxes": torch.tensor([[0.70, 0.70, 0.40, 0.40]])}],
        )
        rows = {row["diagnostic_transfer_mode"]: row for row in accumulator.results()}
        self.assertEqual(rows["top_query_nearest_candidate"]["acc50"], 0.0)
        self.assertEqual(
            rows["top_query_nearest_candidate_external_box"]["acc50"], 1.0
        )

    def test_fusion_uses_patch_logit_and_adapter_rank_key(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = ["nearest_iou"]
        settings["iou_powers"] = []
        settings["patch_weights"] = [1.0]
        settings["text_weights"] = [0.0, 1.0]
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        accumulator = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            settings
        )
        accumulator.update(
            _patch_outputs(),
            _external_outputs(),
            [{"boxes": torch.tensor([[0.70, 0.70, 0.20, 0.20]])}],
        )
        rows = {
            row["diagnostic_text_weight"]: row for row in accumulator.results()
        }
        self.assertEqual(rows[0.0]["acc50"], 0.0)
        self.assertEqual(rows[1.0]["acc50"], 1.0)
        for row in rows.values():
            self.assertIs(row["diagnostic_only"], True)
            self.assertIs(row["formal_gate_eligible"], False)
            self.assertNotIn("records_jsonl", row)

    def test_patch_internal_rank_identity_is_exact_standard_ref_beta0(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = ["nearest_iou"]
        settings["iou_powers"] = []
        settings["patch_weights"] = [0.0]
        settings["text_weights"] = [1.0]
        settings["include_patch_internal_rank_identity"] = True
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        patch_outputs = _patch_outputs()
        patch_outputs["stage_a_captions"] = ["object ."]
        patch_cfg = _patch_cfg(stage_b_v11_candidate_topk=2)

        standard_scores = ref_eval._slot_scores(patch_outputs, patch_cfg, 0.0)
        candidate_idx = patch_outputs["stage_b_v11_candidate_idx"]
        (
            identity_scores,
            identity_winner_flat,
            identity_winner_query,
            identity_winner_box,
        ) = ref_eval._diagnostic_patch_internal_rank_identity_winner(
            patch_outputs,
            patch_cfg=patch_cfg,
            candidate_idx=candidate_idx,
        )
        standard_winner_flat = standard_scores.reshape(1, -1).argmax(dim=1)
        standard_winner_query = torch.div(
            standard_winner_flat,
            standard_scores.shape[-1],
            rounding_mode="floor",
        )
        standard_boxes = ref_eval._normalized_cxcywh_to_xyxy(
            patch_outputs["pred_boxes"], name="standard boxes"
        )
        standard_winner_box = standard_boxes[
            torch.arange(standard_boxes.shape[0]), standard_winner_query
        ]
        self.assertTrue(torch.equal(identity_scores, standard_scores))
        self.assertTrue(torch.equal(identity_winner_flat, standard_winner_flat))
        self.assertTrue(torch.equal(identity_winner_query, standard_winner_query))
        self.assertTrue(torch.equal(identity_winner_box, standard_winner_box))

        target = {"boxes": torch.tensor([[0.20, 0.20, 0.10, 0.10]])}
        accumulator = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            settings, patch_cfg=patch_cfg
        )
        accumulator.update(patch_outputs, _external_outputs(), [target])
        identity_rows = [
            row
            for row in accumulator.results()
            if row.get("diagnostic_descriptor_kind")
            == ref_eval._PATCH_INTERNAL_RANK_IDENTITY_KIND
        ]
        self.assertEqual(len(identity_rows), 1)
        identity_row = identity_rows[0]
        expected_iou = float(
            ref_eval.box_ops.box_iou(
                standard_winner_box,
                ref_eval.box_ops.box_cxcywh_to_xyxy(target["boxes"]),
            )[0]
            .view(-1)[0]
            .item()
        )
        self.assertEqual(identity_row["mean_iou_top1"], expected_iou)
        self.assertEqual(identity_row["acc50"], float(expected_iou >= 0.5))
        self.assertNotIn("diagnostic_transfer_mode", identity_row)
        self.assertNotIn("diagnostic_patch_weight", identity_row)
        self.assertFalse(identity_row["uses_external_rank_score"])
        self.assertFalse(identity_row["formal_gate_eligible"])

    def test_external_base_identity_is_direct_zero_adapter_global_argmax(self):
        external = _external_outputs()
        external["stage_b_gdino_base_score"] = torch.full(
            (1, 900), 0.01, dtype=torch.float32
        )
        external["stage_b_gdino_base_score"][0, 4] = 0.99
        external["stage_b_gdino_base_score"][0, 8] = 0.99
        external["pred_boxes"] = external["pred_boxes"].clone()
        external["pred_boxes"][0, 4] = torch.tensor(
            [0.30, 0.40, 0.20, 0.10]
        )
        external["pred_boxes"][0, 8] = torch.tensor(
            [0.80, 0.70, 0.10, 0.20]
        )

        base_score, winner_query, winner_box = (
            ref_eval._diagnostic_external_gdino_base_identity_winner(external)
        )
        direct_winner = external["stage_b_gdino_base_score"].argmax(dim=1)
        zero_adapter_scores = ref_eval._slot_scores(
            {"stage_b_gdino_rank_score": base_score},
            _external_cfg(),
            0.0,
        )
        zero_adapter_winner = torch.div(
            zero_adapter_scores.reshape(1, -1).argmax(dim=1),
            zero_adapter_scores.shape[-1],
            rounding_mode="floor",
        )
        expected_boxes = ref_eval._normalized_cxcywh_to_xyxy(
            external["pred_boxes"], name="expected external boxes"
        )
        expected_winner_box = expected_boxes[
            torch.arange(expected_boxes.shape[0]), direct_winner
        ]
        self.assertTrue(
            torch.equal(base_score, external["stage_b_gdino_base_score"])
        )
        self.assertEqual(winner_query.tolist(), [4])
        self.assertTrue(torch.equal(winner_query, direct_winner))
        self.assertTrue(torch.equal(winner_query, zero_adapter_winner))
        self.assertTrue(torch.equal(winner_box, expected_winner_box))

    def test_external_base_identity_validates_key_shape_and_finiteness(self):
        missing = _external_outputs()
        missing.pop("stage_b_gdino_base_score")
        with self.assertRaisesRegex(KeyError, "stage_b_gdino_base_score"):
            ref_eval._diagnostic_external_gdino_base_identity_winner(missing)

        wrong_query_count = _external_outputs()
        wrong_query_count["stage_b_gdino_base_score"] = wrong_query_count[
            "stage_b_gdino_base_score"
        ][:, :899]
        with self.assertRaisesRegex(ValueError, "exactly 900"):
            ref_eval._diagnostic_external_gdino_base_identity_winner(
                wrong_query_count
            )

        wrong_rank = _external_outputs()
        wrong_rank["stage_b_gdino_base_score"] = wrong_rank[
            "stage_b_gdino_base_score"
        ].unsqueeze(-1)
        with self.assertRaisesRegex(ValueError, r"floating \(B,Q\)"):
            ref_eval._diagnostic_external_gdino_base_identity_winner(wrong_rank)

        nonfinite = _external_outputs()
        nonfinite["stage_b_gdino_base_score"] = nonfinite[
            "stage_b_gdino_base_score"
        ].clone()
        nonfinite["stage_b_gdino_base_score"][0, 17] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            ref_eval._diagnostic_external_gdino_base_identity_winner(nonfinite)

    def test_external_base_identity_row_coexists_and_uses_exact_caption(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = ["nearest_iou"]
        settings["iou_powers"] = []
        settings["patch_weights"] = [0.0]
        settings["text_weights"] = [1.0]
        settings["include_patch_internal_rank_identity"] = True
        settings["include_external_gdino_base_identity"] = True
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        patch_outputs = _patch_outputs()
        patch_outputs["stage_a_captions"] = [" Fire__TRUCK... "]
        external = _external_outputs()
        external["stage_b_gdino_base_score"] = torch.full((1, 900), 0.01)
        external["stage_b_gdino_base_score"][0, 17] = 1.0
        external["pred_boxes"] = external["pred_boxes"].clone()
        external["pred_boxes"][0, 17] = torch.tensor(
            [0.35, 0.45, 0.20, 0.10]
        )
        target = {"boxes": torch.tensor([[0.35, 0.45, 0.20, 0.10]])}
        accumulator = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            settings,
            patch_cfg=_patch_cfg(stage_b_v11_candidate_topk=2),
        )
        accumulator.update(patch_outputs, external, [target])
        rows = accumulator.results()
        self.assertEqual(len(rows), 3)
        base_rows = [
            row
            for row in rows
            if row.get("diagnostic_descriptor_kind")
            == ref_eval._EXTERNAL_GDINO_BASE_IDENTITY_KIND
        ]
        self.assertEqual(len(base_rows), 1)
        row = base_rows[0]
        self.assertEqual(
            row["diagnostic_identity_kind"], "external_gdino_base_direct"
        )
        self.assertEqual(
            row["diagnostic_score_key"], "stage_b_gdino_base_score"
        )
        self.assertEqual(row["diagnostic_query_count"], 900)
        self.assertEqual(
            row["diagnostic_output_box_source"],
            "external_outputs.pred_boxes_at_direct_global_argmax",
        )
        self.assertEqual(row["acc50"], 1.0)
        self.assertFalse(row["uses_patch_top50_admission"])
        self.assertFalse(row["uses_top_query_mapping"])
        self.assertFalse(row["formal_gate_eligible"])
        self.assertNotIn("diagnostic_transfer_mode", row)
        self.assertNotIn("diagnostic_patch_weight", row)
        self.assertEqual(
            row["by_canonical_stage_a_caption"]["fire truck"][
                "num_expressions"
            ],
            1,
        )

    def test_identity_addition_does_not_change_existing_transfer_rows(self):
        base_settings = _runtime_settings()
        base_settings["transfer_modes"] = ["nearest_iou"]
        base_settings["iou_powers"] = []
        base_settings["patch_weights"] = [0.0]
        base_settings["text_weights"] = [1.0]
        base_settings["fixed_grid"] = (
            ref_eval._diagnostic_external_rank_transfer_grid(base_settings)
        )
        identity_settings = dict(base_settings)
        identity_settings["include_patch_internal_rank_identity"] = True
        patch_cfg = _patch_cfg(stage_b_v11_candidate_topk=2)
        target = {"boxes": torch.tensor([[0.70, 0.70, 0.20, 0.20]])}

        baseline = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            base_settings
        )
        baseline.update(_patch_outputs(), _external_outputs(), [target])
        with_identity = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            identity_settings, patch_cfg=patch_cfg
        )
        with_identity.update(_patch_outputs(), _external_outputs(), [target])
        external_rows = [
            row
            for row in with_identity.results()
            if row.get("diagnostic_descriptor_kind") == "external_rank_transfer"
        ]
        self.assertEqual(external_rows, baseline.results())
        self.assertEqual(len(with_identity.results()), len(external_rows) + 1)

        base_identity_settings = dict(base_settings)
        base_identity_settings["include_external_gdino_base_identity"] = True
        with_base_identity = (
            ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
                base_identity_settings
            )
        )
        with_base_identity.update(
            _patch_outputs(), _external_outputs(), [target]
        )
        base_external_rows = [
            row
            for row in with_base_identity.results()
            if row.get("diagnostic_descriptor_kind") == "external_rank_transfer"
        ]
        self.assertEqual(base_external_rows, baseline.results())
        self.assertEqual(
            len(with_base_identity.results()), len(base_external_rows) + 1
        )

    def test_diagnostic_reports_stage_a_person_other_groups_without_target_routing(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = ["nearest_iou"]
        settings["iou_powers"] = []
        settings["patch_weights"] = [0.0]
        settings["text_weights"] = [1.0]
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        accumulator = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            settings
        )

        person_outputs = _patch_outputs()
        person_outputs["stage_a_captions"] = ["person ."]
        accumulator.update(
            person_outputs,
            _external_outputs(),
            [{"boxes": torch.tensor([[0.70, 0.70, 0.20, 0.20]])}],
        )
        other_outputs = _patch_outputs()
        other_outputs["stage_a_captions"] = ["car ."]
        accumulator.update(
            other_outputs,
            _external_outputs(),
            [{"boxes": torch.tensor([[0.20, 0.20, 0.10, 0.10]])}],
        )

        row = accumulator.results()[0]
        groups = row["by_canonical_stage_a_group"]
        self.assertEqual(groups["person"]["num_expressions"], 1)
        self.assertEqual(groups["person"]["acc50"], 1.0)
        self.assertEqual(groups["other"]["num_expressions"], 1)
        self.assertEqual(groups["other"]["acc50"], 0.0)
        self.assertFalse(
            row["canonical_stage_a_group_contract"][
                "uses_target_category_or_box_for_routing"
            ]
        )

    def test_exact_normalized_caption_breakdown_is_on_every_descriptor(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = ["nearest_iou"]
        settings["iou_powers"] = []
        settings["patch_weights"] = [0.0]
        settings["text_weights"] = [1.0]
        settings["include_patch_internal_rank_identity"] = True
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        accumulator = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
            settings,
            patch_cfg=_patch_cfg(stage_b_v11_candidate_topk=2),
        )
        examples = (
            (" Person.__ ", [0.20, 0.20, 0.10, 0.10]),
            ("Fire__Truck...", [0.70, 0.70, 0.20, 0.20]),
            ("  FIRE truck . ", [0.20, 0.20, 0.10, 0.10]),
        )
        for caption, box in examples:
            patch_outputs = _patch_outputs()
            patch_outputs["stage_a_captions"] = [caption]
            accumulator.update(
                patch_outputs,
                _external_outputs(),
                [{"boxes": torch.tensor([box])}],
            )

        rows = accumulator.results()
        self.assertEqual(len(rows), 2)
        for row in rows:
            captions = row["by_canonical_stage_a_caption"]
            self.assertEqual(set(captions), {"person", "fire truck"})
            self.assertEqual(captions["person"]["num_expressions"], 1)
            self.assertEqual(captions["fire truck"]["num_expressions"], 2)
            self.assertEqual(
                row["canonical_stage_a_caption_contract"]["key"],
                "exact_normalized_caption",
            )
            self.assertEqual(
                row["by_canonical_stage_a_group"]["person"][
                    "num_expressions"
                ],
                1,
            )
            self.assertEqual(
                row["by_canonical_stage_a_group"]["other"]["num_expressions"],
                2,
            )

    def test_missing_caption_omits_breakdowns_and_malformed_caption_fails(self):
        settings = _runtime_settings()
        settings["transfer_modes"] = ["nearest_iou"]
        settings["iou_powers"] = []
        settings["patch_weights"] = [0.0]
        settings["text_weights"] = [1.0]
        settings["fixed_grid"] = ref_eval._diagnostic_external_rank_transfer_grid(
            settings
        )
        target = {"boxes": torch.tensor([[0.20, 0.20, 0.10, 0.10]])}

        missing = ref_eval.DiagnosticExternalGDINORankTransferAccumulator(settings)
        missing.update(_patch_outputs(), _external_outputs(), [target])
        row = missing.results()[0]
        self.assertNotIn("by_canonical_stage_a_group", row)
        self.assertNotIn("by_canonical_stage_a_caption", row)

        malformed_values = (123, [], [""], ["..."])
        for malformed in malformed_values:
            with self.subTest(malformed=malformed):
                outputs = _patch_outputs()
                outputs["stage_a_captions"] = malformed
                accumulator = (
                    ref_eval.DiagnosticExternalGDINORankTransferAccumulator(
                        settings
                    )
                )
                with self.assertRaisesRegex(ValueError, "stage_a_captions"):
                    accumulator.update(outputs, _external_outputs(), [target])

    def test_missing_rank_key_bad_900_shape_and_unnormalized_boxes_fail_closed(self):
        no_rank = _external_outputs()
        no_rank.pop("stage_b_gdino_rank_score")
        with self.assertRaisesRegex(KeyError, "stage_b_gdino_rank_score"):
            ref_eval._diagnostic_external_rank_candidate_scores(
                _patch_outputs(), no_rank, settings=_runtime_settings()
            )

        wrong_queries = _external_outputs()
        wrong_queries["pred_boxes"] = wrong_queries["pred_boxes"][:, :899]
        wrong_queries["stage_b_gdino_rank_score"] = wrong_queries[
            "stage_b_gdino_rank_score"
        ][:, :899]
        with self.assertRaisesRegex(ValueError, "exactly"):
            ref_eval._diagnostic_external_rank_candidate_scores(
                _patch_outputs(), wrong_queries, settings=_runtime_settings()
            )

        bad_boxes = _external_outputs()
        bad_boxes["pred_boxes"] = bad_boxes["pred_boxes"].clone()
        bad_boxes["pred_boxes"][0, 0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "normalized"):
            ref_eval._diagnostic_external_rank_candidate_scores(
                _patch_outputs(), bad_boxes, settings=_runtime_settings()
            )

    def test_external_forward_routes_full_caption_and_requires_adapter_rank_key(self):
        samples = NestedTensor(
            torch.ones(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool)
        )
        batch = (
            samples,
            [{"caption": "the red car .", "boxes": torch.ones(1, 4) * 0.2}],
        )
        model = _AdapterModel()
        outputs, targets = ref_eval._forward_external_gdino_rank_adapter(
            model,
            batch,
            torch.device("cpu"),
            amp=False,
            cfg=_external_cfg(),
        )
        self.assertEqual(model.calls, [["the red car ."]])
        self.assertIn("stage_b_gdino_rank_score", outputs)
        self.assertEqual(tuple(targets[0]["boxes"].shape), (1, 4))

        with self.assertRaisesRegex(KeyError, "stage_b_gdino_rank_score"):
            ref_eval._forward_external_gdino_rank_adapter(
                _AdapterModel(include_rank=False),
                batch,
                torch.device("cpu"),
                amp=False,
                cfg=_external_cfg(),
            )

    def test_settings_reject_reused_components_and_bind_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch_config = root / "patch.py"
            external_config = root / "external.py"
            patch_checkpoint = root / "patch.pth"
            external_checkpoint = root / "external.pth"
            patch_config.write_text("patch = True\n", encoding="utf-8")
            external_config.write_text("adapter = True\n", encoding="utf-8")
            patch_checkpoint.write_bytes(b"patch checkpoint")
            external_checkpoint.write_bytes(b"external checkpoint")

            kwargs = {
                "external_config_path": str(external_config),
                "external_checkpoint_path": str(external_checkpoint),
                "transfer_modes": ["nearest_iou", "max_score_iou_power"],
                "iou_powers": [0.5, 1.0],
                "patch_weights": [1.0],
                "text_weights": [0.0, 1.0],
                "patch_cfg": _patch_cfg(),
                "external_cfg": _external_cfg(),
                "patch_config_path": str(patch_config),
                "patch_checkpoint_paths": [str(patch_checkpoint)],
            }
            settings = ref_eval._diagnostic_external_rank_transfer_settings(
                **kwargs
            )
            self.assertEqual(
                settings["external_config"]["sha256"],
                hashlib.sha256(external_config.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                settings["external_checkpoint"]["sha256"],
                hashlib.sha256(external_checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(settings["fixed_grid"]), 6)
            self.assertIn("signed_score_policy", settings["transfer_contract"])

            identity_settings = (
                ref_eval._diagnostic_external_rank_transfer_settings(
                    **kwargs,
                    include_patch_internal_rank_identity=True,
                    include_external_gdino_base_identity=True,
                )
            )
            self.assertTrue(
                identity_settings["include_patch_internal_rank_identity"]
            )
            identity_contract = identity_settings[
                "patch_internal_rank_identity_contract"
            ]
            self.assertEqual(
                identity_contract["descriptor_kind"],
                ref_eval._PATCH_INTERNAL_RANK_IDENTITY_KIND,
            )
            self.assertTrue(identity_contract["diagnostic_only"])
            self.assertFalse(identity_contract["formal_gate_eligible"])
            self.assertFalse(identity_contract["uses_external_rank_score"])
            self.assertTrue(
                identity_settings["include_external_gdino_base_identity"]
            )
            base_identity_contract = identity_settings[
                "external_gdino_base_identity_contract"
            ]
            self.assertEqual(
                base_identity_contract["identity_kind"],
                "external_gdino_base_direct",
            )
            self.assertEqual(
                base_identity_contract["score_source"],
                "stage_b_gdino_base_score",
            )
            self.assertFalse(
                base_identity_contract["uses_patch_top50_admission"]
            )
            self.assertEqual(
                identity_settings["fixed_grid"], settings["fixed_grid"]
            )

            reused_config = dict(kwargs)
            reused_config["external_config_path"] = str(patch_config)
            with self.assertRaisesRegex(ValueError, "configs must be independent"):
                ref_eval._diagnostic_external_rank_transfer_settings(
                    **reused_config
                )

            reused_checkpoint = dict(kwargs)
            reused_checkpoint["external_checkpoint_path"] = str(patch_checkpoint)
            with self.assertRaisesRegex(ValueError, "checkpoints must be independent"):
                ref_eval._diagnostic_external_rank_transfer_settings(
                    **reused_checkpoint
                )

            malformed = dict(kwargs)
            malformed["external_cfg"] = _external_cfg(num_queries=899)
            with self.assertRaisesRegex(ValueError, "num_queries=900"):
                ref_eval._diagnostic_external_rank_transfer_settings(**malformed)

    def test_summary_is_diagnostic_and_contains_external_contract_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / name
                for name in ("patch.py", "external.py", "patch.pth", "external.pth")
            }
            for index, path in enumerate(paths.values()):
                path.write_bytes(f"component-{index}".encode())
            settings = ref_eval._diagnostic_external_rank_transfer_settings(
                external_config_path=str(paths["external.py"]),
                external_checkpoint_path=str(paths["external.pth"]),
                transfer_modes=["nearest_iou"],
                iou_powers=None,
                patch_weights=[1.0],
                text_weights=[0.0, 1.0],
                patch_cfg=_patch_cfg(),
                external_cfg=_external_cfg(),
                patch_config_path=str(paths["patch.py"]),
                patch_checkpoint_paths=[str(paths["patch.pth"])],
                include_patch_internal_rank_identity=True,
                include_external_gdino_base_identity=True,
            )
            rows = []
            for descriptor in settings["fixed_grid"]:
                rows.append(
                    {
                        "diagnostic_only": True,
                        "formal_gate_eligible": False,
                        "diagnostic_transfer_mode": descriptor["transfer_mode"],
                        "diagnostic_iou_power": descriptor["iou_power"],
                        "diagnostic_patch_weight": descriptor["patch_weight"],
                        "diagnostic_text_weight": descriptor["text_weight"],
                        "candidate_topk": 50,
                        "num_expressions": 8,
                        "acc50": 0.5,
                        "mean_iou_top1": 0.4,
                        "candidate_oracle_recall50": 0.9,
                        "run_id": str(descriptor),
                        "dataset": "refcoco_val",
                        "seed": 42,
                        "checkpoint": str(paths["patch.pth"]),
                        "patch_checkpoint_sha256": settings["patch_checkpoints"][0]["sha256"],
                        "patch_config_sha256": settings["patch_config"]["sha256"],
                        "external_gdino_checkpoint_sha256": settings["external_checkpoint"]["sha256"],
                        "external_gdino_config_sha256": settings["external_config"]["sha256"],
                        "external_gdino_rank_score_key": "stage_b_gdino_rank_score",
                        "external_gdino_query_count": 900,
                        "transfer_contract_version": 1,
                    }
                )
            rows.append(
                {
                    "diagnostic_only": True,
                    "formal_gate_eligible": False,
                    "diagnostic_descriptor_kind": (
                        ref_eval._PATCH_INTERNAL_RANK_IDENTITY_KIND
                    ),
                    "diagnostic_standard_ref_beta": 0.0,
                    "uses_external_rank_score": False,
                    "uses_external_box": False,
                    "uses_fusion_weights": False,
                    "patch_internal_rank_identity_contract_version": 1,
                    "candidate_topk": 50,
                    "num_expressions": 8,
                    "acc50": 0.5,
                    "mean_iou_top1": 0.4,
                    "candidate_oracle_recall50": 0.9,
                    "run_id": "patch:diagnostic_patch_internal_rank_identity=standard_ref_beta0",
                    "dataset": "refcoco_val",
                    "seed": 42,
                    "checkpoint": str(paths["patch.pth"]),
                    "patch_checkpoint_sha256": settings["patch_checkpoints"][0][
                        "sha256"
                    ],
                    "patch_config_sha256": settings["patch_config"]["sha256"],
                    "external_gdino_checkpoint_sha256": settings[
                        "external_checkpoint"
                    ]["sha256"],
                    "external_gdino_config_sha256": settings["external_config"][
                        "sha256"
                    ],
                    "external_gdino_rank_score_key": "stage_b_gdino_rank_score",
                    "external_gdino_query_count": 900,
                    "transfer_contract_version": 1,
                }
            )
            rows.append(
                {
                    "diagnostic_only": True,
                    "formal_gate_eligible": False,
                    "diagnostic_descriptor_kind": (
                        ref_eval._EXTERNAL_GDINO_BASE_IDENTITY_KIND
                    ),
                    "diagnostic_identity_kind": "external_gdino_base_direct",
                    "diagnostic_score_key": "stage_b_gdino_base_score",
                    "diagnostic_query_count": 900,
                    "diagnostic_output_box_source": (
                        "external_outputs.pred_boxes_at_direct_global_argmax"
                    ),
                    "diagnostic_standard_ref_beta": 0.0,
                    "diagnostic_score_source": "stage_b_gdino_base_score",
                    "diagnostic_winner_rule": (
                        "first_argmax_over_full_external_query_axis"
                    ),
                    "diagnostic_query_domain": (
                        "all_900_external_gdino_queries"
                    ),
                    "uses_external_base_score": True,
                    "uses_external_rank_score": False,
                    "uses_external_box": True,
                    "uses_adapter_rank_residual": False,
                    "uses_patch_top50_admission": False,
                    "uses_top_query_mapping": False,
                    "uses_fusion_weights": False,
                    "external_gdino_base_identity_contract_version": 1,
                    "external_query_count": 900,
                    "candidate_topk": 50,
                    "candidate_topk_scope": (
                        "run_context_only_not_used_by_this_descriptor"
                    ),
                    "num_expressions": 8,
                    "acc50": 0.5,
                    "mean_iou_top1": 0.4,
                    "candidate_oracle_recall50": 0.9,
                    "run_id": (
                        "patch:diagnostic_external_gdino_base_identity="
                        "direct_global_argmax"
                    ),
                    "dataset": "refcoco_val",
                    "seed": 42,
                    "checkpoint": str(paths["patch.pth"]),
                    "patch_checkpoint_sha256": settings["patch_checkpoints"][0][
                        "sha256"
                    ],
                    "patch_config_sha256": settings["patch_config"]["sha256"],
                    "external_gdino_checkpoint_sha256": settings[
                        "external_checkpoint"
                    ]["sha256"],
                    "external_gdino_config_sha256": settings["external_config"][
                        "sha256"
                    ],
                    "external_gdino_rank_score_key": "stage_b_gdino_rank_score",
                    "external_gdino_query_count": 900,
                    "transfer_contract_version": 1,
                }
            )
            metadata = (
                ref_eval._diagnostic_external_rank_transfer_summary_metadata(
                    settings, rows
                )
            )
            self.assertIs(metadata["diagnostic_only"], True)
            self.assertIs(metadata["formal_gate_eligible"], False)
            self.assertEqual(
                metadata["external_gdino_checkpoint"]["sha256"],
                settings["external_checkpoint"]["sha256"],
            )
            self.assertTrue(metadata["include_patch_internal_rank_identity"])
            self.assertTrue(metadata["include_external_gdino_base_identity"])
            ref_eval._write_summary(root, rows, "acc50", diagnostic_metadata=metadata)
            payload = json.loads((root / "summary.json").read_text())
            self.assertEqual(payload["diagnostic_kind"], "external_gdino_rank_transfer")
            self.assertFalse(payload["formal_gate_eligible"])
            self.assertFalse(list(root.glob("*.records.jsonl")))

    def test_main_final_write_preserves_external_diagnostic_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            paths = {
                name: root / name
                for name in (
                    "patch.py",
                    "external.py",
                    "patch.pth",
                    "external.pth",
                )
            }
            for index, path in enumerate(paths.values()):
                path.write_bytes(f"main-component-{index}".encode())

            patch_cfg = _patch_cfg()
            external_cfg = _external_cfg()
            settings = ref_eval._diagnostic_external_rank_transfer_settings(
                external_config_path=str(paths["external.py"]),
                external_checkpoint_path=str(paths["external.pth"]),
                transfer_modes=["nearest_iou"],
                iou_powers=None,
                patch_weights=[1.0],
                text_weights=[1.0],
                patch_cfg=patch_cfg,
                external_cfg=external_cfg,
                patch_config_path=str(paths["patch.py"]),
                patch_checkpoint_paths=[str(paths["patch.pth"])],
            )
            descriptor = settings["fixed_grid"][0]
            row = {
                "diagnostic_only": True,
                "formal_gate_eligible": False,
                "diagnostic_transfer_mode": descriptor["transfer_mode"],
                "diagnostic_iou_power": descriptor["iou_power"],
                "diagnostic_patch_weight": descriptor["patch_weight"],
                "diagnostic_text_weight": descriptor["text_weight"],
                "candidate_topk": 50,
                "num_expressions": 1,
                "acc50": 1.0,
                "acc25": 1.0,
                "mean_iou_top1": 0.75,
                "recall50@1": 1.0,
                "candidate_oracle_recall50": 1.0,
                "run_id": "patch:diagnostic_external_gdino_rank_transfer=nearest_iou,p=none,patch=1,text=1",
                "dataset": "refcoco_val",
                "seed": 42,
                "checkpoint": str(paths["patch.pth"]),
                "patch_checkpoint_sha256": settings["patch_checkpoints"][0][
                    "sha256"
                ],
                "patch_config_sha256": settings["patch_config"]["sha256"],
                "external_gdino_checkpoint_sha256": settings[
                    "external_checkpoint"
                ]["sha256"],
                "external_gdino_config_sha256": settings["external_config"][
                    "sha256"
                ],
                "external_gdino_rank_score_key": "stage_b_gdino_rank_score",
                "external_gdino_query_count": 900,
                "transfer_contract_version": 1,
            }
            eval_jsonl = root / "refcoco_val.jsonl"
            argv = [
                "eval_refcoco_stageb.py",
                "--config",
                str(paths["patch.py"]),
                "--ckpts",
                str(paths["patch.pth"]),
                "--output_dir",
                str(output_dir),
                "--data_root",
                str(root),
                "--device",
                "cpu",
                "--splits",
                "refcoco_val",
                "--diagnostic_external_gdino_config",
                str(paths["external.py"]),
                "--diagnostic_external_gdino_checkpoint",
                str(paths["external.pth"]),
                "--diagnostic_external_transfer_modes",
                "nearest_iou",
                "--diagnostic_external_patch_weights",
                "1",
                "--diagnostic_external_text_weights",
                "1",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ref_eval.SLConfig,
                    "fromfile",
                    side_effect=[patch_cfg, external_cfg],
                ),
                mock.patch.object(
                    ref_eval, "_load_canonical_name_maps", return_value=({}, {})
                ),
                mock.patch.object(ref_eval, "_load_phrase_maps", return_value={}),
                mock.patch.object(
                    ref_eval, "load_holdout_keys", return_value=(set(), set())
                ),
                mock.patch.object(
                    ref_eval,
                    "_build_split_jsonl",
                    return_value=(eval_jsonl, 1),
                ),
                mock.patch.object(
                    ref_eval,
                    "_make_datasetinfo",
                    return_value={"anno": str(eval_jsonl)},
                ),
                mock.patch.object(ref_eval, "_load_model", return_value=object()),
                mock.patch.object(
                    ref_eval, "evaluate_dataset", return_value=[dict(row)]
                ),
            ):
                ref_eval.main()

            payload = json.loads((output_dir / "summary.json").read_text())
            self.assertIs(payload["diagnostic_only"], True)
            self.assertIs(payload["formal_gate_eligible"], False)
            self.assertEqual(
                payload["diagnostic_kind"], "external_gdino_rank_transfer"
            )
            self.assertEqual(
                payload["external_gdino_checkpoint"]["sha256"],
                settings["external_checkpoint"]["sha256"],
            )
            self.assertIn(
                "formal_gate_eligible=false",
                (output_dir / "summary.md").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
