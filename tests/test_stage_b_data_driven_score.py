import types
import unittest
import re

import torch

from engine import (
    _build_stage_b_data_driven_assignment_captions,
    _build_stage_b_data_driven_pair_captions,
    _build_stage_b_data_driven_positive_captions,
)
from models.GroundingDINO.stage_b_data_driven_score import (
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE,
    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
    StageBDataDrivenCriterion,
    StageBDataDrivenScoreHeads,
    build_direct_trace_token_roles,
    data_driven_category_gate_mask,
    official_assignment_delta_loss,
)


class _LexicalFastTokenizer:
    """Minimal deterministic fast-tokenizer contract for offset unit tests."""

    def __init__(self):
        self._ids = {}

    def __call__(
        self,
        captions,
        *,
        padding,
        return_tensors,
        return_offsets_mapping=False,
    ):
        rows = []
        offsets = []
        for caption in captions:
            tokens = list(re.finditer(r"[A-Za-z0-9]+", caption))
            ids = [101]
            spans = [(0, 0)]
            for token in tokens:
                norm = token.group(0).lower()
                if norm not in self._ids:
                    self._ids[norm] = 1000 + len(self._ids)
                ids.append(self._ids[norm])
                spans.append((token.start(), token.end()))
            ids.append(102)
            spans.append((0, 0))
            rows.append(ids)
            offsets.append(spans)
        width = max(map(len, rows))
        for ids, spans in zip(rows, offsets):
            pad = width - len(ids)
            ids.extend([0] * pad)
            spans.extend([(0, 0)] * pad)
        result = {"input_ids": torch.tensor(rows, dtype=torch.int64)}
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(offsets, dtype=torch.int64)
        return result


def _pair_target(positive="a pink box .", negative="a blue box ."):
    return {
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "global_tn_verified": torch.tensor([True]),
        "tn_scope": "image_global_topk_verified",
        "stage_b_data_driven_expression_captions": [positive, negative],
        "stage_b_data_driven_trace": {
            "category": "color",
            "replace_from": "pink",
            "replace_to": "blue",
            "replace_span": [1, 2],
        },
    }


def _tokenized_pair(tokenizer, pair):
    encoded = tokenizer(
        pair,
        padding="longest",
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"].reshape(1, 2, -1)
    mask = (ids != 0) & (ids != 101) & (ids != 102)
    return ids, mask


class StageBDataDrivenScoreTest(unittest.TestCase):
    def _heads(self, *, gate=False, gap=0.0):
        torch.manual_seed(7)
        return StageBDataDrivenScoreHeads(
            hidden_dim=8,
            rank_dim=4,
            confidence_dim=5,
            gate_hidden_dim=6,
            category_gate=gate,
            category_gate_max_gap=gap,
        )

    def test_rank_confidence_parameters_are_disjoint_and_base_free(self):
        heads = self._heads()
        rank_ids = {id(parameter) for parameter in heads.rank_parameters()}
        confidence_ids = {
            id(parameter) for parameter in heads.confidence_parameters()
        }
        self.assertTrue(rank_ids)
        self.assertTrue(confidence_ids)
        self.assertFalse(rank_ids & confidence_ids)
        names = {name for name, _parameter in heads.named_parameters()}
        self.assertFalse(any("base" in name or "teacher" in name for name in names))

    def test_absolute_heads_return_independent_token_scores(self):
        heads = self._heads().eval()
        query = torch.randn(2, 3, 8)
        text = torch.randn(2, 4, 8)
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        before = heads(query, text, mask)
        with torch.no_grad():
            heads.rank_branch.query_proj.weight.add_(0.5)
        after = heads(query, text, mask)
        self.assertFalse(
            torch.equal(before["text_rank_token_logits"], after["text_rank_token_logits"])
        )
        self.assertTrue(
            torch.equal(
                before["confidence_token_logits"],
                after["confidence_token_logits"],
            )
        )
        self.assertEqual(tuple(after["rank_score"].shape), (2, 3))
        self.assertEqual(tuple(after["confidence_score"].shape), (2, 3))

    def test_category_gate_preserves_text_order_inside_patch_set(self):
        heads = self._heads(gate=True, gap=0.0).eval()
        query = torch.randn(1, 3, 8)
        text = torch.randn(1, 2, 8)
        mask = torch.ones(1, 2, dtype=torch.bool)
        patch = torch.tensor([[0.0, 5.0, 1.0]])
        result = heads(query, text, mask, patch_score=patch)
        eligible = result["category_gate_eligible_mask"]
        self.assertEqual(eligible.tolist(), [[False, True, False]])
        self.assertEqual(
            result["rank_score"].argmax(dim=1).tolist(), [1]
        )

    def test_positive_caption_builder_separates_canonical_and_expression(self):
        target = {
            "cap_list": ["small red truck"],
            "stage_a_caption": "truck .",
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "is_negative_episode": torch.tensor([0]),
            "is_lvis_neg_category_episode": torch.tensor([0]),
        }
        canonical, expression = _build_stage_b_data_driven_positive_captions(
            [target]
        )
        self.assertEqual(canonical, ["truck ."])
        self.assertEqual(expression, ["small red truck ."])

    def test_official_assignment_caption_builder_preserves_two_referents(self):
        target = {
            "cap_list": ["small red truck"],
            "stage_a_caption": "truck .",
            "stage_b_data_driven_assignment_pair_schema": (
                "pivot.stageb.data_driven.official_assignment_pair/v1"
            ),
            "stage_b_data_driven_assignment_expressions": [
                "small red truck",
                "large blue truck",
            ],
            "stage_b_data_driven_assignment_valid": torch.tensor([True]),
            "stage_b_data_driven_assignment_role": torch.tensor([0, 1]),
            "boxes": torch.tensor(
                [[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]]
            ),
            "is_negative_episode": torch.tensor([0]),
            "is_lvis_neg_category_episode": torch.tensor([0]),
        }
        canonical, expressions = (
            _build_stage_b_data_driven_assignment_captions([target])
        )
        self.assertEqual(canonical, ["truck ."])
        self.assertEqual(
            expressions, [["small red truck .", "large blue truck ."]]
        )
        target["stage_b_data_driven_assignment_role"] = torch.tensor([0, 0])
        with self.assertRaisesRegex(ValueError, "two exact targets"):
            _build_stage_b_data_driven_assignment_captions([target])

    def test_confidence_caption_builder_requires_image_global_pair(self):
        target = {
            "cap_list": ["small red truck", "small blue truck"],
            "stage_a_caption": "truck .",
            "stage_b_data_driven_trace": {
                "category": "color",
                "replace_from": "red",
                "replace_to": "blue",
                "replace_span": [1, 2],
            },
            "tn_scope": "image_global_topk_verified",
            "global_tn_verified": torch.tensor([True]),
            "verifier_pair_stride": torch.tensor([2]),
            "is_negative_episode": torch.tensor([0]),
            "is_lvis_neg_category_episode": torch.tensor([0]),
        }
        canonical, expressions = _build_stage_b_data_driven_pair_captions(
            [target]
        )
        self.assertEqual(canonical, ["truck ."])
        self.assertEqual(
            expressions, [["small red truck .", "small blue truck ."]]
        )
        target["tn_scope"] = "proposal_covered_verified"
        target["global_tn_verified"] = torch.tensor([False])
        with self.assertRaisesRegex(ValueError, "image-global"):
            _build_stage_b_data_driven_pair_captions([target])

    def test_direct_trace_roles_reconstruct_and_recover_unique_stale_span(self):
        tokenizer = _LexicalFastTokenizer()
        pair = ["a pink box .", "a blue box ."]
        ids, mask = _tokenized_pair(tokenizer, pair)
        trace = {
            "category": "color",
            "replace_from": "pink",
            "replace_to": "blue",
            "replace_span": [1, 9],
        }
        roles = build_direct_trace_token_roles(
            tokenizer, [pair], [trace], ids, mask, max_text_len=256
        )
        self.assertEqual(roles["valid"].tolist(), [True])
        self.assertEqual(int(roles["changed"].sum().item()), 1)
        changed_index = int(torch.nonzero(roles["changed"][0, 1])[0].item())
        self.assertEqual(int(ids[0, 1, changed_index].item()), tokenizer._ids["blue"])

    def test_direct_trace_roles_skip_extra_edits_and_deletion_only(self):
        tokenizer = _LexicalFastTokenizer()
        pairs = [
            ["a pink box .", "an blue box ."],
            ["a red striped box .", "a red box ."],
        ]
        flat = [caption for pair in pairs for caption in pair]
        encoded = tokenizer(
            flat,
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"].reshape(2, 2, -1)
        mask = (ids != 0) & (ids != 101) & (ids != 102)
        traces = [
            {
                "category": "color",
                "replace_from": "pink",
                "replace_to": "blue",
                "replace_span": [1, 2],
            },
            {
                "category": "pattern",
                "replace_from": "red striped",
                "replace_to": "red",
                "replace_span": [1, 3],
            },
        ]
        roles = build_direct_trace_token_roles(
            tokenizer, pairs, traces, ids, mask, max_text_len=256
        )
        self.assertEqual(roles["valid"].tolist(), [False, False])

    def test_direct_trace_roles_ignore_incidental_edits_when_enabled(self):
        tokenizer = _LexicalFastTokenizer()
        pairs = [
            ["a pink box .", "an blue box ."],
            ["a red striped box .", "a red box ."],
        ]
        encoded = tokenizer(
            [caption for pair in pairs for caption in pair],
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"].reshape(2, 2, -1)
        mask = (ids != 0) & (ids != 101) & (ids != 102)
        roles = build_direct_trace_token_roles(
            tokenizer,
            pairs,
            [
                {
                    "category": "color",
                    "replace_from": "pink",
                    "replace_to": "blue",
                    "replace_span": [1, 2],
                },
                {
                    "category": "pattern",
                    "replace_from": "red striped",
                    "replace_to": "red",
                    "replace_span": [1, 3],
                },
            ],
            ids,
            mask,
            max_text_len=256,
            allow_incidental_edits=True,
        )
        self.assertEqual(roles["valid"].tolist(), [True, False])
        changed_ids = ids[0, 1][roles["changed"][0, 1]].tolist()
        shared_ids = ids[0, 1][roles["shared"][0, 1]].tolist()
        self.assertEqual(changed_ids, [tokenizer._ids["blue"]])
        self.assertEqual(shared_ids, [tokenizer._ids["box"]])
        self.assertNotIn(tokenizer._ids["an"], shared_ids)

    def test_direct_rank_patch_loss_has_no_teacher_input(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=False,
        )
        rank = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        patch = torch.tensor([[[0.0], [1.0], [-1.0]]], requires_grad=True)
        outputs = {
            "stage_b_data_driven_text_rank_score": rank,
            "pred_logits_patch": patch,
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]]]
            ),
        }
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([4]),
                "primary_instance_mask": torch.tensor([True]),
                "stage_b_u2_category_complete": torch.tensor([False]),
            }
        ]
        losses = criterion(outputs, targets)
        total = sum(
            losses[key] * weight for key, weight in criterion.weight_dict.items()
        )
        total.backward()
        self.assertIsNotNone(rank.grad)
        self.assertIsNotNone(patch.grad)
        self.assertTrue(torch.isfinite(rank.grad).all())
        self.assertTrue(torch.isfinite(patch.grad).all())

    def test_dd1_requires_category_complete_marker(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
        )
        outputs = {
            "stage_b_data_driven_text_rank_score": torch.zeros(1, 2),
            "pred_logits_patch": torch.zeros(1, 2, 1),
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]
            ),
        }
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([4]),
                "primary_instance_mask": torch.tensor([True]),
            }
        ]
        with self.assertRaisesRegex(ValueError, "target variant"):
            criterion(outputs, targets)

    @staticmethod
    def _same_category_target(*, auxiliary=True, overlap=False):
        primary_box = [0.25, 0.5, 0.2, 0.2]
        boxes = [primary_box]
        if auxiliary:
            boxes.append(primary_box if overlap else [0.75, 0.5, 0.2, 0.2])
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.full((len(boxes),), 4, dtype=torch.int64),
            "primary_instance_mask": torch.tensor(
                [True] + [False] * (len(boxes) - 1)
            ),
            "stage_b_u2_category_complete": torch.tensor([True]),
        }

    @staticmethod
    def _same_category_outputs(rank, patch=None, boxes=None):
        if boxes is None:
            boxes = [
                [0.25, 0.5, 0.2, 0.2],
                [0.75, 0.5, 0.2, 0.2],
                [0.5, 0.1, 0.1, 0.1],
            ]
        if patch is None:
            patch = torch.zeros(1, len(boxes), 1, requires_grad=True)
        return {
            "stage_b_data_driven_text_rank_score": rank,
            "stage_b_data_driven_candidate_mask": torch.ones(
                1, len(boxes), dtype=torch.bool
            ),
            "pred_logits_patch": patch,
            "pred_boxes": torch.tensor([boxes], dtype=torch.float32),
        }

    @staticmethod
    def _assignment_target(*, valid=True):
        return {
            "boxes": torch.tensor(
                [[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([4, 4], dtype=torch.int64),
            "primary_instance_mask": torch.tensor([True, False]),
            "stage_b_u2_category_complete": torch.tensor([True]),
            "stage_b_data_driven_assignment_valid": torch.tensor(
                [valid], dtype=torch.bool
            ),
            "stage_b_data_driven_assignment_role": torch.tensor(
                [0, 1 if valid else -1], dtype=torch.int64
            ),
        }

    @staticmethod
    def _assignment_outputs(rank, patch=None):
        if patch is None:
            patch = torch.zeros(1, 3, 1, requires_grad=True)
        return {
            "stage_b_data_driven_text_rank_score": rank,
            "stage_b_data_driven_candidate_mask": torch.ones(
                1, 3, 2, dtype=torch.bool
            ),
            "pred_logits_patch": patch,
            "pred_boxes": torch.tensor(
                [
                    [
                        [0.25, 0.5, 0.2, 0.2],
                        [0.75, 0.5, 0.2, 0.2],
                        [0.5, 0.1, 0.1, 0.1],
                    ]
                ],
                dtype=torch.float32,
            ),
        }

    def test_same_category_rank_roles_have_exact_gradient_ownership(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
            temperature=1.0,
        )
        rank = torch.tensor([[0.0, 0.0, 100.0]], requires_grad=True)
        result = criterion(
            self._same_category_outputs(rank),
            [self._same_category_target()],
        )
        self.assertAlmostEqual(
            float(result["loss_stage_b_data_driven_rank"].detach()),
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )
        self.assertEqual(
            float(result["stage_b_data_driven_rank_positive_queries"]), 1.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_rank_hard_negative_queries"]),
            1.0,
        )
        self.assertEqual(
            float(result["stage_b_data_driven_rank_ignored_queries"]), 1.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_rank_listwise_rows"]), 1.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_rank_ignored_winner_rows"]), 1.0
        )
        result["loss_stage_b_data_driven_rank"].backward()
        self.assertLess(float(rank.grad[0, 0]), 0.0)
        self.assertGreater(float(rank.grad[0, 1]), 0.0)
        self.assertEqual(float(rank.grad[0, 2]), 0.0)

    def test_gap3_coverage_adds_only_inference_eligible_competitors(self):
        target = self._same_category_target()
        boxes = [
            [0.25, 0.5, 0.2, 0.2],
            [0.75, 0.5, 0.2, 0.2],
            [0.5, 0.1, 0.1, 0.1],
            [0.25, 0.5, 0.13, 0.13],
        ]
        h_rank = torch.tensor(
            [[0.0, 0.0, 100.0, 90.0]], requires_grad=True
        )
        hc_rank = h_rank.detach().clone().requires_grad_(True)
        h_patch = torch.zeros(1, 4, 1, requires_grad=True)
        hc_patch = h_patch.detach().clone().requires_grad_(True)
        h = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
            temperature=1.0,
        )
        hc = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=(
                DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE
            ),
            temperature=1.0,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
        )
        h_result = h(
            self._same_category_outputs(h_rank, h_patch, boxes), [target]
        )
        hc_result = hc(
            self._same_category_outputs(hc_rank, hc_patch, boxes), [target]
        )

        self.assertEqual(
            int(hc.rank_supervision_contract_id.item()), 3
        )
        self.assertEqual(
            float(hc_result["stage_b_data_driven_rank_hard_negative_queries"]),
            1.0,
        )
        self.assertEqual(
            float(
                hc_result[
                    "stage_b_data_driven_rank_coverage_negative_queries"
                ]
            ),
            2.0,
        )
        self.assertEqual(
            float(hc_result["stage_b_data_driven_rank_total_negative_queries"]),
            3.0,
        )
        self.assertEqual(
            float(hc_result["stage_b_data_driven_rank_gap3_eligible_queries"]),
            4.0,
        )
        self.assertEqual(
            float(
                hc_result[
                    "stage_b_data_driven_rank_gap3_ambiguous_coverage_queries"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(
                hc_result[
                    "stage_b_data_driven_rank_gap3_negative_coverage_queries"
                ]
            ),
            1.0,
        )

        hc_result["loss_stage_b_data_driven_rank"].backward()
        self.assertLess(float(hc_rank.grad[0, 0]), 0.0)
        self.assertGreater(float(hc_rank.grad[0, 1]), 0.0)
        self.assertGreater(float(hc_rank.grad[0, 2]), 0.0)
        self.assertGreater(float(hc_rank.grad[0, 3]), 0.0)
        self.assertIsNone(hc_patch.grad)

        h_result["loss_stage_b_data_driven_patch"].backward()
        hc_result["loss_stage_b_data_driven_patch"].backward()
        self.assertTrue(torch.equal(h_patch.grad, hc_patch.grad))

    def test_official_assignment_separates_two_directions(self):
        score = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [7.0, 7.0]]],
            requires_grad=True,
        )
        iou = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]
        )
        candidate = torch.ones(1, 3, 2, dtype=torch.bool)
        patch = torch.zeros(1, 3, 1, requires_grad=True)
        kwargs = dict(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            temperature=0.1,
        )
        base = official_assignment_delta_loss(
            score,
            iou,
            torch.tensor([True]),
            candidate,
            patch,
            **kwargs,
        )
        biased_score = score.detach() + torch.tensor([[[15.0, -9.0]]])
        # One additive bias per expression cancels within both directions.
        biased = official_assignment_delta_loss(
            biased_score,
            iou,
            torch.tensor([True]),
            candidate,
            patch,
            **kwargs,
        )
        self.assertTrue(torch.allclose(base["delta"], torch.tensor([4.0])))
        self.assertTrue(torch.allclose(base["delta"], biased["delta"]))
        self.assertTrue(
            torch.allclose(base["direction_delta"], torch.tensor([[2.0, 2.0]]))
        )
        self.assertEqual(base["selected_query0"].tolist(), [0])
        self.assertEqual(base["selected_query1"].tolist(), [1])
        self.assertEqual(base["deployment_correct_direction"].tolist(), [[False, False]])
        base["loss"].backward()
        self.assertIsNone(patch.grad)
        self.assertLess(float(score.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(score.grad[0, 0, 1]), 0.0)
        self.assertGreater(float(score.grad[0, 1, 0]), 0.0)
        self.assertLess(float(score.grad[0, 1, 1]), 0.0)
        self.assertTrue(torch.equal(score.grad[0, 2], torch.zeros(2)))

    def test_official_assignment_uses_one_gt_selected_query_per_referent(self):
        score = torch.zeros(1, 5, 2, requires_grad=True)
        iou = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.8, 0.1],
                    [0.0, 1.0],
                    [0.2, 0.7],
                    [0.0, 0.0],
                ]
            ]
        )
        candidate = torch.ones(1, 5, 2, dtype=torch.bool)
        patch = torch.zeros(1, 5, 1, requires_grad=True)
        kwargs = dict(
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            temperature=1.0,
        )
        base = official_assignment_delta_loss(
            score, iou, torch.tensor([True]), candidate, patch, **kwargs
        )
        expression_bias = torch.tensor([5.0, -7.0]).view(1, 1, 2)
        expression_biased = official_assignment_delta_loss(
            score.detach() + expression_bias,
            iou,
            torch.tensor([True]),
            candidate,
            patch,
            **kwargs,
        )
        self.assertTrue(
            torch.allclose(
                base["direction_delta"],
                expression_biased["direction_delta"],
                atol=1e-6,
            )
        )
        query_bias = torch.tensor([10.0, -4.0, 3.0, 9.0, 100.0]).view(
            1, 5, 1
        )
        query_biased = official_assignment_delta_loss(
            score.detach() + query_bias,
            iou,
            torch.tensor([True]),
            candidate,
            patch,
            **kwargs,
        )
        self.assertEqual(base["selected_query0"].tolist(), [0])
        self.assertEqual(base["selected_query1"].tolist(), [2])
        self.assertTrue(
            torch.allclose(
                query_biased["direction_delta"], torch.tensor([[7.0, -7.0]])
            )
        )
        self.assertGreater(
            float(query_biased["loss"].detach()), float(base["loss"].detach())
        )
        base["loss"].backward()
        self.assertIsNone(patch.grad)
        self.assertTrue(bool((score.grad[0, (0, 2)].abs() > 0).all().item()))
        self.assertTrue(torch.equal(score.grad[0, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(score.grad[0, 3], torch.zeros(2)))
        self.assertTrue(torch.equal(score.grad[0, 4], torch.zeros(2)))

    def test_deployment_hard_loss_owns_only_selected_and_hardest_wrong_queries(self):
        score = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [10.0, 9.0],
                    [8.0, 7.0],
                ]
            ],
            requires_grad=True,
        )
        iou = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ]
        )
        result = official_assignment_delta_loss(
            score,
            iou,
            torch.tensor([True]),
            torch.ones(1, 4, 2, dtype=torch.bool),
            torch.zeros(1, 4, 1, requires_grad=True),
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            temperature=1.0,
        )
        self.assertEqual(result["deployment_hard_query"].tolist(), [[2, 2]])
        self.assertEqual(result["deployment_hard_valid"].tolist(), [[True, True]])
        result["deployment_hard_loss"].backward()
        self.assertLess(float(score.grad[0, 0, 0]), 0.0)
        self.assertLess(float(score.grad[0, 1, 1]), 0.0)
        self.assertGreater(float(score.grad[0, 2, 0]), 0.0)
        self.assertGreater(float(score.grad[0, 2, 1]), 0.0)
        self.assertEqual(float(score.grad[0, 0, 1]), 0.0)
        self.assertEqual(float(score.grad[0, 1, 0]), 0.0)
        self.assertEqual(float(score.grad[0, 3].abs().sum()), 0.0)

    def test_official_assignment_gt_ties_use_cross_iou_then_query_index(self):
        score = torch.zeros(1, 6, 2, requires_grad=True)
        iou = torch.tensor(
            [
                [
                    [0.8, 0.1],
                    [0.8, 0.05],
                    [0.8, 0.05],
                    [0.2, 0.9],
                    [0.1, 0.9],
                    [0.1, 0.9],
                ]
            ]
        )
        result = official_assignment_delta_loss(
            score,
            iou,
            torch.tensor([True]),
            torch.ones(1, 6, 2, dtype=torch.bool),
            torch.zeros(1, 6, 1, requires_grad=True),
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            temperature=0.1,
        )
        self.assertEqual(result["selected_query0"].tolist(), [1])
        self.assertEqual(result["selected_query1"].tolist(), [4])
        result["loss"].backward()
        self.assertTrue(bool((score.grad[0, (1, 4)].abs() > 0).all().item()))
        self.assertEqual(float(score.grad[0, (0, 2, 3, 5)].abs().sum()), 0.0)

    def test_official_assignment_one_sided_unreachable_is_not_a_collision(self):
        score = torch.zeros(1, 2, 2, requires_grad=True)
        result = official_assignment_delta_loss(
            score,
            torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            torch.tensor([True]),
            torch.ones(1, 2, 2, dtype=torch.bool),
            torch.zeros(1, 2, 1, requires_grad=True),
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.3,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
            temperature=0.1,
        )
        self.assertEqual(result["runtime_valid"].tolist(), [False])
        self.assertEqual(result["role0_reachable"].tolist(), [True])
        self.assertEqual(result["role1_reachable"].tolist(), [False])
        self.assertEqual(result["query_collision"].tolist(), [False])
        self.assertEqual(float(result["loss"].detach()), 0.0)
        result["loss"].backward()
        self.assertEqual(float(score.grad.abs().sum()), 0.0)

    def test_official_assignment_is_sole_rank_loss_and_skips_invalid_pair(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=(
                DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT
            ),
            assignment_weight=1.0,
            temperature=1.0,
        )
        rank = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]]],
            requires_grad=True,
        )
        patch = torch.zeros(1, 3, 1, requires_grad=True)
        result = criterion(
            self._assignment_outputs(rank, patch),
            [self._assignment_target()],
        )
        self.assertEqual(int(criterion.rank_supervision_contract_id.item()), 4)
        self.assertEqual(
            float(result["stage_b_data_driven_assignment_data_rows"]), 1.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_assignment_runtime_rows"]), 1.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_assignment_correct_rows"]), 1.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_assignment_correct_directions"]), 2.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_assignment_deployment_correct_rows"]),
            0.0,
        )
        self.assertNotIn("loss_stage_b_data_driven_rank", criterion.weight_dict)
        self.assertNotIn(
            "loss_stage_b_data_driven_deployment_hard", criterion.weight_dict
        )
        self.assertNotIn(
            "loss_stage_b_data_driven_deployment_hard", result
        )
        self.assertEqual(
            float(result["loss_stage_b_data_driven_rank"].detach()), 0.0
        )
        result["loss_stage_b_data_driven_assignment"].backward()
        self.assertIsNone(patch.grad)
        self.assertTrue(bool((rank.grad[0, :2].abs() > 0).all().item()))
        self.assertEqual(float(rank.grad[0, 2].abs().sum()), 0.0)

        invalid_rank = rank.detach().clone().requires_grad_(True)
        invalid = criterion(
            self._assignment_outputs(invalid_rank),
            [self._assignment_target(valid=False)],
        )
        self.assertEqual(
            float(
                invalid["loss_stage_b_data_driven_assignment"].detach()
            ),
            0.0,
        )
        self.assertEqual(
            float(invalid["stage_b_data_driven_assignment_runtime_rows"]), 0.0
        )
        self.assertEqual(
            float(invalid["stage_b_data_driven_assignment_query_collision_rows"]),
            0.0,
        )

        with self.assertRaisesRegex(ValueError, "positive assignment_weight"):
            StageBDataDrivenCriterion(
                train_mode="rank_patch_only",
                category_complete=True,
                rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
                assignment_weight=0.0,
            )

        hard_criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
            assignment_weight=1.0,
            deployment_weight=1.0,
        )
        self.assertEqual(
            hard_criterion.weight_dict[
                "loss_stage_b_data_driven_deployment_hard"
            ],
            1.0,
        )
        hard_result = hard_criterion(
            self._assignment_outputs(rank.detach().clone().requires_grad_(True)),
            [self._assignment_target()],
        )
        self.assertIn("loss_stage_b_data_driven_deployment_hard", hard_result)

        with self.assertRaisesRegex(ValueError, "deployment_weight"):
            StageBDataDrivenCriterion(
                train_mode="rank_patch_only",
                category_complete=True,
                rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
                assignment_weight=1.0,
                deployment_weight=-1.0,
            )

    def test_shared_gap3_helper_matches_score_head_gate(self):
        patch = torch.tensor([[5.0, 2.0, -1.0, -10.0]])
        candidate = torch.tensor([[True, True, True, False]])
        eligible, normalized = data_driven_category_gate_mask(
            patch, candidate, max_gap=3.0, clip=5.0
        )
        heads = StageBDataDrivenScoreHeads(
            4,
            rank_dim=4,
            confidence_dim=4,
            gate_hidden_dim=4,
            category_gate=True,
            category_gate_max_gap=3.0,
            patch_score_clip=5.0,
        ).eval()
        rank = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        routed, observed_eligible, observed_normalized = (
            heads._apply_category_gate(patch, rank, candidate)
        )
        self.assertTrue(torch.equal(eligible, observed_eligible))
        self.assertTrue(torch.equal(normalized, observed_normalized))
        self.assertEqual(
            int(routed.argmax(dim=1).item()),
            int(rank.masked_fill(~eligible, -torch.inf).argmax(dim=1).item()),
        )

    def test_same_category_rank_requires_auxiliary_but_skips_missing_aux_hit(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
        )
        rank = torch.zeros(1, 2, requires_grad=True)
        outputs = self._same_category_outputs(
            rank,
            boxes=[
                [0.25, 0.5, 0.2, 0.2],
                [0.5, 0.1, 0.1, 0.1],
            ],
        )
        result = criterion(outputs, [self._same_category_target()])
        self.assertEqual(
            float(result["loss_stage_b_data_driven_rank"].detach()), 0.0
        )
        self.assertEqual(
            float(
                result[
                    "stage_b_data_driven_rank_skipped_no_hard_negative_rows"
                ]
            ),
            1.0,
        )
        result["loss_stage_b_data_driven_rank"].backward()
        self.assertEqual(float(rank.grad.abs().sum()), 0.0)

        with self.assertRaisesRegex(ValueError, "at least one auxiliary"):
            criterion(outputs, [self._same_category_target(auxiliary=False)])

    def test_same_category_primary_overlap_takes_precedence(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
        )
        rank = torch.zeros(1, 2, requires_grad=True)
        outputs = self._same_category_outputs(
            rank,
            boxes=[
                [0.25, 0.5, 0.2, 0.2],
                [0.5, 0.1, 0.1, 0.1],
            ],
        )
        result = criterion(
            outputs, [self._same_category_target(overlap=True)]
        )
        self.assertEqual(
            float(result["stage_b_data_driven_rank_positive_queries"]), 1.0
        )
        self.assertEqual(
            float(result["stage_b_data_driven_rank_hard_negative_queries"]),
            0.0,
        )

    def test_same_category_policy_is_fail_closed_and_patch_is_unchanged(self):
        with self.assertRaisesRegex(ValueError, "same-category"):
            StageBDataDrivenCriterion(
                train_mode="rank_patch_only",
                category_complete=False,
                rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
            )
        with self.assertRaisesRegex(ValueError, "same-category"):
            StageBDataDrivenCriterion(
                train_mode="rank_patch_only",
                category_complete=False,
                rank_supervision=(
                    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE
                ),
            )
        with self.assertRaisesRegex(ValueError, "rank_supervision"):
            StageBDataDrivenCriterion(
                train_mode="rank_patch_only",
                category_complete=True,
                rank_supervision="unknown",
            )

        target = self._same_category_target()
        legacy_patch = torch.tensor(
            [[[0.1], [0.2], [-0.3]]], requires_grad=True
        )
        hard_patch = legacy_patch.detach().clone().requires_grad_(True)
        legacy = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
        )
        hard = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
        )
        legacy_result = legacy(
            self._same_category_outputs(
                torch.zeros(1, 3, requires_grad=True), legacy_patch
            ),
            [target],
        )
        hard_result = hard(
            self._same_category_outputs(
                torch.zeros(1, 3, requires_grad=True), hard_patch
            ),
            [target],
        )
        self.assertTrue(
            torch.equal(
                legacy_result["loss_stage_b_data_driven_patch"],
                hard_result["loss_stage_b_data_driven_patch"],
            )
        )
        legacy_result["loss_stage_b_data_driven_patch"].backward()
        hard_result["loss_stage_b_data_driven_patch"].backward()
        self.assertTrue(torch.equal(legacy_patch.grad, hard_patch.grad))

    def test_same_category_rank_rejects_nonfinite_ignored_score(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
        )
        rank = torch.tensor([[0.0, 0.0, float("nan")]])
        with self.assertRaisesRegex(ValueError, "all be finite"):
            criterion(
                self._same_category_outputs(rank),
                [self._same_category_target()],
            )

    def test_dd2_confidence_uses_every_tn_and_commits_queue_after_step(self):
        criterion = StageBDataDrivenCriterion(
            train_mode="confidence_pair",
            category_complete=True,
            token_weight=0.0,
            positive_queue_size=8,
        )
        score = torch.tensor(
            [
                [[0.4, 0.2], [0.1, 0.3], [0.2, 0.1]],
                [[0.6, 0.5], [0.2, 0.1], [0.3, 0.4]],
            ],
            requires_grad=True,
        )
        token_logits = torch.zeros(2, 3, 2, 4, requires_grad=True)
        outputs = {
            "stage_b_data_driven_confidence_score": score,
            "stage_b_data_driven_confidence_token_logits": token_logits,
            "stage_b_data_driven_expression_token_mask": torch.ones(
                2, 2, 4, dtype=torch.bool
            ),
            "stage_b_data_driven_expression_input_ids": torch.ones(
                2, 2, 4, dtype=torch.int64
            ),
            "stage_b_data_driven_candidate_mask": torch.ones(
                2, 3, 2, dtype=torch.bool
            ),
            "pred_boxes": torch.tensor(
                [
                    [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]],
                    [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]],
                ]
            ),
        }
        targets = [_pair_target(), _pair_target()]
        result = criterion(outputs, targets)
        result["loss_stage_b_data_driven_confidence"].backward()
        self.assertGreater(float(score.grad[0, 1, 1]), 0.0)
        self.assertGreater(float(score.grad[1, 0, 1]), 0.0)
        self.assertEqual(int(criterion.fpr_positive_queue_count.item()), 0)
        criterion.commit_tail_queue(False)
        self.assertEqual(int(criterion.fpr_positive_queue_count.item()), 0)
        criterion(outputs, targets)
        criterion.commit_tail_queue(True)
        self.assertEqual(int(criterion.fpr_positive_queue_count.item()), 2)

    def test_dd3_token_targets_have_expected_gradient_signs(self):
        tokenizer = _LexicalFastTokenizer()
        pair = ["a pink box .", "a blue box ."]
        ids, expression_mask = _tokenized_pair(tokenizer, pair)
        criterion = StageBDataDrivenCriterion(
            train_mode="confidence_pair",
            category_complete=True,
            tokenizer=tokenizer,
            token_weight=1.0,
            shared_token_weight=0.25,
            positive_queue_size=0,
        )
        token_logits = torch.zeros(1, 2, 2, ids.shape[-1], requires_grad=True)
        outputs = {
            "stage_b_data_driven_confidence_score": torch.tensor(
                [[[0.5, 0.4], [0.2, 0.3]]], requires_grad=True
            ),
            "stage_b_data_driven_confidence_token_logits": token_logits,
            "stage_b_data_driven_expression_token_mask": expression_mask,
            "stage_b_data_driven_expression_input_ids": ids,
            "stage_b_data_driven_candidate_mask": torch.ones(
                1, 2, 2, dtype=torch.bool
            ),
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]]
            ),
        }
        result = criterion(outputs, [_pair_target()])
        self.assertEqual(float(result["stage_b_data_driven_trace_valid_rows"]), 1.0)
        result["loss_stage_b_data_driven_token"].backward()
        roles = build_direct_trace_token_roles(
            tokenizer,
            [pair],
            [_pair_target()["stage_b_data_driven_trace"]],
            ids,
            expression_mask,
            max_text_len=256,
        )
        changed = int(torch.nonzero(roles["changed"][0, 1])[0].item())
        shared = int(torch.nonzero(roles["shared"][0, 1])[0].item())
        positive = int(torch.nonzero(roles["positive"][0, 0])[0].item())
        self.assertGreater(float(token_logits.grad[0, 0, 1, changed]), 0.0)
        self.assertLess(float(token_logits.grad[0, 0, 1, shared]), 0.0)
        self.assertLess(float(token_logits.grad[0, 0, 0, positive]), 0.0)
        self.assertTrue(torch.equal(token_logits.grad[0, 1], torch.zeros_like(token_logits.grad[0, 1])))


if __name__ == "__main__":
    unittest.main()
