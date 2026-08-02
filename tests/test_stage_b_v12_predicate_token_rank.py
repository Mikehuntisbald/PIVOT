import unittest

import torch

from groundingdino.util import get_tokenlizer
from models.GroundingDINO.groundingdino import GroundingDINO
from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)
from models.GroundingDINO.stage_b_fixed_text_scorer import (
    FixedBoxFullTextScorer,
    build_stage_b_pair_token_diff_masks_from_ids,
)


class _V11TokenizerHarness:
    _tokenize_stage_b_v11_captions = GroundingDINO._tokenize_stage_b_v11_captions
    _build_stage_b_v11_pair_predicate_masks = (
        GroundingDINO._build_stage_b_v11_pair_predicate_masks
    )
    _build_stage_b_v15_score_token_masks = (
        GroundingDINO._build_stage_b_v15_score_token_masks
    )
    _build_stage_b_v21_direct_trace_token_roles = (
        GroundingDINO._build_stage_b_v21_direct_trace_token_roles
    )

    def __init__(self, max_text_len=32, *, dense_duty=False):
        self.max_text_len = int(max_text_len)
        self.stage_b_dense_duty = bool(dense_duty)
        self.stage_b_dense_duty_allow_incidental_trace_edits = False
        self.tokenizer = get_tokenlizer.get_tokenlizer(
            "/home/haoyi/.cache/huggingface/hub/models--bert-base-uncased/"
            "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
        )


class StageBV12PredicateTokenRankTest(unittest.TestCase):
    def test_id_diff_omits_shared_and_repeated_tokens(self):
        # 101/102/0 stand in for CLS/SEP/PAD and are ineligible. Pair 1 has a
        # shared object token; pair 2 changes only the first repeated token.
        input_ids = torch.tensor(
            [
                [[101, 11, 21, 102, 0, 0], [101, 12, 21, 102, 0, 0]],
                [[101, 31, 41, 32, 51, 102], [101, 32, 41, 32, 51, 102]],
            ]
        )
        attention = input_ids.ne(0)
        eligible = attention & ~torch.isin(
            input_ids, torch.tensor([0, 101, 102])
        )
        masks, valid = build_stage_b_pair_token_diff_masks_from_ids(
            input_ids,
            attention,
            torch.ones((2, 2), dtype=torch.bool),
            eligible,
            max_text_len=8,
        )

        self.assertEqual(valid.tolist(), [True, True])
        self.assertEqual(masks[0, 0].nonzero().flatten().tolist(), [1])
        self.assertEqual(masks[0, 1].nonzero().flatten().tolist(), [1])
        self.assertEqual(masks[1, 0].nonzero().flatten().tolist(), [1])
        self.assertEqual(masks[1, 1].nonzero().flatten().tolist(), [1])
        self.assertFalse(bool(masks[:, :, 2:].any().item()))

    def test_one_sided_diff_and_clean_slot_are_invalid_graph_inputs(self):
        input_ids = torch.tensor(
            [
                [[101, 11, 12, 21, 102], [101, 12, 21, 102, 0]],
                [[101, 31, 21, 102, 0], [101, 32, 21, 102, 0]],
            ]
        )
        attention = input_ids.ne(0)
        eligible = attention & ~torch.isin(
            input_ids, torch.tensor([0, 101, 102])
        )
        masks, valid = build_stage_b_pair_token_diff_masks_from_ids(
            input_ids,
            attention,
            torch.tensor([[True, True], [True, False]]),
            eligible,
            max_text_len=5,
        )
        self.assertEqual(valid.tolist(), [False, False])
        self.assertFalse(bool(masks.any().item()))

    def test_real_tokenizer_diff_and_batch_padding_alignment(self):
        harness = _V11TokenizerHarness(max_text_len=32)
        captions = [
            ["red car .", "blue car ."],
            ["black and white kitty .", "white and white kitty ."],
            ["small dog .", "object ."],
        ]
        valid = torch.tensor([[True, True], [True, True], [True, False]])
        masks, pair_valid = harness._build_stage_b_v11_pair_predicate_masks(
            captions, valid, torch.device("cpu")
        )
        self.assertEqual(pair_valid.tolist(), [True, True, False])

        flat = [caption for row in captions for caption in row]
        full = harness._tokenize_stage_b_v11_captions(flat)
        full_sequences = []
        for row_ids, row_attention in zip(full["input_ids"], full["attention_mask"]):
            full_sequences.append(row_ids[row_attention.bool()].tolist())
        split_sequences = []
        for start in range(0, len(flat), 3):
            chunk = harness._tokenize_stage_b_v11_captions(flat[start : start + 3])
            for row_ids, row_attention in zip(
                chunk["input_ids"], chunk["attention_mask"]
            ):
                split_sequences.append(row_ids[row_attention.bool()].tolist())
        self.assertEqual(full_sequences, split_sequences)

        padded_ids = full["input_ids"][:, : harness.max_text_len]
        width = int(padded_ids.shape[1])
        padded_ids = padded_ids.view(len(captions), 2, width)
        selected_tokens = []
        for batch_idx in range(2):
            pair_tokens = []
            for slot_idx in range(2):
                indices = masks[batch_idx, slot_idx, :width].nonzero().flatten()
                pair_tokens.append(
                    harness.tokenizer.convert_ids_to_tokens(
                        padded_ids[batch_idx, slot_idx, indices].tolist()
                    )
                )
            selected_tokens.append(pair_tokens)
        self.assertEqual(selected_tokens[0], [["red"], ["blue"]])
        self.assertEqual(selected_tokens[1], [["black"], ["white"]])
        self.assertFalse(bool(masks[2].any().item()))

    def test_real_tokenizer_noncanonical_mask_aligns_with_full_expression(self):
        harness = _V11TokenizerHarness(max_text_len=32)
        captions = [
            ["red car .", "blue car ."],
            ["small dog .", "dog ."],
        ]
        valid = torch.ones((2, 2), dtype=torch.bool)
        masks = harness._build_stage_b_v15_score_token_masks(
            captions,
            ["car .", "dog ."],
            valid,
            torch.device("cpu"),
        )
        tokenized = harness._tokenize_stage_b_v11_captions(
            [caption for row in captions for caption in row]
        )
        ids = tokenized["input_ids"].view(2, 2, -1)
        selected = []
        for batch_idx in range(2):
            selected_row = []
            for slot_idx in range(2):
                indices = masks[batch_idx, slot_idx, : ids.shape[-1]].nonzero().flatten()
                selected_row.append(
                    harness.tokenizer.convert_ids_to_tokens(
                        ids[batch_idx, slot_idx, indices].tolist()
                    )
                )
            selected.append(selected_row)
        self.assertEqual(selected, [[["red"], ["blue"]], [["small"], ["dog"]]])

    def test_dense_duty_keeps_category_only_score_mask_empty(self):
        harness = _V11TokenizerHarness(max_text_len=32, dense_duty=True)
        captions = [
            ["red car .", "car ."],
            ["small dog .", "dog ."],
        ]
        masks = harness._build_stage_b_v15_score_token_masks(
            captions,
            ["car .", "dog ."],
            torch.ones((2, 2), dtype=torch.bool),
            torch.device("cpu"),
        )
        tokenized = harness._tokenize_stage_b_v11_captions(
            [caption for row in captions for caption in row]
        )
        ids = tokenized["input_ids"].view(2, 2, -1)
        selected = []
        for batch_idx in range(2):
            selected_row = []
            for slot_idx in range(2):
                indices = masks[
                    batch_idx, slot_idx, : ids.shape[-1]
                ].nonzero().flatten()
                selected_row.append(
                    harness.tokenizer.convert_ids_to_tokens(
                        ids[batch_idx, slot_idx, indices].tolist()
                    )
                )
            selected.append(selected_row)
        self.assertEqual(selected, [[["red"], []], [["small"], []]])

    def test_word_veto_groups_real_bert_wordpieces_by_lexical_word(self):
        harness = _V11TokenizerHarness(max_text_len=32, dense_duty=True)
        captions = [["multicolored firetruck .", "blue firetruck ."]]
        score_mask, word_groups = harness._build_stage_b_v15_score_token_masks(
            captions,
            ["vehicle ."],
            torch.ones((1, 2), dtype=torch.bool),
            torch.device("cpu"),
            return_word_group_ids=True,
        )
        tokenized = harness._tokenize_stage_b_v11_captions(
            [caption for row in captions for caption in row]
        )
        ids = tokenized["input_ids"].view(1, 2, -1)
        positive_indices = score_mask[0, 0, : ids.shape[-1]].nonzero().flatten()
        positive_tokens = harness.tokenizer.convert_ids_to_tokens(
            ids[0, 0, positive_indices].tolist()
        )
        positive_groups = word_groups[0, 0, positive_indices].tolist()
        self.assertEqual(
            positive_tokens,
            ["multi", "##color", "##ed", "fire", "##tr", "##uck"],
        )
        self.assertEqual(positive_groups[:3], [0, 0, 0])
        self.assertEqual(positive_groups[3:], [1, 1, 1])
        self.assertTrue(bool((word_groups[~score_mask] == -1).all().item()))

    def test_dense_duty_direct_trace_roles_require_exact_reconstruction(self):
        harness = _V11TokenizerHarness(max_text_len=32, dense_duty=True)
        captions = [
            ["a pink striped box .", "a blue striped box ."],
            ["a pink striped box .", "an blue striped box ."],
            ["a red striped box .", "a red box ."],
        ]
        score_mask = harness._build_stage_b_v15_score_token_masks(
            captions,
            ["box .", "box .", "box ."],
            torch.ones((3, 2), dtype=torch.bool),
            torch.device("cpu"),
        )
        roles = harness._build_stage_b_v21_direct_trace_token_roles(
            captions,
            [
                {
                    "category": "color",
                    "replace_from": "pink",
                    "replace_to": "blue",
                    "replace_span": [1, 2],
                },
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
            score_mask,
            torch.device("cpu"),
        )
        self.assertEqual(roles["valid"].tolist(), [True, False, False])
        tokenized = harness._tokenize_stage_b_v11_captions(
            [caption for row in captions for caption in row]
        )
        ids = tokenized["input_ids"].view(3, 2, -1)

        def selected(mask, row, slot):
            indices = mask[row, slot, : ids.shape[-1]].nonzero().flatten()
            return harness.tokenizer.convert_ids_to_tokens(
                ids[row, slot, indices].tolist()
            )

        self.assertEqual(selected(roles["changed"], 0, 1), ["blue"])
        self.assertEqual(selected(roles["shared"], 0, 1), ["a", "striped"])
        self.assertFalse(bool(roles["positive"][1].any().item()))
        self.assertFalse(bool(roles["changed"][1].any().item()))
        self.assertFalse(bool(roles["positive"][2].any().item()))
        self.assertFalse(bool(roles["changed"][2].any().item()))

    def test_predicate_rank_uses_only_iou_positive_candidates(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=1.0,
            predicate_tn_rank_margin=0.3,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            batch_tail_separation_weight=0.0,
        )
        phrase_positive = torch.zeros((1, 4), requires_grad=True)
        phrase_tn = torch.zeros((1, 4), requires_grad=True)
        predicate_positive = torch.tensor(
            [[0.0, 3.0, -0.5, 4.0]], requires_grad=True
        )
        predicate_tn = torch.tensor(
            [[1.0, -3.0, 0.5, -4.0]], requires_grad=True
        )
        losses = criterion(
            candidate_logits=phrase_positive,
            candidate_ious=torch.tensor([[0.9, 0.2, 0.6, 0.4]]),
            local_tn_logits=phrase_tn,
            local_tn_mask=torch.tensor([True]),
            positive_predicate_logits=predicate_positive,
            local_tn_predicate_logits=predicate_tn,
            predicate_pair_valid=torch.tensor([True]),
        )
        losses["loss_fixed_text_predicate_tn_rank"].backward()

        self.assertEqual(
            float(losses["fixed_text_predicate_pair_query_count"]), 2.0
        )
        self.assertLess(float(predicate_positive.grad[0, 0]), 0.0)
        self.assertLess(float(predicate_positive.grad[0, 2]), 0.0)
        self.assertGreater(float(predicate_tn.grad[0, 0]), 0.0)
        self.assertGreater(float(predicate_tn.grad[0, 2]), 0.0)
        self.assertEqual(float(predicate_positive.grad[0, [1, 3]].abs().sum()), 0.0)
        self.assertEqual(float(predicate_tn.grad[0, [1, 3]].abs().sum()), 0.0)

    def test_masked_aggregation_gives_shared_object_zero_gradient(self):
        token_logits = torch.tensor(
            [
                [[0.0, -0.5, 8.0, 0.0]],
                [[0.0, 0.5, 8.0, 0.0]],
            ],
            requires_grad=True,
        )
        changed_mask = torch.tensor(
            [[False, True, False, False], [False, True, False, False]]
        )
        aggregate = FixedBoxFullTextScorer._aggregate_phrase_logits(
            token_logits, changed_mask
        )
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=1.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            batch_tail_separation_weight=0.0,
        )
        phrase_positive = torch.zeros((1, 1), requires_grad=True)
        phrase_tn = torch.zeros((1, 1), requires_grad=True)
        loss = criterion(
            candidate_logits=phrase_positive,
            candidate_ious=torch.ones((1, 1)),
            local_tn_logits=phrase_tn,
            positive_predicate_logits=aggregate[0].view(1, 1),
            local_tn_predicate_logits=aggregate[1].view(1, 1),
            predicate_pair_valid=torch.tensor([True]),
        )["loss_fixed_text_predicate_tn_rank"]
        loss.backward()

        self.assertNotEqual(float(token_logits.grad[:, :, 1].abs().sum()), 0.0)
        self.assertEqual(float(token_logits.grad[:, :, [0, 2, 3]].abs().sum()), 0.0)

    def test_invalid_pair_returns_graph_connected_zero(self):
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=1.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
            batch_tail_separation_weight=0.0,
        )
        phrase_positive = torch.zeros((1, 2), requires_grad=True)
        predicate_positive = torch.ones((1, 2), requires_grad=True)
        predicate_tn = torch.ones((1, 2), requires_grad=True)
        loss = criterion(
            candidate_logits=phrase_positive,
            candidate_ious=torch.tensor([[0.9, 0.1]]),
            positive_predicate_logits=predicate_positive,
            local_tn_predicate_logits=predicate_tn,
            predicate_pair_valid=torch.tensor([False]),
        )["loss_fixed_text_predicate_tn_rank"]
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(predicate_positive.grad)
        self.assertIsNotNone(predicate_tn.grad)
        self.assertEqual(float(predicate_positive.grad.abs().sum()), 0.0)
        self.assertEqual(float(predicate_tn.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
