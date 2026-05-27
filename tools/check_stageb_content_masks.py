#!/usr/bin/env python3
from __future__ import annotations

import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.patch_episode import (
    PatchEpisodeJsonlDataset,
    _TN_GROUP_TO_ID,
    _tn_category_group,
)
from models.GroundingDINO.groundingdino import PostProcessStageB
from models.GroundingDINO.groundingdino import GroundingDINO
from models.GroundingDINO.stage_b_score import compute_stage_b_slot_logits
from models.GroundingDINO.stage_b_score import aggregate_stage_b_tokens
from models.GroundingDINO.stage_b_criterion import StageBCriterion
from util.misc import NestedTensor
import engine


def _make_dataset_shell() -> PatchEpisodeJsonlDataset:
    ds = object.__new__(PatchEpisodeJsonlDataset)
    ds.cfg = SimpleNamespace(
        build_text_token_masks=True,
        max_text_len=64,
        text_mask_warn_limit=0,
        patch_text_aug_max_words=64,
        skip_ambiguous_tn=True,
        skip_tn_if_changed_span_not_found=True,
        skip_tn_if_changed_span_empty_after_filter=True,
        skip_tn_if_neg_overlaps_canonical=True,
        use_tn_category_weights=True,
        default_tn_category_weight=1.0,
        skip_relation_like_tn_in_v1=False,
    )
    ds._text_mask_warn_count = 0
    ds._text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    return ds


def _make_dataset_shell_without_text_masks() -> PatchEpisodeJsonlDataset:
    ds = object.__new__(PatchEpisodeJsonlDataset)
    ds.cfg = SimpleNamespace(build_text_token_masks=False, patch_text_aug_max_words=64)
    ds._text_tokenizer = None
    return ds


def _masked_tokens(ds: PatchEpisodeJsonlDataset, caption: str, mask: torch.Tensor) -> set[str]:
    tokenized = ds._text_tokenizer(caption, truncation=True, max_length=int(ds.cfg.max_text_len))
    tokens = ds._text_tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    return {str(tokens[i]).replace("##", "") for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()}


def check_content_mask_semantics() -> None:
    ds = _make_dataset_shell()
    (
        caption,
        phrase_to_token_mask,
        canonical_to_token_mask,
        attr_pos_to_token_mask,
        attr_neg_to_token_mask,
        _relation_to_token_mask,
        content_to_token_mask,
        is_tn,
        attr_neg_weight_mask,
        tn_group_ids,
        _rank_phrase_mask,
        _rank_canonical_mask,
        _has_rank_positive,
        _rank_positive_captions,
        invalid_records,
    ) = ds._build_slot_text_masks(
        ["the blue bird left of the tree"],
        ["bird"],
        [["bird"]],
        slot_records=[{"phrase": "the blue bird left of the tree", "head_phrase": "bird", "text_is_negative": False}],
    )
    assert not invalid_records, invalid_records
    assert phrase_to_token_mask[0].any()
    assert not is_tn[0].item()
    assert int(tn_group_ids[0].item()) == _TN_GROUP_TO_ID["other"]
    content_tokens = _masked_tokens(ds, caption, content_to_token_mask[0])
    canonical_tokens = _masked_tokens(ds, caption, canonical_to_token_mask[0])
    neg_tokens = _masked_tokens(ds, caption, attr_neg_to_token_mask[0])
    assert "bird" in canonical_tokens
    assert "bird" not in content_tokens
    assert "the" not in content_tokens
    assert "[CLS]" not in content_tokens and "[SEP]" not in content_tokens
    assert not (content_to_token_mask[0] & ~phrase_to_token_mask[0]).any()
    assert "blue" in content_tokens
    assert "left" in content_tokens and "of" in content_tokens
    assert "tree" in content_tokens
    assert torch.equal(content_to_token_mask, attr_pos_to_token_mask)
    assert not neg_tokens
    assert not attr_neg_weight_mask[0].any()

    (
        caption,
        _phrase_to_token_mask,
        _canonical_to_token_mask,
        _attr_pos_to_token_mask,
        _attr_neg_to_token_mask,
        _relation_to_token_mask,
        content_to_token_mask,
        _is_tn,
        _w,
        _g,
        _rank_phrase_mask,
        _rank_canonical_mask,
        _has_rank_positive,
        _rank_positive_captions,
        invalid_records,
    ) = ds._build_slot_text_masks(
        ["bird next to tree"],
        ["bird"],
        [["bird"]],
        slot_records=[{"phrase": "bird next to tree", "head_phrase": "bird", "text_is_negative": False}],
    )
    assert not invalid_records, invalid_records
    assert "to" in _masked_tokens(ds, caption, content_to_token_mask[0])


def check_build_slot_text_masks_early_return_arity() -> None:
    ds = _make_dataset_shell_without_text_masks()
    result = ds._build_slot_text_masks(
        ["blue bird"],
        ["bird"],
        [["bird"]],
        slot_records=[{"phrase": "blue bird", "head_phrase": "bird", "text_is_negative": False}],
    )
    assert len(result) == 15
    caption, *rest = result
    assert caption == "blue bird ."
    assert all(value is None for value in rest[:12])
    assert rest[12] == []
    assert rest[13] == []


def check_tn_changed_tokens_retained() -> None:
    ds = _make_dataset_shell()
    (
        caption,
        _phrase_to_token_mask,
        _canonical_to_token_mask,
        attr_pos_to_token_mask,
        attr_neg_to_token_mask,
        _relation_to_token_mask,
        content_to_token_mask,
        is_tn,
        attr_neg_weight_mask,
        tn_group_ids,
        _rank_phrase_mask,
        _rank_canonical_mask,
        _has_rank_positive,
        _rank_positive_captions,
        invalid_records,
    ) = ds._build_slot_text_masks(
        ["bird next to tree"],
        ["bird"],
        [["bird"]],
        slot_records=[
            {
                "phrase": "bird next to tree",
                "head_phrase": "bird",
                "text_is_negative": True,
                "positive_phrase": "bird beside tree",
                "replace_from": ["beside"],
                "replace_to": ["next to"],
                "replace_category": ["spatial relation"],
            }
        ],
    )
    assert not invalid_records, invalid_records
    assert is_tn[0].item()
    assert int(tn_group_ids[0].item()) == _TN_GROUP_TO_ID["spatial_like"]
    neg_tokens = _masked_tokens(ds, caption, attr_neg_to_token_mask[0])
    content_tokens = _masked_tokens(ds, caption, content_to_token_mask[0])
    pos_tokens = _masked_tokens(ds, caption, attr_pos_to_token_mask[0])
    assert "next" in neg_tokens and "to" in neg_tokens
    assert "next" in content_tokens and "to" in content_tokens
    assert "next" not in pos_tokens and "to" not in pos_tokens
    effective_content = content_to_token_mask[0] & ~attr_neg_to_token_mask[0]
    effective_content_tokens = _masked_tokens(ds, caption, effective_content)
    assert "next" not in effective_content_tokens and "to" not in effective_content_tokens
    assert attr_neg_weight_mask[0][attr_neg_to_token_mask[0]].min().item() == 1.0
    assert _tn_category_group("action") == "relation_action_like"
    assert _tn_category_group("hair color") == "color_like"


def check_postprocess_ignores_training_masks() -> None:
    torch.manual_seed(7)
    post = PostProcessStageB(beta=1.0, canonical_weight=0.15, text_agg="mean", output_sigmoid_scores=False)
    outputs = {
        "pred_logits_patch": torch.randn(1, 3, 2),
        "pred_logits_text": torch.randn(1, 3, 8),
        "phrase_to_token_mask": torch.tensor(
            [[[0, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0]]],
            dtype=torch.bool,
        ),
        "canonical_to_token_mask": torch.tensor(
            [[[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]]],
            dtype=torch.bool,
        ),
    }
    base = post.compute_slot_logits(outputs)
    outputs_changed = dict(outputs)
    outputs_changed["content_to_token_mask"] = torch.zeros_like(outputs["phrase_to_token_mask"])
    outputs_changed["phrase_semantic_token_mask"] = torch.ones_like(outputs["phrase_to_token_mask"])
    changed = post.compute_slot_logits(outputs_changed)
    assert torch.allclose(base, changed), (base, changed)


def check_phrase_loss_disabled() -> None:
    class DummyPatchCriterion:
        matcher = None
        weight_dict = {}

    criterion = StageBCriterion(
        patch_criterion=DummyPatchCriterion(),
        lambda_text=1.0,
        canonical_pos_weight=0.15,
        use_phrase_tn_loss=False,
        lambda_phrase=0.0,
    )
    outputs = {
        "pred_logits_text": torch.tensor([[[0.2, 0.7, -0.6, 1.1, -1.4, 0.3]]], dtype=torch.float32)
    }
    target = {
        "phrase_to_token_mask": torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.bool),
        "canonical_to_token_mask": torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.bool),
        "content_to_token_mask": torch.tensor([[0, 1, 0, 1, 1, 0]], dtype=torch.bool),
        "attr_neg_to_token_mask": torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.bool),
        "attr_neg_weight_mask": torch.tensor([[0, 0, 0, 0, 1, 0]], dtype=torch.float32),
        "is_tn": torch.tensor([True]),
        "tn_group_ids": torch.tensor([_TN_GROUP_TO_ID["relation_action_like"]], dtype=torch.long),
    }
    match_ctx = {
        "all_indices": [(torch.tensor([0]), torch.tensor([0]))],
        "matched_patch_idx_list": [torch.tensor([0])],
    }
    losses = criterion._compute_text_loss(outputs, [target], match_ctx)
    expected = losses["content_pos_loss"] + losses["canonical_loss"] + losses["tn_neg_loss"]
    assert losses["text_phrase_tn_loss_raw"].item() == 0.0
    assert losses["text_phrase_tn_slot_count"].item() == 0.0
    assert torch.allclose(losses["loss_text"], expected)
    assert losses["tn_neg_count_relation_action_like"].item() == 1.0


def check_rank_positive_uses_positive_phrase_only() -> None:
    ds = _make_dataset_shell()
    (
        _caption,
        _phrase_to_token_mask,
        _canonical_to_token_mask,
        _attr_pos_to_token_mask,
        _attr_neg_to_token_mask,
        _relation_to_token_mask,
        _content_to_token_mask,
        _is_tn,
        _attr_neg_weight_mask,
        _tn_group_ids,
        _rank_phrase_mask,
        _rank_canonical_mask,
        has_rank_positive,
        rank_positive_captions,
        invalid_records,
    ) = ds._build_slot_text_masks(
        ["bird next to tree"],
        ["bird"],
        [["bird"]],
        slot_records=[
            {
                "phrase": "bird next to tree",
                "head_phrase": "bird",
                "text_is_negative": True,
                "try_tn_head_phrase": "bird beside tree",
                "replace_from": ["beside"],
                "replace_to": ["next to"],
                "replace_category": ["spatial relation"],
            }
        ],
    )
    assert not invalid_records, invalid_records
    assert not has_rank_positive[0].item()
    assert rank_positive_captions[0] is None

    (
        _caption,
        _phrase_to_token_mask,
        _canonical_to_token_mask,
        _attr_pos_to_token_mask,
        _attr_neg_to_token_mask,
        _relation_to_token_mask,
        _content_to_token_mask,
        _is_tn,
        _attr_neg_weight_mask,
        _tn_group_ids,
        rank_phrase_mask,
        rank_canonical_mask,
        has_rank_positive,
        rank_positive_captions,
        invalid_records,
    ) = ds._build_slot_text_masks(
        ["bird next to tree"],
        ["bird"],
        [["bird"]],
        slot_records=[
            {
                "phrase": "bird next to tree",
                "head_phrase": "bird",
                "text_is_negative": True,
                "positive_phrase": "bird beside tree",
                "try_tn_head_phrase": "wrong fallback",
                "replace_from": ["beside"],
                "replace_to": ["next to"],
                "replace_category": ["spatial relation"],
            }
        ],
    )
    assert not invalid_records, invalid_records
    assert has_rank_positive[0].item()
    assert rank_positive_captions[0] == "bird beside tree ."
    assert rank_phrase_mask[0].any()
    assert rank_canonical_mask[0].any()


def check_shared_score_helper_matches_postprocess() -> None:
    torch.manual_seed(11)
    outputs = {
        "pred_logits_patch": torch.randn(1, 4, 2),
        "pred_logits_text": torch.randn(1, 4, 9),
        "phrase_to_token_mask": torch.tensor(
            [[[0, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0]]],
            dtype=torch.bool,
        ),
        "canonical_to_token_mask": torch.tensor(
            [[[0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0]]],
            dtype=torch.bool,
        ),
    }
    post = PostProcessStageB(beta=0.8, canonical_weight=0.2, text_agg="mean", output_sigmoid_scores=False)
    from_post = post.compute_slot_logits(outputs)
    from_helper = compute_stage_b_slot_logits(
        outputs,
        beta=0.8,
        canonical_weight=0.2,
        text_agg="mean",
        detach_patch=False,
    )
    assert torch.allclose(from_post, from_helper)

    post_mix = PostProcessStageB(
        beta=0.8,
        canonical_weight=0.2,
        text_agg="mean_norm_softmin",
        softmin_tau=0.7,
        mean_softmin_alpha=0.35,
        output_sigmoid_scores=False,
    )
    from_post_mix = post_mix.compute_slot_logits(outputs)
    from_helper_mix = compute_stage_b_slot_logits(
        outputs,
        beta=0.8,
        canonical_weight=0.2,
        text_agg="mean_norm_softmin",
        softmin_tau=0.7,
        mean_softmin_alpha=0.35,
        detach_patch=False,
    )
    assert torch.allclose(from_post_mix, from_helper_mix)


def check_mean_normalized_softmin_scorer() -> None:
    logits = torch.tensor([[[1.0, 3.0, -1.0, 9.0]]])
    mask = torch.tensor([[[1, 1, 1, 0]]], dtype=torch.bool)
    tau = 0.7
    alpha = 0.25
    score = aggregate_stage_b_tokens(
        logits,
        mask,
        text_agg="mean_norm_softmin",
        softmin_tau=tau,
        mean_softmin_alpha=alpha,
    )
    token_logits = torch.tensor([1.0, 3.0, -1.0])
    mean_score = token_logits.mean()
    softmin_score = -tau * torch.logsumexp(-token_logits / tau, dim=0)
    normalized_softmin = softmin_score + tau * torch.log(torch.tensor(float(token_logits.numel())))
    expected = alpha * mean_score + (1.0 - alpha) * normalized_softmin
    assert torch.allclose(score[0, 0, 0], expected)

    equal_logits = torch.full((1, 1, 4), 2.5)
    equal_mask = torch.tensor([[[1, 1, 1, 1]]], dtype=torch.bool)
    equal_score = aggregate_stage_b_tokens(
        equal_logits,
        equal_mask,
        text_agg="mean_normalized_softmin",
        softmin_tau=0.3,
        mean_softmin_alpha=0.0,
    )
    assert torch.allclose(equal_score[0, 0, 0], torch.tensor(2.5), atol=1e-6)


def check_phrase_rank_loss_independent_and_match_by_target() -> None:
    class DummyPatchCriterion:
        matcher = None
        weight_dict = {}

        def compute_matching(self, outputs, targets):
            if outputs.get("is_rank_pos", False):
                return {
                    "all_indices": [(torch.tensor([1]), torch.tensor([0]))],
                    "matched_patch_idx_list": [torch.tensor([0])],
                }
            return {
                "all_indices": [(torch.tensor([0, 2]), torch.tensor([0, 1]))],
                "matched_patch_idx_list": [torch.tensor([0, 0])],
            }

    criterion = StageBCriterion(
        patch_criterion=DummyPatchCriterion(),
        lambda_text=1.0,
        stage_b_rank_margin=0.3,
        stage_b_rank_loss_coef=1.0,
        stage_b_rank_detach_patch=True,
        stage_b_rank_beta=1.0,
        stage_b_rank_canonical_weight=0.0,
    )
    pred_patch_neg = torch.tensor([[[0.0], [0.0], [0.0]]], requires_grad=True)
    pred_patch_pos = torch.tensor([[[0.0], [0.0], [0.0]]], requires_grad=True)
    outputs = {
        "pred_logits_patch": pred_patch_neg,
        "pred_logits_text": torch.tensor([[[0.2, 0.2], [0.0, 0.0], [0.9, 0.9]]], requires_grad=True),
        "pred_boxes": torch.zeros(1, 3, 4),
        "phrase_to_token_mask": torch.tensor([[[1, 1]]], dtype=torch.bool),
        "canonical_to_token_mask": torch.zeros(1, 1, 2, dtype=torch.bool),
        "rank_pos_outputs": {
            "is_rank_pos": True,
            "pred_logits_patch": pred_patch_pos,
            "pred_logits_text": torch.tensor([[[0.1, 0.1], [0.4, 0.4], [0.0, 0.0]]], requires_grad=True),
            "pred_boxes": torch.zeros(1, 3, 4),
            "phrase_to_token_mask": torch.tensor([[[1, 1]]], dtype=torch.bool),
            "canonical_to_token_mask": torch.zeros(1, 1, 2, dtype=torch.bool),
        },
        "rank_pair_map": torch.tensor([0], dtype=torch.long),
    }
    targets = [{"labels": torch.tensor([5, 5]), "boxes": torch.zeros(2, 4)}]
    rank_pos_targets = [
        {
            "labels": torch.tensor([5]),
            "boxes": torch.zeros(1, 4),
            "support_class": torch.tensor([5]),
            "rank_source_slot": torch.tensor([0]),
            "rank_target_ids": torch.tensor([1]),
        }
    ]
    outputs["rank_pos_targets"] = rank_pos_targets
    match_ctx = DummyPatchCriterion().compute_matching(outputs, targets)
    rank_losses = criterion._compute_phrase_rank_loss(outputs, targets, match_ctx)
    # Match-by-target should use neg query 2 for original target 1, not neg query 0.
    expected = torch.relu(torch.tensor(0.9 - 0.4 + 0.3))
    assert torch.allclose(rank_losses["loss_phrase_rank"], expected)
    text_losses = criterion._compute_text_loss(
        {"pred_logits_text": torch.tensor([[[0.2, 0.7, -0.6]]])},
        [
            {
                "phrase_to_token_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
                "canonical_to_token_mask": torch.tensor([[0, 1, 0]], dtype=torch.bool),
                "content_to_token_mask": torch.tensor([[1, 0, 0]], dtype=torch.bool),
                "attr_neg_to_token_mask": torch.tensor([[0, 0, 1]], dtype=torch.bool),
                "attr_neg_weight_mask": torch.tensor([[0, 0, 1]], dtype=torch.float32),
                "is_tn": torch.tensor([True]),
            }
        ],
        {"all_indices": [(torch.tensor([0]), torch.tensor([0]))], "matched_patch_idx_list": [torch.tensor([0])]},
    )
    assert "loss_phrase_rank" not in text_losses
    rank_losses["loss_phrase_rank"].backward()
    assert pred_patch_neg.grad is None
    assert pred_patch_pos.grad is None


def check_phrase_rank_violation_count() -> None:
    class DummyPatchCriterion:
        matcher = None
        weight_dict = {}

        def __init__(self, neg_query: int = 0, pos_query: int = 0):
            self.neg_query = int(neg_query)
            self.pos_query = int(pos_query)

        def compute_matching(self, outputs, targets):
            query = self.pos_query if outputs.get("is_rank_pos", False) else self.neg_query
            return {
                "all_indices": [(torch.tensor([query]), torch.tensor([0]))],
                "matched_patch_idx_list": [torch.tensor([0])],
            }

    def run_case(s_pos: float, s_neg: float):
        patch_criterion = DummyPatchCriterion()
        criterion = StageBCriterion(
            patch_criterion=patch_criterion,
            lambda_text=1.0,
            stage_b_rank_margin=0.3,
            stage_b_rank_loss_coef=1.0,
            stage_b_rank_detach_patch=True,
            stage_b_rank_beta=1.0,
            stage_b_rank_canonical_weight=0.0,
        )
        outputs = {
            "pred_logits_patch": torch.zeros(1, 1, 1),
            "pred_logits_text": torch.tensor([[[s_neg]]], dtype=torch.float32),
            "pred_boxes": torch.zeros(1, 1, 4),
            "phrase_to_token_mask": torch.tensor([[[1]]], dtype=torch.bool),
            "canonical_to_token_mask": torch.zeros(1, 1, 1, dtype=torch.bool),
            "rank_pos_outputs": {
                "is_rank_pos": True,
                "pred_logits_patch": torch.zeros(1, 1, 1),
                "pred_logits_text": torch.tensor([[[s_pos]]], dtype=torch.float32),
                "pred_boxes": torch.zeros(1, 1, 4),
                "phrase_to_token_mask": torch.tensor([[[1]]], dtype=torch.bool),
                "canonical_to_token_mask": torch.zeros(1, 1, 1, dtype=torch.bool),
            },
            "rank_pair_map": torch.tensor([0], dtype=torch.long),
            "rank_pos_targets": [
                {
                    "labels": torch.tensor([5]),
                    "boxes": torch.zeros(1, 4),
                    "rank_source_slot": torch.tensor([0]),
                    "rank_target_ids": torch.tensor([0]),
                }
            ],
        }
        targets = [{"labels": torch.tensor([5]), "boxes": torch.zeros(1, 4)}]
        return criterion._compute_phrase_rank_loss(outputs, targets, patch_criterion.compute_matching(outputs, targets))

    no_violation = run_case(s_pos=1.0, s_neg=0.0)
    assert torch.allclose(no_violation["loss_phrase_rank"], torch.tensor(0.0))
    assert no_violation["phrase_rank_used_pair_count"].item() == 1.0
    assert no_violation["phrase_rank_violation_count"].item() == 0.0
    assert "phrase_rank_active_pair_count" not in no_violation

    violation = run_case(s_pos=0.1, s_neg=0.5)
    assert violation["loss_phrase_rank"].item() > 0.0
    assert violation["phrase_rank_used_pair_count"].item() == 1.0
    assert violation["phrase_rank_violation_count"].item() == 1.0
    assert "phrase_rank_active_pair_count" not in violation


def check_rank_forward_disables_patch_dn() -> None:
    model_forward_src = inspect.getsource(GroundingDINO.forward)
    engine_src = inspect.getsource(engine.train_one_epoch)
    assert 'disable_patch_dn = bool(kw.get("disable_patch_dn", False))' in model_forward_src
    assert "and (not disable_patch_dn)" in model_forward_src
    assert '"phrase_to_token_mask",' in model_forward_src
    assert '"phrase_to_token_mask",' in engine_src
    assert "phrase_to_token_mask=torch.stack" in engine_src
    assert "has_rank_pairs = bool(rank_subbatch is not None and rank_subbatch.get(\"indices\"))" in engine_src
    assert "disable_patch_dn=has_rank_pairs" in engine_src
    assert "disable_patch_dn=True" in engine_src


def check_drift_batch_handles_non_tensor_targets() -> None:
    samples = NestedTensor(
        torch.zeros(1, 3, 4, 4),
        torch.zeros(1, 4, 4, dtype=torch.bool),
    )
    targets = [
        {
            "labels": torch.tensor([1]),
            "rank_positive_captions": ["a red car ."],
            "meta": {"source": "tn"},
        }
    ]
    batch = engine._clone_stage_b_drift_batch(samples, targets, ["a blue car ."], None, None, None)
    assert batch["targets"][0]["rank_positive_captions"] == ["a red car ."]
    assert batch["targets"][0]["rank_positive_captions"] is not targets[0]["rank_positive_captions"]
    assert batch["targets"][0]["meta"] == {"source": "tn"}
    moved = engine._move_stage_b_drift_batch_to_device(batch, torch.device("cpu"))
    assert moved["targets"][0]["rank_positive_captions"] == ["a red car ."]
    assert torch.equal(moved["targets"][0]["labels"], torch.tensor([1]))


def check_rank_subbatch_keeps_multiple_valid_slots() -> None:
    samples = NestedTensor(
        torch.zeros(1, 3, 4, 4),
        torch.zeros(1, 4, 4, dtype=torch.bool),
    )
    target = {
        "labels": torch.tensor([10, 20]),
        "boxes": torch.zeros(2, 4),
        "support_classes": torch.tensor([10, 20]),
        "is_tn": torch.tensor([True, True]),
        "has_rank_positive": torch.tensor([True, True]),
        "rank_positive_captions": ["blue bird .", "red car ."],
        "rank_positive_phrase_to_token_mask": torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool),
        "rank_positive_canonical_to_token_mask": torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.bool),
        "phrase_to_token_mask": torch.ones(2, 3, dtype=torch.bool),
        "canonical_to_token_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    patch_global = torch.zeros(1, 2, 4)
    patch_mask = torch.ones(1, 2, dtype=torch.bool)
    subbatch = engine._build_stage_b_rank_subbatch(
        SimpleNamespace(stage_b_enable_phrase_rank=True),
        samples,
        [target],
        ["tn caption ."],
        None,
        patch_global,
        patch_mask,
    )
    assert subbatch is not None
    assert subbatch["indices"] == [0, 0]
    assert subbatch["captions"] == ["blue bird .", "red car ."]
    assert len(subbatch["targets"]) == 2
    assert subbatch["patch_global"].shape == (2, 1, 4)
    assert subbatch["targets"][0]["rank_source_slot"].item() == 0
    assert subbatch["targets"][1]["rank_source_slot"].item() == 1
    assert subbatch["targets"][0]["rank_target_ids"].tolist() == [0]
    assert subbatch["targets"][1]["rank_target_ids"].tolist() == [1]


def main() -> None:
    check_content_mask_semantics()
    check_build_slot_text_masks_early_return_arity()
    check_tn_changed_tokens_retained()
    check_postprocess_ignores_training_masks()
    check_phrase_loss_disabled()
    check_rank_positive_uses_positive_phrase_only()
    check_shared_score_helper_matches_postprocess()
    check_mean_normalized_softmin_scorer()
    check_phrase_rank_loss_independent_and_match_by_target()
    check_phrase_rank_violation_count()
    check_rank_forward_disables_patch_dn()
    check_drift_batch_handles_non_tensor_targets()
    check_rank_subbatch_keeps_multiple_valid_slots()
    print("Stage-B content-mask sanity checks passed.")


if __name__ == "__main__":
    main()
