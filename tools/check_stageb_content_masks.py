#!/usr/bin/env python3
from __future__ import annotations

import sys
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
from models.GroundingDINO.stage_b_criterion import StageBCriterion


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

    caption, *_masks, content_to_token_mask, _is_tn, _w, _g, invalid_records = ds._build_slot_text_masks(
        ["bird next to tree"],
        ["bird"],
        [["bird"]],
        slot_records=[{"phrase": "bird next to tree", "head_phrase": "bird", "text_is_negative": False}],
    )
    assert not invalid_records, invalid_records
    assert "to" in _masked_tokens(ds, caption, content_to_token_mask[0])


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


def main() -> None:
    check_content_mask_semantics()
    check_tn_changed_tokens_retained()
    check_postprocess_ignores_training_masks()
    check_phrase_loss_disabled()
    print("Stage-B content-mask sanity checks passed.")


if __name__ == "__main__":
    main()
