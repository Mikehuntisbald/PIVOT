# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.visualizer import COCOVisualizer
from groundingdino.util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss
from .stage_b_score import compute_stage_b_slot_logits

from .matcher import build_matcher
from .patch_encoder import PatchEncoder

class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        patch_gate_with_text: bool = True,
        patch_only: bool = False,
        enable_patch_branch: bool = True,
        patch_only_compute_text_logits: bool = False,
        patch_logit_scale_init: float = 14.2857142857,  # CLIP: 1/0.07
        patch_logit_scale_max: float = 100.0,
        patch_dn_num_queries: int = 0,
        patch_dn_box_noise_scale: float = 0.4,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = int(max_text_len)
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone

        self.patch_only = patch_only
        self.enable_patch_branch = bool(enable_patch_branch)
        self.patch_only_compute_text_logits = bool(patch_only_compute_text_logits)
        if self.enable_patch_branch:
            self.patch_encoder = PatchEncoder(
                backbone=self.backbone,
                hidden_dim=self.hidden_dim,
                gate_with_text=patch_gate_with_text,
                max_text_len=self.max_text_len,
            )

            # Project decoder queries before computing patch similarity.
            self.query_proj_for_patch = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(self.query_proj_for_patch.weight)
            nn.init.constant_(self.query_proj_for_patch.bias, 0)

            # CLIP-style learnable temperature for patch similarity.
            # Stored in log-space; scale = exp(logit_scale). Clamp for stability.
            self.patch_logit_scale = nn.Parameter(torch.tensor(float(patch_logit_scale_init)).log())
            self.patch_logit_scale_max = float(patch_logit_scale_max)
        else:
            self.patch_encoder = None
            self.query_proj_for_patch = None
            self.patch_logit_scale = None
            self.patch_logit_scale_max = float(patch_logit_scale_max)
        self.stage_b_verifier = None
        self.stage_b_fixed_text_scorer = None
        self.stage_b_legacy_global_gate = None
        self.stage_b_legacy_global_gate_score_kwargs = {}
        self.stage_b_gdino_score_adapter = None
        self.stage_b_u0_patch_rank_adapter = None
        self.stage_b_u0_gate_aligned_rank_residual = None
        self.stage_b_u0_gate_aligned_patch_residual = None
        self.stage_b_data_driven_score_heads = None
        self.stage_b_data_driven_patch_residual = None
        self.stage_b_native_patch_category = False
        self.stage_b_v11_candidate_topk = 50
        self.stage_b_v15_exclude_canonical_from_score = False
        self.stage_b_dense_duty = False
        self.stage_b_dense_duty_confidence_word_groups = False
        self.stage_b_dense_duty_allow_incidental_trace_edits = False

        # Patch-only DN / GT-guided queries: add extra queries initialized from GT boxes (+noise)
        # to guarantee some positives when box head is imperfect or iou_thr is strict.
        self.patch_dn_num_queries = int(patch_dn_num_queries)
        self.patch_dn_box_noise_scale = float(patch_dn_box_noise_scale)
        if self.patch_dn_num_queries > 0:
            self.patch_dn_tgt = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.normal_(self.patch_dn_tgt, std=0.02)
        else:
            self.patch_dn_tgt = None

        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    @staticmethod
    def select_stage_b_v11_candidates(
        query_hs: torch.Tensor,
        pred_boxes: torch.Tensor,
        patch_score: torch.Tensor,
        topk: int,
    ):
        """Select immutable Stage-A candidates using only the patch ranking."""
        if query_hs.dim() != 3:
            raise ValueError(f"query_hs must be (B,Q,D), got {tuple(query_hs.shape)}")
        if pred_boxes.dim() != 3 or pred_boxes.shape[-1] != 4:
            raise ValueError(f"pred_boxes must be (B,Q,4), got {tuple(pred_boxes.shape)}")
        if query_hs.shape[:2] != pred_boxes.shape[:2]:
            raise ValueError("query_hs and pred_boxes must share their (B,Q) dimensions")
        if patch_score.dim() == 3:
            if patch_score.shape[-1] != 1:
                raise ValueError(
                    "Stage B v11 requires exactly one patch/localization slot, "
                    f"got patch_score shape {tuple(patch_score.shape)}"
                )
            patch_score = patch_score[..., 0]
        if patch_score.shape != query_hs.shape[:2]:
            raise ValueError(
                f"patch_score must be (B,Q), got {tuple(patch_score.shape)} for "
                f"queries {tuple(query_hs.shape)}"
            )
        if int(topk) <= 0:
            raise ValueError("Stage B v11 candidate topk must be positive")

        candidate_count = min(int(topk), int(query_hs.shape[1]))
        candidate_idx = torch.topk(
            patch_score.detach(), candidate_count, dim=1, largest=True, sorted=True
        ).indices
        candidate_hs = torch.gather(
            query_hs.detach(),
            1,
            candidate_idx.unsqueeze(-1).expand(-1, -1, query_hs.shape[-1]),
        )
        candidate_boxes = torch.gather(
            pred_boxes.detach(),
            1,
            candidate_idx.unsqueeze(-1).expand(-1, -1, 4),
        )
        return candidate_idx, candidate_hs, candidate_boxes

    @staticmethod
    def assert_stage_b_v15_exact_candidates(
        candidate_idx: torch.Tensor,
        candidate_boxes: torch.Tensor,
        *,
        exact_mask: torch.Tensor,
        expected_indices: torch.Tensor,
        expected_boxes: torch.Tensor,
        box_atol: float,
    ) -> None:
        """Require runtime Stage-A Top-K to match the reviewed candidate rows."""
        if (
            exact_mask.dtype != torch.bool
            or tuple(exact_mask.shape) != (candidate_idx.shape[0],)
        ):
            raise RuntimeError("exact Stage-A candidate mask has invalid shape/dtype")
        if tuple(expected_indices.shape) != tuple(candidate_idx.shape):
            raise RuntimeError("exact Stage-A candidate indices have invalid shape")
        if tuple(expected_boxes.shape) != tuple(candidate_boxes.shape):
            raise RuntimeError("exact Stage-A candidate boxes have invalid shape")
        if expected_indices.dtype != torch.int64:
            raise RuntimeError("exact Stage-A candidate indices must be int64")
        try:
            box_atol = float(box_atol)
        except (TypeError, ValueError) as error:
            raise RuntimeError("exact Stage-A candidate box tolerance is invalid") from error
        if not 0.0 <= box_atol <= 1.0e-3:
            raise RuntimeError("exact Stage-A candidate box tolerance is unsafe")
        if not bool(exact_mask.any().item()):
            return
        runtime_indices = candidate_idx[exact_mask]
        bound_indices = expected_indices.to(
            device=candidate_idx.device, dtype=torch.int64
        )[exact_mask]
        mismatch = runtime_indices != bound_indices
        if bool(mismatch.any().item()):
            rows = mismatch.any(dim=1).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "runtime Stage-A ordered Top-K differs from exact verified "
                f"candidate indices for exact-row offsets {rows}"
            )
        runtime_boxes = candidate_boxes[exact_mask].float()
        bound_boxes = expected_boxes.to(
            device=candidate_boxes.device, dtype=torch.float32
        )[exact_mask]
        if not bool(torch.isfinite(bound_boxes).all().item()):
            raise RuntimeError("exact verified candidate boxes are not finite")
        max_error = float((runtime_boxes - bound_boxes).abs().max().item())
        if max_error > box_atol:
            raise RuntimeError(
                "runtime Stage-A candidate boxes drifted from exact verified boxes: "
                f"max_abs_error={max_error:.8g}, atol={box_atol:.8g}"
            )

    @staticmethod
    def scatter_stage_b_v11_candidates(
        candidate_logits: torch.Tensor,
        candidate_score: torch.Tensor,
        candidate_idx: torch.Tensor,
        num_queries: int,
        expression_valid_mask: torch.Tensor,
        candidate_valid_mask: Optional[torch.Tensor] = None,
    ):
        """Scatter candidate scores without admitting non-candidates or invalid slots."""
        if candidate_logits.dim() != 3 or candidate_score.shape != candidate_logits.shape:
            raise ValueError("candidate logits and scores must both be (B,N,K)")
        if candidate_idx.shape != candidate_logits.shape[:2]:
            raise ValueError("candidate_idx must align with candidate logits (B,N)")
        if expression_valid_mask.shape != (
            candidate_logits.shape[0],
            candidate_logits.shape[2],
        ):
            raise ValueError("expression_valid_mask must be (B,K)")
        if int(num_queries) <= 0:
            raise ValueError("num_queries must be positive")
        if candidate_valid_mask is not None and tuple(candidate_valid_mask.shape) not in {
            tuple(candidate_idx.shape),
            tuple(candidate_logits.shape),
        }:
            raise ValueError(
                "candidate_valid_mask must be (B,N) or align with candidate "
                "logits (B,N,K)"
            )

        batch_size, _, slot_count = candidate_logits.shape
        scatter_idx = candidate_idx.unsqueeze(-1).expand(-1, -1, slot_count)
        dense_logits = candidate_logits.new_full(
            (batch_size, int(num_queries), slot_count),
            torch.finfo(candidate_logits.dtype).min,
        )
        dense_score = candidate_score.new_zeros(
            (batch_size, int(num_queries), slot_count)
        )
        dense_mask = torch.zeros(
            (batch_size, int(num_queries), slot_count),
            dtype=torch.bool,
            device=candidate_logits.device,
        )
        if candidate_valid_mask is None:
            compact_valid_slots = torch.ones_like(candidate_logits, dtype=torch.bool)
        else:
            compact_valid_slots = candidate_valid_mask.to(
                device=candidate_logits.device, dtype=torch.bool
            )
            if compact_valid_slots.dim() == 2:
                compact_valid_slots = compact_valid_slots.unsqueeze(-1).expand_as(
                    candidate_logits
                )
        dense_mask.scatter_(1, scatter_idx, compact_valid_slots)
        valid = expression_valid_mask.to(
            device=candidate_logits.device, dtype=torch.bool
        )[:, None, :]
        dense_logits.scatter_(
            1,
            scatter_idx,
            candidate_logits.masked_fill(
                ~compact_valid_slots, torch.finfo(candidate_logits.dtype).min
            ),
        )
        dense_score.scatter_(
            1, scatter_idx, candidate_score.masked_fill(~compact_valid_slots, 0.0)
        )
        dense_logits = dense_logits.masked_fill(
            ~valid, torch.finfo(dense_logits.dtype).min
        )
        dense_score = dense_score.masked_fill(~valid, 0.0)
        dense_mask &= valid
        return dense_logits, dense_score, dense_mask

    def _tokenize_stage_b_v11_captions(
        self,
        captions: Sequence[str],
        device: Optional[torch.device] = None,
    ):
        """Single tokenizer entry point for v11 scoring and pair-token diffs."""
        if not captions:
            raise ValueError("Stage B v11 requires at least one caption")
        if str(getattr(self.tokenizer, "padding_side", "right")) != "right":
            raise ValueError(
                "Stage B v11 requires right tokenizer padding so token positions "
                "are invariant to expression microbatch size"
            )
        tokenized = self.tokenizer(
            list(captions), padding="longest", return_tensors="pt"
        )
        return tokenized.to(device) if device is not None else tokenized

    def _build_stage_b_v11_pair_predicate_masks(
        self,
        expression_captions: Sequence[Sequence[str]],
        expression_valid_mask: torch.Tensor,
        device: torch.device,
    ):
        from .stage_b_fixed_text_scorer import (
            build_stage_b_pair_token_diff_masks_from_ids,
        )

        batch_size = len(expression_captions)
        if batch_size <= 0:
            raise ValueError("Stage B v11 expression batch must be non-empty")
        slot_count = len(expression_captions[0])
        if any(len(row) != slot_count for row in expression_captions):
            raise ValueError("Stage B v11 expression captions must be rectangular")
        if tuple(expression_valid_mask.shape) != (batch_size, slot_count):
            raise ValueError("Stage B v11 expression validity does not align with captions")
        if slot_count != 2:
            return (
                torch.zeros(
                    (batch_size, slot_count, self.max_text_len),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.zeros((batch_size,), dtype=torch.bool, device=device),
            )

        flat_captions = [caption for row in expression_captions for caption in row]
        tokenized = self._tokenize_stage_b_v11_captions(flat_captions)
        input_ids = tokenized["input_ids"][:, : self.max_text_len].contiguous()
        attention_mask = tokenized["attention_mask"][
            :, : self.max_text_len
        ].bool().contiguous()

        flat_ids = input_ids.reshape(-1).tolist()
        flat_tokens = self.tokenizer.convert_ids_to_tokens(flat_ids)
        special_ids = {int(token_id) for token_id in self.tokenizer.all_special_ids}
        eligible = torch.as_tensor(
            [
                int(token_id) not in special_ids
                and any(char.isalnum() for char in str(token).replace("##", ""))
                for token_id, token in zip(flat_ids, flat_tokens)
            ],
            dtype=torch.bool,
        ).view_as(input_ids)
        eligible &= attention_mask

        predicate_mask, pair_valid = build_stage_b_pair_token_diff_masks_from_ids(
            input_ids.view(batch_size, slot_count, -1),
            attention_mask.view(batch_size, slot_count, -1),
            expression_valid_mask.detach().to(device="cpu", dtype=torch.bool),
            eligible.view(batch_size, slot_count, -1),
            max_text_len=self.max_text_len,
        )
        return predicate_mask.to(device=device), pair_valid.to(device=device)

    def _build_stage_b_v15_score_token_masks(
        self,
        expression_captions: Sequence[Sequence[str]],
        canonical_captions: Sequence[str],
        expression_valid_mask: torch.Tensor,
        device: torch.device,
        *,
        return_word_group_ids: bool = False,
    ):
        from .stage_b_fixed_text_scorer import (
            build_stage_b_noncanonical_token_masks_from_ids,
        )

        batch_size = len(expression_captions)
        if batch_size <= 0 or len(canonical_captions) != batch_size:
            raise ValueError("Stage B v15 canonical and expression batches must align")
        slot_count = len(expression_captions[0])
        if slot_count <= 0 or any(
            len(row) != slot_count for row in expression_captions
        ):
            raise ValueError("Stage B v15 expression captions must be rectangular")
        if tuple(expression_valid_mask.shape) != (batch_size, slot_count):
            raise ValueError("Stage B v15 expression validity does not align with captions")

        flat_captions = [
            str(caption) if str(caption).strip() else "object ."
            for row in expression_captions
            for caption in row
        ]
        canonical_captions = [
            str(caption) if str(caption).strip() else "object ."
            for caption in canonical_captions
        ]
        expression_tokens = self._tokenize_stage_b_v11_captions(flat_captions)
        canonical_tokens = self._tokenize_stage_b_v11_captions(canonical_captions)
        expression_ids = expression_tokens["input_ids"][:, : self.max_text_len]
        expression_attention = expression_tokens["attention_mask"][
            :, : self.max_text_len
        ].bool()
        canonical_ids = canonical_tokens["input_ids"][:, : self.max_text_len]
        canonical_attention = canonical_tokens["attention_mask"][
            :, : self.max_text_len
        ].bool()

        flat_ids = expression_ids.reshape(-1).tolist()
        flat_tokens = self.tokenizer.convert_ids_to_tokens(flat_ids)
        special_ids = {int(token_id) for token_id in self.tokenizer.all_special_ids}
        eligible = torch.as_tensor(
            [
                int(token_id) not in special_ids
                and any(char.isalnum() for char in str(token).replace("##", ""))
                for token_id, token in zip(flat_ids, flat_tokens)
            ],
            dtype=torch.bool,
        ).view_as(expression_ids)
        eligible &= expression_attention

        score_mask = build_stage_b_noncanonical_token_masks_from_ids(
            expression_ids.view(batch_size, slot_count, -1),
            expression_attention.view(batch_size, slot_count, -1),
            canonical_ids,
            canonical_attention,
            expression_valid_mask.detach().to(device="cpu", dtype=torch.bool),
            eligible.view(batch_size, slot_count, -1),
            max_text_len=self.max_text_len,
            fallback_to_eligible=not bool(
                getattr(self, "stage_b_dense_duty", False)
            ),
        )
        score_mask = score_mask.to(device=device)
        if not return_word_group_ids:
            return score_mask
        if not bool(getattr(self.tokenizer, "is_fast", False)):
            raise RuntimeError(
                "word-veto confidence requires a fast tokenizer with word_ids()"
            )
        word_group_ids = torch.full(
            (batch_size * slot_count, self.max_text_len),
            -1,
            dtype=torch.long,
        )
        for row_idx in range(batch_size * slot_count):
            row_word_ids = expression_tokens.word_ids(batch_index=row_idx)
            if row_word_ids is None or len(row_word_ids) != int(expression_ids.shape[1]):
                raise RuntimeError(
                    "fast tokenizer word_ids do not align with expression input_ids"
                )
            width = min(len(row_word_ids), self.max_text_len)
            if width > 0:
                word_group_ids[row_idx, :width] = torch.as_tensor(
                    [
                        -1 if word_id is None else int(word_id)
                        for word_id in row_word_ids[:width]
                    ],
                    dtype=torch.long,
                )
        word_group_ids = word_group_ids.view(
            batch_size, slot_count, self.max_text_len
        ).to(device=device)
        word_group_ids = word_group_ids.masked_fill(~score_mask, -1)
        if bool((score_mask & word_group_ids.lt(0)).any().item()):
            raise RuntimeError(
                "every noncanonical confidence token requires a lexical word id"
            )
        return score_mask, word_group_ids

    def _build_stage_b_v21_direct_trace_token_roles(
        self,
        expression_captions: Sequence[Sequence[str]],
        edit_traces: Sequence[Optional[Mapping[str, object]]],
        score_token_mask: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        from .stage_b_data_driven_score import build_direct_trace_token_roles

        batch_size = len(expression_captions)
        if batch_size <= 0 or len(edit_traces) != batch_size:
            raise ValueError("Stage B v21 edit traces must align with expressions")
        if any(len(row) != 2 for row in expression_captions):
            raise ValueError("Stage B v21 direct traces require paired expressions")
        if tuple(score_token_mask.shape) != (
            batch_size,
            2,
            self.max_text_len,
        ):
            raise ValueError("Stage B v21 score mask must have shape (B,2,T)")

        flat_captions = [caption for row in expression_captions for caption in row]
        tokenized = self._tokenize_stage_b_v11_captions(flat_captions)
        input_ids = tokenized["input_ids"][:, : self.max_text_len].contiguous()
        token_width = int(input_ids.shape[-1])
        input_ids = input_ids.view(batch_size, 2, token_width)
        roles = build_direct_trace_token_roles(
            self.tokenizer,
            expression_captions,
            [trace if isinstance(trace, Mapping) else {} for trace in edit_traces],
            input_ids,
            score_token_mask[:, :, :token_width],
            max_text_len=self.max_text_len,
            allow_incidental_edits=(
                self.stage_b_dense_duty_allow_incidental_trace_edits
            ),
        )
        padded_roles: Dict[str, torch.Tensor] = {"valid": roles["valid"]}
        for name in ("positive", "shared", "changed"):
            padded = torch.zeros_like(score_token_mask, device=device)
            padded[:, :, :token_width] = roles[name].to(device=device)
            padded_roles[name] = padded
        return padded_roles

    def _encode_stage_b_v11_captions(
        self,
        captions: Sequence[str],
        device: torch.device,
        *,
        apply_feat_map: bool = True,
    ):
        tokenized = self._tokenize_stage_b_v11_captions(captions, device=device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in tokenized:
                    tokenized[key] = tokenized[key][:, : self.max_text_len]

        if self.sub_sentence_present:
            tokenized_for_encoder = {
                key: value for key, value in tokenized.items() if key != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        with torch.no_grad():
            bert_output = self.bert(**tokenized_for_encoder)
            encoded_text = bert_output["last_hidden_state"]
            if apply_feat_map:
                encoded_text = self.feat_map(encoded_text)
        text_token_mask = tokenized["attention_mask"].bool()
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        phrase_token_mask = torch.zeros_like(text_token_mask, dtype=torch.bool)
        for batch_idx, phrase_rows in enumerate(cate_to_token_mask_list):
            phrase_rows = phrase_rows[:, : text_token_mask.shape[1]].to(
                device=device, dtype=torch.bool
            )
            if phrase_rows.numel() > 0:
                # Each scorer caption is one expression. Union any sentence
                # fragments so internal punctuation cannot silently drop text.
                phrase_token_mask[batch_idx, : phrase_rows.shape[1]] = phrase_rows.any(dim=0)

        return {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "input_ids": tokenized["input_ids"],
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }, phrase_token_mask

    def _build_stage_b_dense_duty_context(
        self,
        captions: Sequence[str],
        owner_indices: torch.Tensor,
        srcs: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        poss: Sequence[torch.Tensor],
    ):
        """Return frozen raw inputs for the private dense-duty text towers."""
        if owner_indices.dim() != 1 or owner_indices.dtype != torch.long:
            raise ValueError(
                "dense-duty owner_indices must be a one-dimensional long tensor"
            )
        if len(captions) != int(owner_indices.numel()):
            raise ValueError("dense-duty captions and owner_indices must align")
        if not srcs:
            raise ValueError("dense-duty scoring requires image feature maps")

        device = srcs[0].device
        owner_indices = owner_indices.to(device=device)
        text_dict, phrase_token_mask = self._encode_stage_b_v11_captions(
            captions, device, apply_feat_map=False
        )
        return {
            "bert_hidden": text_dict["encoded_text"].detach(),
            "text_token_mask": text_dict["text_token_mask"].detach(),
            "position_ids": text_dict["position_ids"].detach(),
            "text_self_attention_masks": text_dict[
                "text_self_attention_masks"
            ].detach(),
            "phrase_token_mask": phrase_token_mask.detach(),
            "srcs": tuple(
                src.index_select(0, owner_indices).detach() for src in srcs
            ),
            "masks": tuple(
                mask.index_select(0, owner_indices).detach() for mask in masks
            ),
            "poss": tuple(
                pos.index_select(0, owner_indices).detach() for pos in poss
            ),
        }

    def _build_stage_b_v11_context(
        self,
        captions: Sequence[str],
        owner_indices: torch.Tensor,
        srcs: Sequence[torch.Tensor],
        masks: Sequence[torch.Tensor],
        poss: Sequence[torch.Tensor],
    ):
        if owner_indices.dim() != 1 or owner_indices.dtype != torch.long:
            raise ValueError("Stage B v11 owner_indices must be a one-dimensional long tensor")
        if len(captions) != int(owner_indices.numel()):
            raise ValueError("Stage B v11 captions and owner_indices must have equal length")
        if not srcs:
            raise ValueError("Stage B v11 requires image feature maps")

        device = srcs[0].device
        owner_indices = owner_indices.to(device=device)
        text_dict, phrase_token_mask = self._encode_stage_b_v11_captions(
            captions, device
        )

        with torch.no_grad():
            src_flatten = []
            mask_flatten = []
            pos_flatten = []
            spatial_shapes = []
            selected_masks = []
            for level, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poss)):
                src = src.index_select(0, owner_indices)
                mask = mask.index_select(0, owner_indices)
                pos_embed = pos_embed.index_select(0, owner_indices)
                _, _, height, width = src.shape
                spatial_shapes.append((height, width))
                src_flatten.append(src.flatten(2).transpose(1, 2))
                mask_flatten.append(mask.flatten(1))
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                if self.transformer.num_feature_levels > 1 and self.transformer.level_embed is not None:
                    pos_embed = pos_embed + self.transformer.level_embed[level].view(1, 1, -1)
                pos_flatten.append(pos_embed)
                selected_masks.append(mask)

            src_flatten = torch.cat(src_flatten, dim=1)
            mask_flatten = torch.cat(mask_flatten, dim=1)
            pos_flatten = torch.cat(pos_flatten, dim=1)
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=device
            )
            level_start_index = torch.cat(
                (
                    spatial_shapes.new_zeros((1,)),
                    spatial_shapes.prod(1).cumsum(0)[:-1],
                )
            )
            valid_ratios = torch.stack(
                [self.transformer.get_valid_ratio(mask) for mask in selected_masks],
                dim=1,
            )
            memory, memory_text = self.transformer.encoder(
                src_flatten,
                pos=pos_flatten,
                level_start_index=level_start_index,
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                memory_text=text_dict["encoded_text"],
                text_attention_mask=~text_dict["text_token_mask"],
                position_ids=text_dict["position_ids"],
                text_self_attention_masks=text_dict["text_self_attention_masks"],
            )

        return {
            "memory": memory,
            "memory_key_padding_mask": mask_flatten,
            "memory_pos": pos_flatten,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
            "text_dict": {
                "encoded_text": memory_text,
                "text_token_mask": text_dict["text_token_mask"],
            },
            "phrase_token_mask": phrase_token_mask,
        }

    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        # Prefer explicit captions from kw (engine passes these in patch-only mode).
        if "captions" in kw and kw["captions"] is not None:
            captions = kw["captions"]
        elif targets is not None:
            captions = [t["caption"] for t in targets]
        else:
            raise KeyError("groundingdino.forward requires `captions` (kw) or targets with `caption`.")
        bs = len(captions)
        data_driven_geometry_diagnostics = kw.get(
            "stage_b_data_driven_score_geometry_diagnostics", False
        )
        return_main_phrase_mask = kw.get(
            "stage_b_data_driven_return_main_phrase_mask", False
        )
        if not isinstance(data_driven_geometry_diagnostics, bool):
            raise TypeError(
                "stage_b_data_driven_score_geometry_diagnostics must be boolean"
            )
        if not isinstance(return_main_phrase_mask, bool):
            raise TypeError(
                "stage_b_data_driven_return_main_phrase_mask must be boolean"
            )
        if self.training and (
            data_driven_geometry_diagnostics or return_main_phrase_mask
        ):
            raise RuntimeError("Stage-B score geometry diagnostics are eval-only")
        if (
            data_driven_geometry_diagnostics
            and self.stage_b_data_driven_score_heads is None
        ):
            raise RuntimeError(
                "Stage-B score geometry diagnostics require data-driven score heads"
            )
        # encoder texts

        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            samples.device
        )
        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }
        # Phrase-to-token masks derived from "." separators. Shape: list[Tensor(num_phrase_i, L)]
        # Keep both a padded tensor view and the original list for downstream training/eval.
        try:
            cate_to_token_mask_list = [m[:, : self.max_text_len] for m in cate_to_token_mask_list]
            max_phrase = max(int(m.shape[0]) for m in cate_to_token_mask_list) if cate_to_token_mask_list else 0
            phrase_mask = torch.zeros((bs, max_phrase), dtype=torch.bool, device=samples.device)
            phrase_to_token = torch.zeros((bs, max_phrase, self.max_text_len), dtype=torch.bool, device=samples.device)
            for b in range(bs):
                m = cate_to_token_mask_list[b].to(torch.bool)
                p = int(m.shape[0])
                if p > 0:
                    phrase_mask[b, :p] = True
                    phrase_to_token[b, :p, : m.shape[1]] = m
            text_dict["cate_to_token_mask_list"] = cate_to_token_mask_list
            text_dict["phrase_mask"] = phrase_mask
            text_dict["phrase_to_token_mask"] = phrase_to_token
        except Exception:
            pass


        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        patch_only = bool(kw.get("patch_only", self.patch_only))
        disable_patch_dn = bool(kw.get("disable_patch_dn", False))
        if (
            patch_only
            and self.training
            and (not disable_patch_dn)
            and (targets is not None)
            and (self.patch_dn_num_queries > 0)
            and (self.patch_dn_tgt is not None)
        ):
            # GT-guided queries from GT boxes (cxcywh normalized).
            bs = len(targets)
            dn_num = int(self.patch_dn_num_queries)
            dn_ref: List[torch.Tensor] = []
            for b in range(bs):
                gt = targets[b].get("boxes", None)
                if gt is None or (not torch.is_tensor(gt)) or gt.numel() == 0:
                    # Fallback: random boxes if GT is missing (should be rare).
                    dn_boxes = torch.rand((dn_num, 4), device=samples.device, dtype=torch.float32)
                    dn_boxes[:, 2:] = dn_boxes[:, 2:] * 0.2 + 0.05
                else:
                    gt = gt.to(samples.device).to(torch.float32)
                    if gt.dim() != 2 or gt.shape[-1] != 4:
                        raise ValueError(f"targets[b]['boxes'] must be (N,4) cxcywh, got {tuple(gt.shape)}")
                    idx = torch.randint(0, gt.shape[0], (dn_num,), device=samples.device)
                    dn_boxes = gt[idx].clone()
                    cxcy = dn_boxes[:, :2]
                    wh = dn_boxes[:, 2:].clamp(min=1e-3)
                    noise_xy = (torch.rand_like(cxcy) * 2 - 1.0) * self.patch_dn_box_noise_scale * wh
                    noise_wh = (torch.rand_like(wh) * 2 - 1.0) * self.patch_dn_box_noise_scale * wh
                    dn_boxes = torch.cat([cxcy + noise_xy, (wh + noise_wh).clamp(min=1e-3)], dim=-1)
                    dn_boxes = dn_boxes.clamp(0.0, 1.0)
                dn_ref.append(dn_boxes)
            dn_ref_t = torch.stack(dn_ref, dim=0).clamp(1e-4, 1.0 - 1e-4)  # (bs, dn, 4) sigmoid-space
            input_query_bbox = inverse_sigmoid(dn_ref_t)
            input_query_label = self.patch_dn_tgt.view(1, 1, -1).repeat(bs, dn_num, 1)
            attn_mask = None

        dense_duty_enabled = bool(
            self.stage_b_fixed_text_scorer is not None
            and getattr(self.stage_b_fixed_text_scorer, "is_dense_duty", False)
        )
        transformer_args = (
            srcs,
            masks,
            input_query_bbox,
            poss,
            input_query_label,
            attn_mask,
            text_dict,
        )
        transformer_output = (
            self.transformer(*transformer_args, return_predecoder_state=True)
            if dense_duty_enabled
            else self.transformer(*transformer_args)
        )
        if dense_duty_enabled:
            (
                hs,
                reference,
                hs_enc,
                ref_enc,
                init_box_proposal,
                stage_b_predecoder_tgt,
                stage_b_predecoder_reference,
            ) = transformer_output
        else:
            hs, reference, hs_enc, ref_enc, init_box_proposal = transformer_output
            stage_b_predecoder_tgt = None
            stage_b_predecoder_reference = None

        
        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # patch_only already computed above
        patches = kw.get("patches", None)
        patch_global_in = kw.get("patch_global", None)
        patch_mask_in = kw.get("patch_mask", None)
        if patch_only and (patches is None) and (patch_global_in is None):
            raise ValueError("patch_only=True requires `patches` or `patch_global` to be provided to model.forward().")

        outputs_class = None
        text_logits = None
        compute_text_logits = (not patch_only) or bool(
            kw.get("patch_only_compute_text_logits", self.patch_only_compute_text_logits)
        )
        if compute_text_logits:
            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs, text_dict)
                    for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )
            text_logits = outputs_class[-1]

        fused_logits = text_logits
        score_patch = None
        base_score_patch = None
        residual_score_patch = None
        aux_score_patch_list = None
        patch_gate = None
        patch_global = patch_global_in
        if patch_global is None and patches is not None:
            if self.patch_encoder is None:
                raise RuntimeError("Patch inputs were provided but this model was built without a patch branch.")
            patch_text_dict = text_dict if self.patch_encoder.gate_with_text else None
            patch_enc_out = self.patch_encoder(patches, text_dict=patch_text_dict, return_tokens=False)
            patch_global = patch_enc_out.get("patch_global", None)
            patch_gate = patch_enc_out.get("patch_gate", None)

        if patch_global is not None:
            if self.patch_logit_scale is None or self.query_proj_for_patch is None:
                raise RuntimeError("patch_global was provided but this model was built without a patch branch.")
            logit_scale = self.patch_logit_scale.exp().clamp(max=self.patch_logit_scale_max)
            query_proj = F.normalize(self.query_proj_for_patch(hs[-1]), dim=-1)
            patch_global = F.normalize(patch_global, dim=-1)
            # Support both single-patch (B,d) and multi-patch (B,K,d) inputs.
            if patch_global.dim() == 2:
                base_score_patch = logit_scale * torch.einsum(
                    "bqd,bd->bq", query_proj, patch_global
                )
                if self.stage_b_data_driven_patch_residual is not None:
                    residual_inputs = (query_proj, patch_global)
                    if bool(
                        getattr(
                            self.stage_b_data_driven_patch_residual,
                            "requires_base_score",
                            False,
                        )
                    ):
                        residual_inputs = residual_inputs + (base_score_patch,)
                    residual_score_patch = (
                        self.stage_b_data_driven_patch_residual(
                            *residual_inputs
                        )
                    )
                    score_patch = base_score_patch.detach() + (
                        logit_scale.detach() * residual_score_patch
                    )
                else:
                    score_patch = base_score_patch
                score_for_fuse = score_patch
                alpha_base = patch_gate  # (B,) or None
            elif patch_global.dim() == 3:
                base_score_patch = logit_scale * torch.einsum(
                    "bqd,bkd->bqk", query_proj, patch_global
                )
                if self.stage_b_data_driven_patch_residual is not None:
                    residual_inputs = (query_proj, patch_global)
                    if bool(
                        getattr(
                            self.stage_b_data_driven_patch_residual,
                            "requires_base_score",
                            False,
                        )
                    ):
                        residual_inputs = residual_inputs + (base_score_patch,)
                    residual_score_patch = (
                        self.stage_b_data_driven_patch_residual(
                            *residual_inputs
                        )
                    )
                    score_patch = base_score_patch.detach() + (
                        logit_scale.detach() * residual_score_patch
                    )
                else:
                    score_patch = base_score_patch
                if patch_mask_in is not None:
                    if (not torch.is_tensor(patch_mask_in)) or patch_mask_in.dim() != 2:
                        raise ValueError("patch_mask must be a bool tensor of shape (B,K) for multi-patch.")
                    invalid_patch = ~patch_mask_in[:, None, :].to(torch.bool)
                    score_patch = score_patch.masked_fill(invalid_patch, -100.0)
                    base_score_patch = base_score_patch.masked_fill(
                        invalid_patch, -100.0
                    )
                score_for_fuse = score_patch.max(dim=-1).values  # union score for visualization/logits
                alpha_base = patch_gate  # (B,K) or (B,) or None
                if alpha_base is not None and alpha_base.dim() == 2:
                    alpha_base = alpha_base[:, 0]
            else:
                raise ValueError(f"patch_global must be (B,d) or (B,K,d), got {tuple(patch_global.shape)}")

            if patch_only:
                # Keep pred_logits as (B,Q,1) for compatibility; patch-only loss uses pred_logits_patch.
                fused_logits = score_for_fuse.unsqueeze(-1)
            else:
                if text_logits is None:
                    raise RuntimeError("text_logits is None but patch_only=False; this should not happen.")
                if (
                    self.stage_b_u0_patch_rank_adapter is not None
                    or self.stage_b_data_driven_score_heads is not None
                    or self.stage_b_native_patch_category
                ):
                    # Dedicated score paths own patch/text fusion. Keep the
                    # canonical-query GDINO logits available only as diagnostics.
                    fused_logits = text_logits
                else:
                    if alpha_base is None:
                        alpha = torch.full((score_for_fuse.shape[0], 1), 0.5, device=score_for_fuse.device)
                    else:
                        alpha = alpha_base.unsqueeze(-1)
                    fused_logits = (1 - alpha) * text_logits + alpha * score_for_fuse.unsqueeze(-1)

            if self.aux_loss and patch_only:
                aux_score_patch_list = []
                for layer_hs in hs[:-1]:
                    aux_query_proj = F.normalize(self.query_proj_for_patch(layer_hs), dim=-1)
                    if patch_global.dim() == 2:
                        aux_score = logit_scale * torch.einsum("bqd,bd->bq", aux_query_proj, patch_global)
                    else:
                        aux_score = logit_scale * torch.einsum("bqd,bkd->bqk", aux_query_proj, patch_global)
                        if patch_mask_in is not None:
                            aux_score = aux_score.masked_fill(~patch_mask_in[:, None, :].to(torch.bool), -100.0)
                    aux_score_patch_list.append(aux_score)

        out = {
            "pred_logits": fused_logits,
            "pred_logits_text": text_logits,
            "pred_logits_patch": score_patch,
            "pred_logits_patch_base": base_score_patch,
            "pred_logits_patch_residual": residual_score_patch,
            "pred_boxes": outputs_coord_list[-1],
        }
        if return_main_phrase_mask:
            main_phrase_rows = text_dict.get("phrase_to_token_mask", None)
            if (
                not torch.is_tensor(main_phrase_rows)
                or main_phrase_rows.dtype != torch.bool
                or main_phrase_rows.dim() != 3
                or int(main_phrase_rows.shape[0]) != bs
                or int(main_phrase_rows.shape[2]) != self.max_text_len
            ):
                raise RuntimeError(
                    "main-forward phrase-to-token mask is unavailable or malformed"
                )
            main_phrase_token_mask = main_phrase_rows.any(dim=1)
            if bool((~main_phrase_token_mask.any(dim=1)).any().item()):
                raise RuntimeError(
                    "every diagnostic main-forward caption requires a phrase token"
                )
            out["stage_b_diagnostic_main_phrase_token_mask"] = (
                main_phrase_token_mask
            )
        if self.stage_b_gdino_score_adapter is not None:
            if patch_only:
                raise RuntimeError(
                    "stage_b_gdino_score_adapter is restricted to ordinary non-patch GroundingDINO"
                )
            if text_logits is None:
                raise RuntimeError(
                    "stage_b_gdino_score_adapter requires final GroundingDINO token logits"
                )
            generated_phrase_mask = text_dict.get("phrase_to_token_mask", None)
            if (
                not torch.is_tensor(generated_phrase_mask)
                or generated_phrase_mask.dim() != 3
                or int(generated_phrase_mask.shape[0]) != bs
            ):
                raise RuntimeError(
                    "stage_b_gdino_score_adapter requires generated full-expression token masks"
                )
            expression_token_mask = generated_phrase_mask.any(dim=1)
            from .stage_b_gdino_score_adapter import (
                aggregate_gdino_full_expression_score,
            )

            base_score = aggregate_gdino_full_expression_score(
                text_logits, expression_token_mask
            )
            adapter_output = self.stage_b_gdino_score_adapter(
                hs[-1], base_score
            )
            out["stage_b_gdino_base_score"] = adapter_output["base_score"]
            out["stage_b_gdino_rank_residual"] = adapter_output[
                "rank_residual"
            ]
            out["stage_b_gdino_rank_score"] = adapter_output["rank_score"]
            out["stage_b_gdino_confidence_gate"] = adapter_output[
                "confidence_gate"
            ]
            out["stage_b_gdino_confidence_score"] = adapter_output[
                "confidence_score"
            ]
            out["stage_b_gdino_expression_token_mask"] = expression_token_mask
            deployed_teacher_rank_score = adapter_output["rank_score"]
            if self.stage_b_u0_gate_aligned_rank_residual is not None:
                d12_output = self.stage_b_u0_gate_aligned_rank_residual(
                    adapter_output["rank_feature"],
                    adapter_output["rank_score"],
                    adapter_output["candidate_mask"],
                )
                deployed_teacher_rank_score = d12_output["rank_score"]
                out["stage_b_u0_d12_teacher_rank_score"] = d12_output[
                    "teacher_rank_score"
                ]
                out["stage_b_u0_d12_rank_residual"] = d12_output[
                    "rank_residual"
                ]
                out["stage_b_u0_d12_rank_score"] = d12_output["rank_score"]
            if self.stage_b_u0_patch_rank_adapter is not None:
                if score_patch is None:
                    raise RuntimeError("Stage-B U0 requires patch scores")
                u0_patch_score = score_patch
                if u0_patch_score.dim() == 3:
                    if int(u0_patch_score.shape[-1]) != 1:
                        raise RuntimeError(
                            "Stage-B U0 requires exactly one support-patch slot"
                        )
                    u0_patch_score = u0_patch_score[..., 0]
                u0_output = self.stage_b_u0_patch_rank_adapter(
                    u0_patch_score,
                    deployed_teacher_rank_score,
                    adapter_output["candidate_mask"],
                )
                out["stage_b_u0_teacher_rank_score"] = u0_output[
                    "teacher_rank_score"
                ]
                out["stage_b_u0_patch_rank_residual"] = u0_output[
                    "patch_rank_residual"
                ]
                out["stage_b_u0_learned_patch_rank_residual"] = u0_output[
                    "learned_patch_rank_residual"
                ]
                out["stage_b_u1_direct_patch_rank_residual"] = u0_output[
                    "direct_patch_rank_residual"
                ]
                out["stage_b_u1_direct_patch_gain"] = u0_output[
                    "direct_patch_gain"
                ]
                out["stage_b_u0_rank_score"] = u0_output["rank_score"]
                out["stage_b_u0_candidate_mask"] = u0_output[
                    "candidate_mask"
                ]
                if "category_gate_eligible_mask" in u0_output:
                    out["stage_b_u0_pre_category_gate_rank_score"] = u0_output[
                        "pre_category_gate_rank_score"
                    ]
                    out["stage_b_u0_category_gate_eligible_mask"] = u0_output[
                        "category_gate_eligible_mask"
                    ]
                    out["stage_b_u0_category_gate_patch_score"] = u0_output[
                        "category_gate_patch_score"
                    ]
                    if self.stage_b_u0_gate_aligned_patch_residual is not None:
                        d13_support = patch_global
                        if d13_support.dim() == 3:
                            if int(d13_support.shape[1]) != 1:
                                raise RuntimeError(
                                    "D13 requires exactly one support-patch slot"
                                )
                            d13_support = d13_support[:, 0]
                        d13_output = self.stage_b_u0_gate_aligned_patch_residual(
                            query_proj,
                            d13_support,
                            u0_output["category_gate_patch_score"],
                            deployed_teacher_rank_score,
                            u0_output["candidate_mask"],
                        )
                        out["stage_b_u0_d13_teacher_patch_score"] = d13_output[
                            "teacher_patch_score"
                        ]
                        out["stage_b_u0_d13_patch_residual"] = d13_output[
                            "patch_residual"
                        ]
                        out["stage_b_u0_d13_patch_score"] = d13_output[
                            "patch_score"
                        ]
                        out["stage_b_u0_d13_teacher_rank_score"] = d13_output[
                            "teacher_rank_score"
                        ]
                        out[
                            "stage_b_u0_d13_teacher_gated_rank_score"
                        ] = d13_output["teacher_gated_rank_score"]
                        out[
                            "stage_b_u0_d13_teacher_eligible_mask"
                        ] = d13_output["teacher_eligible_mask"]
                        out["stage_b_u0_rank_score"] = d13_output["rank_score"]
                        out[
                            "stage_b_u0_category_gate_eligible_mask"
                        ] = d13_output["eligible_mask"]
                        out[
                            "stage_b_u0_category_gate_patch_score"
                        ] = d13_output["patch_score"]
        if self.stage_b_data_driven_score_heads is not None:
            expression_captions = kw.get(
                "stage_b_data_driven_expression_captions", None
            )
            if not isinstance(expression_captions, (list, tuple)) or len(
                expression_captions
            ) != bs:
                raise ValueError(
                    "data-driven scoring expressions must align with the "
                    "canonical-query image batch"
                )
            paired_expressions = bool(expression_captions) and all(
                isinstance(row, (list, tuple)) for row in expression_captions
            )
            if paired_expressions:
                expression_slots = len(expression_captions[0])
                if expression_slots <= 0 or any(
                    len(row) != expression_slots for row in expression_captions
                ):
                    raise ValueError(
                        "data-driven expression captions must be a non-empty "
                        "rectangular batch"
                    )
                flat_expression_captions = [
                    caption for row in expression_captions for caption in row
                ]
            elif all(isinstance(caption, str) for caption in expression_captions):
                expression_slots = 1
                flat_expression_captions = list(expression_captions)
            else:
                raise ValueError(
                    "data-driven expressions must be either B strings or a "
                    "rectangular BxK string batch"
                )
            if any(
                not isinstance(caption, str) or not caption.strip()
                for caption in flat_expression_captions
            ):
                raise ValueError(
                    "every data-driven full expression must be non-empty"
                )
            expression_text, expression_token_mask = (
                self._encode_stage_b_v11_captions(
                    flat_expression_captions, device=samples.device
                )
            )
            scorer_hs = hs[-1]
            scorer_patch = score_patch
            scorer_image_owners = torch.arange(
                bs, device=scorer_hs.device, dtype=torch.long
            )
            if expression_slots > 1:
                scorer_image_owners = scorer_image_owners.repeat_interleave(
                    expression_slots
                )
                scorer_hs = (
                    scorer_hs[:, None]
                    .expand(
                        bs,
                        expression_slots,
                        int(scorer_hs.shape[1]),
                        int(scorer_hs.shape[2]),
                    )
                    .reshape(
                        bs * expression_slots,
                        int(scorer_hs.shape[1]),
                        int(scorer_hs.shape[2]),
                    )
                )
                if scorer_patch is not None:
                    scorer_patch = (
                        scorer_patch[:, None]
                        .expand(bs, expression_slots, *scorer_patch.shape[1:])
                        .reshape(bs * expression_slots, *scorer_patch.shape[1:])
                    )
            diagnostic_geometry_outputs = {}
            if data_driven_geometry_diagnostics:
                from .stage_b_data_driven_score import (
                    groundingdino_raw_dot_phrase_geometry,
                )

                with torch.no_grad():
                    diagnostic_geometry_outputs["raw_expression_native"] = (
                        groundingdino_raw_dot_phrase_geometry(
                            scorer_hs,
                            expression_text["encoded_text"],
                            expression_text["text_token_mask"],
                            expression_token_mask,
                            max_text_len=self.max_text_len,
                        )
                    )
                    expression_context = self._build_stage_b_v11_context(
                        flat_expression_captions,
                        scorer_image_owners,
                        srcs,
                        masks,
                        poss,
                    )
                    diagnostic_geometry_outputs[
                        "encoder_fused_expression_fixed_query_native"
                    ] = groundingdino_raw_dot_phrase_geometry(
                        scorer_hs,
                        expression_context["text_dict"]["encoded_text"],
                        expression_context["text_dict"]["text_token_mask"],
                        expression_context["phrase_token_mask"],
                        max_text_len=self.max_text_len,
                    )
            data_driven_output = self.stage_b_data_driven_score_heads(
                scorer_hs,
                expression_text["encoded_text"],
                expression_token_mask,
                patch_score=scorer_patch,
                query_boxes=out["pred_boxes"],
                image_features=srcs,
                image_masks=masks,
                image_owner_indices=scorer_image_owners,
            )
            if expression_slots > 1:
                query_count = int(scorer_hs.shape[1])
                token_count = int(expression_token_mask.shape[1])
                query_token_keys = {
                    "text_rank_token_logits",
                    "confidence_token_logits",
                }
                query_keys = {
                    "text_rank_score",
                    "rank_score",
                    "confidence_base_score",
                    "confidence_score",
                    "candidate_mask",
                    "category_gate_eligible_mask",
                    "category_gate_patch_score",
                }
                for key in query_token_keys:
                    value = data_driven_output[key]
                    data_driven_output[key] = (
                        value.reshape(
                            bs, expression_slots, query_count, token_count
                        ).permute(0, 2, 1, 3).contiguous()
                    )
                for key in query_keys:
                    value = data_driven_output[key]
                    data_driven_output[key] = (
                        value.reshape(bs, expression_slots, query_count)
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                data_driven_output["confidence_gate"] = data_driven_output[
                    "confidence_gate"
                ].reshape(bs, expression_slots)
                data_driven_output["expression_token_mask"] = data_driven_output[
                    "expression_token_mask"
                ].reshape(bs, expression_slots, token_count)
            for key, value in data_driven_output.items():
                out[f"stage_b_data_driven_{key}"] = value
            for route_name, geometry_output in diagnostic_geometry_outputs.items():
                geometry_logits = geometry_output["token_logits"]
                geometry_mask = geometry_output["expression_token_mask"]
                geometry_score = geometry_output["score"]
                if expression_slots > 1:
                    geometry_logits = (
                        geometry_logits.reshape(
                            bs,
                            expression_slots,
                            int(scorer_hs.shape[1]),
                            self.max_text_len,
                        )
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                    geometry_mask = geometry_mask.reshape(
                        bs, expression_slots, self.max_text_len
                    )
                    geometry_score = (
                        geometry_score.reshape(
                            bs, expression_slots, int(scorer_hs.shape[1])
                        )
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                prefix = f"stage_b_data_driven_{route_name}"
                out[f"{prefix}_token_logits"] = geometry_logits
                out[f"{prefix}_expression_token_mask"] = geometry_mask
                out[f"{prefix}_score"] = geometry_score
            expression_input_ids = expression_text["input_ids"]
            if expression_slots > 1:
                expression_input_ids = expression_input_ids.reshape(
                    bs, expression_slots, int(expression_input_ids.shape[1])
                )
            out["stage_b_data_driven_expression_input_ids"] = (
                expression_input_ids
            )
        stage_b_v7_verifier_captions = kw.get("stage_b_v7_verifier_captions", None)
        stage_b_v11_expression_captions = kw.get(
            "stage_b_v11_expression_captions", None
        )
        stage_b_v11_expression_valid_mask = kw.get(
            "stage_b_v11_expression_valid_mask", None
        )
        stage_b_v21_edit_traces = kw.get("stage_b_v21_edit_traces", None)
        stage_b_v11_v7_compat = False
        if (
            self.stage_b_fixed_text_scorer is not None
            and stage_b_v11_expression_captions is None
            and stage_b_v7_verifier_captions is not None
        ):
            # Existing Stage-B evaluation tools issue one verifier caption per
            # forward. Preserve that contract without rebuilding the old verifier.
            stage_b_v11_expression_captions = [
                [str(caption)] for caption in stage_b_v7_verifier_captions
            ]
            stage_b_v11_expression_valid_mask = torch.tensor(
                [[True] for _ in stage_b_v7_verifier_captions],
                dtype=torch.bool,
                device=out["pred_boxes"].device,
            )
            stage_b_v11_v7_compat = True

        stage_b_v11_ran = stage_b_v11_expression_captions is not None
        if (
            bool(kw.get("return_stage_b_v7_features", False))
            or stage_b_v7_verifier_captions is not None
            or stage_b_v11_ran
        ):
            out["hs"] = hs[-1]
            out["patch_score"] = score_patch.sigmoid() if score_patch is not None else None
            out["stage_b_v7_roi_feature_map"] = srcs[0]
            out["stage_b_v7_roi_feature_mask"] = masks[0]

        if stage_b_v11_ran:
            if self.stage_b_fixed_text_scorer is None:
                raise RuntimeError(
                    "stage_b_v11_expression_captions were provided but "
                    "stage_b_fixed_text_scorer is not built"
                )
            if score_patch is None:
                raise RuntimeError("Stage B v11 requires Stage-A patch scores")
            if stage_b_v11_expression_valid_mask is None:
                slot_count = len(stage_b_v11_expression_captions[0])
                stage_b_v11_expression_valid_mask = torch.ones(
                    (len(stage_b_v11_expression_captions), slot_count),
                    dtype=torch.bool,
                    device=out["pred_boxes"].device,
                )
            else:
                stage_b_v11_expression_valid_mask = torch.as_tensor(
                    stage_b_v11_expression_valid_mask,
                    dtype=torch.bool,
                    device=out["pred_boxes"].device,
                )
            (
                stage_b_v11_predicate_token_mask,
                stage_b_v11_predicate_pair_valid,
            ) = self._build_stage_b_v11_pair_predicate_masks(
                stage_b_v11_expression_captions,
                stage_b_v11_expression_valid_mask,
                out["pred_boxes"].device,
            )
            stage_b_v15_score_token_mask = None
            stage_b_v15_score_word_group_ids = None
            if self.stage_b_v15_exclude_canonical_from_score:
                score_mask_result = self._build_stage_b_v15_score_token_masks(
                    stage_b_v11_expression_captions,
                    captions,
                    stage_b_v11_expression_valid_mask,
                    out["pred_boxes"].device,
                    return_word_group_ids=(
                        dense_duty_enabled
                        and self.stage_b_dense_duty_confidence_word_groups
                    ),
                )
                if isinstance(score_mask_result, tuple):
                    (
                        stage_b_v15_score_token_mask,
                        stage_b_v15_score_word_group_ids,
                    ) = score_mask_result
                else:
                    stage_b_v15_score_token_mask = score_mask_result
            stage_b_v21_direct_trace_roles = None
            if stage_b_v21_edit_traces is not None:
                if not dense_duty_enabled:
                    raise RuntimeError(
                        "Stage B v21 direct trace roles are reserved for dense-duty"
                    )
                if stage_b_v15_score_token_mask is None:
                    raise RuntimeError(
                        "Stage B v21 direct trace roles require canonical-excluded "
                        "score masks"
                    )
                stage_b_v21_direct_trace_roles = (
                    self._build_stage_b_v21_direct_trace_token_roles(
                        stage_b_v11_expression_captions,
                        stage_b_v21_edit_traces,
                        stage_b_v15_score_token_mask,
                        out["pred_boxes"].device,
                    )
                )

            assert_fixed_candidates = bool(
                kw.get("stage_b_v11_assert_fixed_candidates", False)
            )
            stage_a_boxes_snapshot = (
                out["pred_boxes"].detach().clone()
                if assert_fixed_candidates
                else None
            )

            if dense_duty_enabled:
                if stage_b_predecoder_tgt is None:
                    raise RuntimeError(
                        "dense-duty Stage B requires the transformer pre-decoder query state"
                    )
                if stage_b_predecoder_tgt.shape != hs[-1].shape:
                    raise RuntimeError(
                        "dense-duty pre-decoder queries must align with final Stage-A queries"
                    )
                candidate_query_source = stage_b_predecoder_tgt
            else:
                candidate_query_source = hs[-1]

            candidate_idx, candidate_hs, candidate_boxes = (
                self.select_stage_b_v11_candidates(
                    candidate_query_source,
                    out["pred_boxes"],
                    score_patch,
                    int(
                        kw.get(
                            "stage_b_v11_candidate_topk",
                            self.stage_b_v11_candidate_topk,
                        )
                    ),
                )
            )
            exact_candidate_values = (
                kw.get("stage_b_v15_exact_candidate_mask", None),
                kw.get("stage_b_v15_exact_candidate_indices", None),
                kw.get("stage_b_v15_exact_candidate_boxes", None),
                kw.get("stage_b_v15_exact_candidate_box_atol", None),
            )
            if any(value is not None for value in exact_candidate_values):
                if any(value is None for value in exact_candidate_values):
                    raise RuntimeError(
                        "exact Stage-A candidate replay requires mask, indices, "
                        "boxes, and box tolerance together"
                    )
                self.assert_stage_b_v15_exact_candidates(
                    candidate_idx,
                    candidate_boxes,
                    exact_mask=exact_candidate_values[0],
                    expected_indices=exact_candidate_values[1],
                    expected_boxes=exact_candidate_values[2],
                    box_atol=exact_candidate_values[3],
                )
            candidate_idx_snapshot = (
                candidate_idx.clone() if assert_fixed_candidates else None
            )
            patch_candidate_source = score_patch
            if patch_candidate_source.dim() == 3:
                if int(patch_candidate_source.shape[-1]) != 1:
                    raise RuntimeError(
                        "Stage B v15 patch fusion requires one localization patch slot"
                    )
                patch_candidate_source = patch_candidate_source[..., 0]
            candidate_patch_logits = torch.gather(
                patch_candidate_source.detach(), 1, candidate_idx
            )

            def context_provider(expression_chunk, owner_indices):
                return self._build_stage_b_v11_context(
                    expression_chunk,
                    owner_indices,
                    srcs,
                    masks,
                    poss,
                )

            def raw_context_provider(expression_chunk, owner_indices):
                return self._build_stage_b_dense_duty_context(
                    expression_chunk,
                    owner_indices,
                    srcs,
                    masks,
                    poss,
                )

            scorer_kwargs = dict(
                candidate_hs=candidate_hs,
                candidate_boxes=candidate_boxes,
                expression_captions=stage_b_v11_expression_captions,
                expression_valid_mask=stage_b_v11_expression_valid_mask,
                expression_predicate_token_mask=stage_b_v11_predicate_token_mask,
                expression_score_token_mask=stage_b_v15_score_token_mask,
                candidate_patch_logits=candidate_patch_logits,
                expression_microbatch=kw.get(
                    "stage_b_v11_expression_microbatch", None
                ),
            )
            if dense_duty_enabled:
                scorer_kwargs.update(
                    candidate_indices=candidate_idx,
                    raw_context_provider=raw_context_provider,
                    expression_score_word_group_ids=(
                        stage_b_v15_score_word_group_ids
                    ),
                )
            else:
                scorer_kwargs["context_provider"] = context_provider
            scorer_out = self.stage_b_fixed_text_scorer(**scorer_kwargs)
            if assert_fixed_candidates:
                if not torch.equal(out["pred_boxes"].detach(), stage_a_boxes_snapshot):
                    raise RuntimeError(
                        "Stage B v11 scorer changed frozen Stage-A pred_boxes"
                    )
                if not torch.equal(candidate_idx, candidate_idx_snapshot):
                    raise RuntimeError(
                        "Stage B v11 scorer changed frozen Stage-A candidate indices"
                    )
                expected_candidate_boxes = torch.gather(
                    stage_a_boxes_snapshot,
                    1,
                    candidate_idx_snapshot.unsqueeze(-1).expand(-1, -1, 4),
                )
                if not torch.equal(candidate_boxes, expected_candidate_boxes):
                    raise RuntimeError(
                        "Stage B v11 candidate boxes are not a bitwise gather of Stage-A boxes"
                    )
            compact_candidate_mask = scorer_out.get("candidate_eligible_mask")
            dense_logits, dense_score, dense_candidate_mask = (
                self.scatter_stage_b_v11_candidates(
                    scorer_out["final_validity_logits"],
                    scorer_out["final_score"],
                    candidate_idx,
                    int(out["pred_boxes"].shape[1]),
                    scorer_out["expression_valid_mask"],
                    candidate_valid_mask=compact_candidate_mask,
                )
            )
            dense_rank_logits, dense_rank_score, _ = (
                self.scatter_stage_b_v11_candidates(
                    scorer_out["final_phrase_logits"],
                    scorer_out["final_rank_score"],
                    candidate_idx,
                    int(out["pred_boxes"].shape[1]),
                    scorer_out["expression_valid_mask"],
                    candidate_valid_mask=compact_candidate_mask,
                )
            )
            out.update(
                {
                    "stage_b_v11_candidate_idx": candidate_idx,
                    "stage_b_v11_candidate_boxes": candidate_boxes,
                    "stage_b_v11_layer_token_logits": scorer_out[
                        "layer_token_logits"
                    ],
                    "stage_b_v11_final_token_logits": scorer_out[
                        "final_token_logits"
                    ],
                    "stage_b_v11_layer_phrase_logits": scorer_out[
                        "layer_phrase_logits"
                    ],
                    "stage_b_v11_final_phrase_logits": scorer_out[
                        "final_phrase_logits"
                    ],
                    "stage_b_v14_layer_validity_logits": scorer_out[
                        "layer_validity_logits"
                    ],
                    "stage_b_v14_final_validity_logits": scorer_out[
                        "final_validity_logits"
                    ],
                    "stage_b_v15_layer_validity_gate_logits": scorer_out[
                        "layer_validity_gate_logits"
                    ],
                    "stage_b_v15_final_validity_gate_logits": scorer_out[
                        "final_validity_gate_logits"
                    ],
                    "stage_b_v15_final_confidence_base_logits": scorer_out[
                        "final_confidence_base_logits"
                    ],
                    "stage_b_v15_score_token_mask": scorer_out[
                        "score_token_mask"
                    ],
                    "stage_b_v15_candidate_patch_logits": candidate_patch_logits,
                    "stage_b_v15_final_rank_score": scorer_out[
                        "final_rank_score"
                    ],
                    "stage_b_v11_layer_predicate_logits": scorer_out[
                        "layer_predicate_logits"
                    ],
                    "stage_b_v11_final_predicate_logits": scorer_out[
                        "final_predicate_logits"
                    ],
                    "stage_b_v11_predicate_token_mask": scorer_out[
                        "predicate_token_mask"
                    ],
                    "stage_b_v11_predicate_valid_mask": scorer_out[
                        "predicate_valid_mask"
                    ],
                    "stage_b_v11_predicate_pair_valid": (
                        stage_b_v11_predicate_pair_valid
                        & scorer_out["predicate_valid_mask"].all(dim=-1)
                    ),
                    "stage_b_v11_final_score": scorer_out["final_score"],
                    "stage_b_v11_expression_valid_mask": scorer_out[
                        "expression_valid_mask"
                    ],
                    "stage_b_v11_dense_logits": dense_logits,
                    "stage_b_v11_dense_score": dense_score,
                    "stage_b_v15_dense_rank_logits": dense_rank_logits,
                    "stage_b_v15_dense_rank_score": dense_rank_score,
                    "stage_b_v11_candidate_mask": dense_candidate_mask,
                    "stage_b_v11_fixed_candidate_asserted": torch.as_tensor(
                        assert_fixed_candidates,
                        dtype=torch.bool,
                        device=candidate_idx.device,
                    ),
                }
            )
            if dense_duty_enabled:
                out["stage_b_dense_duty_predecoder_reference"] = (
                    stage_b_predecoder_reference
                )
                out["stage_b_dense_duty_candidate_eligible_mask"] = (
                    compact_candidate_mask
                )
                for scorer_key, output_key in (
                    (
                        "final_rank_token_logits",
                        "stage_b_dense_duty_final_rank_token_logits",
                    ),
                    (
                        "final_frozen_rank_full_expression_global_logits",
                        "stage_b_dense_duty_frozen_rank_full_expression_global_logits",
                    ),
                    (
                        "final_confidence_token_logits",
                        "stage_b_dense_duty_final_confidence_token_logits",
                    ),
                    (
                        "final_confidence_token_residual_logits",
                        "stage_b_dense_duty_final_confidence_token_residual_logits",
                    ),
                    (
                        "final_global_confidence_logits",
                        "stage_b_dense_duty_global_confidence_logits",
                    ),
                    (
                        "final_confidence_pool_absolute_logits",
                        "stage_b_dense_duty_confidence_pool_absolute_logits",
                    ),
                    (
                        "final_positive_confidence_logits",
                        "stage_b_dense_duty_positive_confidence_logits",
                    ),
                    (
                        "final_negative_confidence_logits",
                        "stage_b_dense_duty_negative_confidence_logits",
                    ),
                    (
                        "final_positive_global_confidence_logits",
                        "stage_b_dense_duty_positive_global_confidence_logits",
                    ),
                    (
                        "final_negative_global_confidence_logits",
                        "stage_b_dense_duty_negative_global_confidence_logits",
                    ),
                    (
                        "final_global_veto_raw_logits",
                        "stage_b_dense_duty_global_veto_raw_logits",
                    ),
                    (
                        "final_global_veto_depth",
                        "stage_b_dense_duty_global_veto_depth",
                    ),
                    (
                        "final_reference_global_confidence_logits",
                        "stage_b_dense_duty_reference_global_confidence_logits",
                    ),
                    (
                        "final_reference_base_logits",
                        "stage_b_dense_duty_reference_base_logits",
                    ),
                    (
                        "final_confidence_base_logits",
                        "stage_b_dense_duty_confidence_base_logits",
                    ),
                    (
                        "final_confidence_delta_logits",
                        "stage_b_dense_duty_confidence_delta_logits",
                    ),
                    (
                        "final_confidence_mismatch_gate",
                        "stage_b_dense_duty_confidence_mismatch_gate",
                    ),
                    (
                        "final_confidence_entailment_probability",
                        "stage_b_dense_duty_confidence_entailment_probability",
                    ),
                    (
                        "final_deployed_candidate_veto_depth",
                        "stage_b_dense_duty_candidate_veto_depth",
                    ),
                    (
                        "final_deployed_candidate_veto_gate",
                        "stage_b_dense_duty_candidate_veto_gate",
                    ),
                    (
                        "final_confidence_raw_mismatch_gate",
                        "stage_b_dense_duty_confidence_raw_mismatch_gate",
                    ),
                    (
                        "final_confidence_deployed_routing_gate",
                        "stage_b_dense_duty_confidence_deployed_routing_gate",
                    ),
                    (
                        "final_confidence_deployed_routing_residual",
                        "stage_b_dense_duty_confidence_deployed_routing_residual",
                    ),
                    (
                        "final_confidence_veto_coverage",
                        "stage_b_dense_duty_confidence_veto_coverage",
                    ),
                    (
                        "final_confidence_veto_sample_gate",
                        "stage_b_dense_duty_confidence_veto_sample_gate",
                    ),
                    (
                        "final_confidence_veto_carrier_index",
                        "stage_b_dense_duty_final_confidence_veto_carrier_index",
                    ),
                    (
                        "final_confidence_veto_absolute_ceiling",
                        "stage_b_dense_duty_confidence_veto_absolute_ceiling",
                    ),
                ):
                    if scorer_key in scorer_out:
                        out[output_key] = scorer_out[scorer_key]
                if stage_b_v15_score_word_group_ids is not None:
                    out["stage_b_dense_duty_score_word_group_ids"] = (
                        stage_b_v15_score_word_group_ids
                    )
                if stage_b_v21_direct_trace_roles is not None:
                    out.update(
                        {
                            "stage_b_v21_positive_token_mask": (
                                stage_b_v21_direct_trace_roles["positive"]
                            ),
                            "stage_b_v21_shared_token_mask": (
                                stage_b_v21_direct_trace_roles["shared"]
                            ),
                            "stage_b_v21_changed_token_mask": (
                                stage_b_v21_direct_trace_roles["changed"]
                            ),
                            "stage_b_v21_direct_trace_valid": (
                                stage_b_v21_direct_trace_roles["valid"]
                            ),
                        }
                    )

            alias_slice = slice(0, 1) if stage_b_v11_v7_compat else slice(None)
            out["stage_b_v7_candidate_mask"] = dense_candidate_mask[
                ..., alias_slice
            ]
            out["stage_b_v7_final_logits"] = dense_logits[..., alias_slice]
            out["stage_b_v7_final_score"] = dense_score[..., alias_slice]
            out["stage_b_v7_predicate_logits"] = dense_logits[..., alias_slice]
            out["stage_b_v7_predicate_score"] = dense_score[..., alias_slice]
            if out["patch_score"] is not None:
                patch_alias = out["patch_score"]
                if patch_alias.dim() == 2:
                    patch_alias = patch_alias.unsqueeze(-1)
                out["stage_b_v7_expanded_patch_score"] = patch_alias

        if stage_b_v7_verifier_captions is not None and not stage_b_v11_ran:
            if self.stage_b_verifier is None:
                raise RuntimeError("stage_b_v7_verifier_captions were provided but stage_b_verifier is not built.")
            verifier_out = self.stage_b_verifier(
                query_feats=out["hs"].detach(),
                boxes=out["pred_boxes"].detach(),
                roi_feature_map=out["stage_b_v7_roi_feature_map"].detach(),
                roi_feature_mask=out["stage_b_v7_roi_feature_mask"],
                predicate_text=stage_b_v7_verifier_captions,
                phrase_to_token_mask=kw.get("phrase_to_token_mask", None),
                canonical_to_token_mask=kw.get("canonical_to_token_mask", None),
                patch_mask=patch_mask_in,
            )
            out["stage_b_v7_predicate_logits"] = verifier_out["predicate_logits"]
            out["stage_b_v7_predicate_token_logits"] = verifier_out["predicate_token_logits"]
            out["stage_b_v7_predicate_score"] = verifier_out["predicate_logits"].sigmoid()
            if out["patch_score"] is not None:
                verifier_scores = self.stage_b_verifier.score_candidates(
                    verifier_out["predicate_logits"],
                    out["patch_score"],
                    pair_stride=kw.get("verifier_pair_stride", None),
                )
                out["stage_b_v7_candidate_mask"] = verifier_scores["candidate_mask"]
                out["stage_b_v7_expanded_patch_score"] = verifier_scores["expanded_patch_score"]
                out["stage_b_v7_final_logits"] = verifier_scores["final_logits"]
                out["stage_b_v7_final_score"] = verifier_scores["final_score"]
        if patch_mask_in is not None:
            out["patch_mask"] = patch_mask_in
        if (
            score_patch is not None
            and not patch_only
            and self.stage_b_u0_patch_rank_adapter is None
            and self.stage_b_data_driven_score_heads is None
            and not self.stage_b_native_patch_category
            and ((patches is not None) or (patch_global_in is not None))
        ):
            out["patch_gate"] = alpha

        # Expose masks for Stage A/Stage B alignment (phrase count from text; patch count from inputs).
        if isinstance(text_dict, dict):
            if "phrase_mask" in text_dict:
                out["phrase_mask"] = text_dict["phrase_mask"]
            if "phrase_to_token_mask" in text_dict:
                out["phrase_to_token_mask"] = text_dict["phrase_to_token_mask"]
        for mask_key in (
            "phrase_to_token_mask",
            "canonical_to_token_mask",
            "content_to_token_mask",
            "attr_pos_to_token_mask",
            "attr_neg_to_token_mask",
            "attr_neg_weight_mask",
            "negative_to_token_mask",
            "phrase_semantic_token_mask",
            "tn_group_ids",
            "is_tn",
            "verifier_pair_stride",
            "verifier_num_patch_slots",
        ):
            mask_value = kw.get(mask_key, None)
            if mask_value is not None:
                if torch.is_tensor(mask_value):
                    mask_value = mask_value.to(samples.device)
                    if mask_value.dim() == 2 and bs == 1:
                        mask_value = mask_value.unsqueeze(0)
                out[mask_key] = mask_value
        if self.stage_b_legacy_global_gate is not None:
            if not patch_only:
                raise RuntimeError("The legacy Stage-B global gate requires patch_only=True")
            if out.get("pred_logits_text", None) is None:
                raise RuntimeError(
                    "The legacy Stage-B global gate requires patch_only_compute_text_logits=True"
                )
            # This is the exact deployed legacy score. It is deliberately
            # detached before the gate so confidence training cannot alter
            # boxes, text logits, patch logits, or their query ordering.
            with torch.no_grad():
                legacy_slot_score = compute_stage_b_slot_logits(
                    out, **self.stage_b_legacy_global_gate_score_kwargs
                )
            gate_out = self.stage_b_legacy_global_gate(hs[-1], legacy_slot_score)
            out["stage_b_legacy_slot_score"] = legacy_slot_score
            out["stage_b_legacy_global_gate_bias"] = gate_out["gate_bias"]
            out["stage_b_legacy_global_confidence"] = gate_out["confidence"]
        if patch_mask_in is not None and isinstance(text_dict, dict) and ("phrase_mask" in text_dict):
            pm = patch_mask_in.to(torch.bool)
            tm = text_dict["phrase_mask"].to(torch.bool)
            if pm.shape == tm.shape:
                out["patch_phrase_mask"] = pm & tm

        # Used to calculate losses
        bs, len_td = text_dict['text_token_mask'].shape
        out['text_mask']=torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(
            samples.device
        )
        for b in range(bs):
            for j in range(len_td):
                if text_dict['text_token_mask'][b][j] == True:
                    out['text_mask'][b][j] = True

        # for intermediate outputs
        if self.aux_loss and patch_only and aux_score_patch_list is not None:
            out['aux_outputs'] = self._set_patch_aux_loss(outputs_class, outputs_coord_list, aux_score_patch_list)
            for aux_out in out['aux_outputs']:
                for meta_key in (
                    "text_mask",
                    "phrase_mask",
                    "patch_mask",
                    "patch_phrase_mask",
                    "phrase_to_token_mask",
                    "canonical_to_token_mask",
                    "content_to_token_mask",
                    "attr_pos_to_token_mask",
                    "attr_neg_to_token_mask",
                    "negative_to_token_mask",
                    "phrase_semantic_token_mask",
                    "tn_group_ids",
                    "is_tn",
                ):
                    if meta_key in out:
                        aux_out[meta_key] = out[meta_key]
        elif self.aux_loss and (not patch_only):
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
        out['token']=one_hot_token
        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        # outputs['pred_logits'].shape
        # torch.Size([4, 900, 256])

        # outputs['pred_boxes'].shape
        # torch.Size([4, 900, 4])

        # outputs['text_mask'].shape
        # torch.Size([256])

        # outputs['text_mask']

        # outputs['aux_outputs'][0].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['aux_outputs'][img_idx]

        # outputs['token']
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>

        # outputs['interm_outputs'].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])


        # outputs['interm_outputs_for_matching_pre'].keys()
        # dict_keys(['pred_logits', 'pred_boxes'])

        # outputs['one_hot'].shape
        # torch.Size([4, 900, 256])

        return out

    def encode_patches(
        self,
        patches: Union[NestedTensor, torch.Tensor, "list[torch.Tensor]"],
        text_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        对 exemplar patches 做编码。

        常见用法：
            1) 纯 OSOD:
                patch_feats = model.encode_patches(patches)["patch_global"]
            2) Text+Patch:
                先在 forward 得到 text_dict，再:
                patch_enc = model.encode_patches(patches, text_dict)
                alpha = patch_enc["patch_gate"]
                patch_global = patch_enc["patch_global"]

        注意: 这里只是一个 thin wrapper，内部直接调用 self.patch_encoder。
        """
        if self.patch_encoder is None:
            raise RuntimeError("This model was built without a patch branch.")
        return self.patch_encoder(patches, text_dict=text_dict)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    @torch.jit.unused
    def _set_patch_aux_loss(self, outputs_class, outputs_coord, outputs_patch):
        if outputs_patch is None:
            outputs_patch = [None for _ in range(len(outputs_coord) - 1)]
        outputs_class_iter = (
            outputs_class[:-1]
            if outputs_class is not None
            else [None for _ in range(len(outputs_coord) - 1)]
        )
        aux_outputs = []
        for a, b, p in zip(outputs_class_iter, outputs_coord[:-1], outputs_patch):
            if p is not None and p.dim() == 3:
                pred_logits = p.max(dim=-1).values.unsqueeze(-1)
            elif p is not None:
                pred_logits = p.unsqueeze(-1)
            else:
                pred_logits = a
            aux_out = {
                "pred_logits": pred_logits,
                "pred_logits_patch": p,
                "pred_boxes": b,
            }
            if a is not None:
                aux_out["pred_logits_text"] = a
            aux_outputs.append(aux_out)
        return aux_outputs




class SetCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_dict,
        focal_alpha,
        focal_gamma,
        losses,
        *,
        gdino_tn_loss_type: str = "dense_focal",
        gdino_tn_alltn_weight: float = 0.0,
        gdino_tn_alltn_topk: int = 10,
        gdino_tn_alltn_tau_neg: float = -2.4,
        gdino_tn_alltn_lse_tau: float = 0.2,
        gdino_tn_alltn_text_agg: str = "mean",
        gdino_tn_token_neg_weight: float = 0.0,
        gdino_tn_token_content_weight: float = 0.0,
        gdino_tn_token_canonical_weight: float = 0.0,
        gdino_tn_token_neg_weight_mode: str = "fixed",
    ):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma= focal_gamma
        self.gdino_tn_loss_type = str(gdino_tn_loss_type).lower().strip()
        if self.gdino_tn_loss_type not in {"dense_focal", "alltn00625"}:
            raise ValueError(
                "gdino_tn_loss_type must be 'dense_focal' or 'alltn00625', "
                f"got {gdino_tn_loss_type!r}"
            )
        self.gdino_tn_alltn_weight = float(gdino_tn_alltn_weight)
        self.gdino_tn_alltn_topk = max(1, int(gdino_tn_alltn_topk))
        self.gdino_tn_alltn_tau_neg = float(gdino_tn_alltn_tau_neg)
        self.gdino_tn_alltn_lse_tau = max(float(gdino_tn_alltn_lse_tau), 1e-6)
        self.gdino_tn_alltn_text_agg = str(gdino_tn_alltn_text_agg).lower().strip()
        if self.gdino_tn_alltn_text_agg not in {"mean", "max"}:
            raise ValueError(
                "gdino_tn_alltn_text_agg must be 'mean' or 'max', "
                f"got {gdino_tn_alltn_text_agg!r}"
            )
        self.gdino_tn_token_neg_weight = float(gdino_tn_token_neg_weight)
        self.gdino_tn_token_content_weight = float(gdino_tn_token_content_weight)
        self.gdino_tn_token_canonical_weight = float(gdino_tn_token_canonical_weight)
        self.gdino_tn_token_neg_weight_mode = str(gdino_tn_token_neg_weight_mode).lower().strip()
        if self.gdino_tn_token_neg_weight_mode not in {"fixed", "token_count"}:
            raise ValueError(
                "gdino_tn_token_neg_weight_mode must be 'fixed' or 'token_count', "
                f"got {gdino_tn_token_neg_weight_mode!r}"
            )

    def _target_is_tn(self, target: dict, *, device: torch.device) -> bool:
        flag = target.get("is_negative", None)
        if torch.is_tensor(flag):
            return bool(flag.to(device=device).view(-1).any().item())
        return False

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses


    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        pred_logits=outputs['pred_logits']
        new_targets=outputs['one_hot'].to(pred_logits.device)
        text_mask=outputs['text_mask']

        assert (new_targets.dim() == 3)
        assert (pred_logits.dim() == 3)  # batch x from x to
        
        bs, n, _ = pred_logits.shape
        alpha=self.focal_alpha
        gamma=self.focal_gamma
        if text_mask is not None:
            # ODVG: each sample has different mask 
            expanded_text_mask = text_mask.repeat(1, pred_logits.size(1)).view(
                outputs['text_mask'].shape[0], -1, outputs['text_mask'].shape[1]
            )
            pred_logits = torch.masked_select(pred_logits, expanded_text_mask)
            new_targets = torch.masked_select(new_targets, expanded_text_mask)

        new_targets=new_targets.float()
        if pred_logits.numel() == 0:
            return {'loss_ce': outputs['pred_logits'].sum() * 0.0}
        p = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss

        total_num_pos=0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos , 1.0)
        loss=loss.sum()/num_pos_avg_per_gpu
        
        losses = {'loss_ce': loss}
        return losses

    def gdino_alltn00625_tn_loss(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        text_mask = outputs.get("text_mask", None)
        zero = pred_logits.new_zeros(())
        if self.gdino_tn_loss_type != "alltn00625" or self.gdino_tn_alltn_weight <= 0:
            return {
                "loss_tn_alltn": zero,
                "tn_alltn_loss_raw": zero.detach(),
                "tn_alltn_sample_count": zero.detach(),
                "tn_alltn_score": zero.detach(),
                "tn_alltn_topk_max_score": zero.detach(),
            }
        if text_mask is None:
            text_mask = torch.ones(
                pred_logits.shape[0],
                pred_logits.shape[-1],
                dtype=torch.bool,
                device=pred_logits.device,
            )
        else:
            text_mask = text_mask.to(device=pred_logits.device, dtype=torch.bool)
        losses = []
        agg_scores = []
        topk_max_scores = []
        for b, target in enumerate(targets):
            if not self._target_is_tn(target, device=pred_logits.device):
                continue
            mask = text_mask[b]
            if not bool(mask.any().item()):
                continue
            probs_b = pred_logits[b].sigmoid()
            if self.gdino_tn_alltn_text_agg == "max":
                query_scores = probs_b.masked_fill(~mask[None, :], 0.0).max(dim=-1).values
            else:
                denom = mask.to(dtype=probs_b.dtype).sum().clamp(min=1.0)
                query_scores = probs_b.masked_fill(~mask[None, :], 0.0).sum(dim=-1) / denom
            k = min(self.gdino_tn_alltn_topk, int(query_scores.numel()))
            if k <= 0:
                continue
            topk_scores = torch.topk(query_scores, k=k, largest=True).values
            tau = max(float(self.gdino_tn_alltn_lse_tau), 1e-6)
            agg = tau * torch.logsumexp(topk_scores / tau, dim=0)
            losses.append(F.softplus(agg - self.gdino_tn_alltn_tau_neg))
            agg_scores.append(agg.detach().reshape(1))
            topk_max_scores.append(topk_scores.max().detach().reshape(1))

        if not losses:
            raw = zero
            count = 0
        else:
            raw = torch.stack([x.reshape(()) for x in losses]).mean()
            count = len(losses)

        def _cat_mean(values):
            if not values:
                return zero.detach()
            return torch.cat(values).mean().detach()

        return {
            "loss_tn_alltn": raw,
            "tn_alltn_loss_raw": raw.detach(),
            "tn_alltn_sample_count": torch.as_tensor(float(count), device=pred_logits.device),
            "tn_alltn_score": _cat_mean(agg_scores),
            "tn_alltn_topk_max_score": _cat_mean(topk_max_scores),
            "tn_alltn_topk": torch.as_tensor(float(self.gdino_tn_alltn_topk), device=pred_logits.device),
            "tn_alltn_tau_neg": torch.as_tensor(float(self.gdino_tn_alltn_tau_neg), device=pred_logits.device),
            "tn_alltn_lse_tau": torch.as_tensor(float(self.gdino_tn_alltn_lse_tau), device=pred_logits.device),
        }

    def gdino_tn_token_loss(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        text_mask = outputs.get("text_mask", None)
        negative_mask = outputs.get("negative_to_token_mask", outputs.get("attr_neg_to_token_mask", None))
        phrase_mask = outputs.get("phrase_to_token_mask", None)
        content_mask = outputs.get("content_to_token_mask", None)
        canonical_mask = outputs.get("canonical_to_token_mask", None)
        zero = pred_logits.new_zeros(())
        if (
            self.gdino_tn_token_neg_weight <= 0
            and self.gdino_tn_token_content_weight <= 0
            and self.gdino_tn_token_canonical_weight <= 0
        ):
            return {
                "loss_tn_tokens": zero,
                "tn_token_neg_loss_raw": zero.detach(),
                "tn_token_neg_loss_weighted_raw": zero.detach(),
                "tn_token_sample_count": zero.detach(),
                "tn_token_valid_count": zero.detach(),
                "tn_token_skipped_no_mask_count": zero.detach(),
                "tn_token_neg_weight": zero.detach(),
                "tn_token_neg_effective_weight": zero.detach(),
                "tn_token_neg_weight_mode_token_count": zero.detach(),
                "tn_token_content_weight": zero.detach(),
                "tn_token_canonical_weight": zero.detach(),
            }
        if text_mask is None:
            text_mask = torch.ones(
                pred_logits.shape[0],
                pred_logits.shape[-1],
                dtype=torch.bool,
                device=pred_logits.device,
            )
        else:
            text_mask = text_mask.to(device=pred_logits.device, dtype=torch.bool)

        losses = []
        neg_raw_losses = []
        content_losses = []
        canonical_losses = []
        neg_effective_weights = []
        valid_token_count = 0
        sample_count = 0
        skipped_no_mask_count = 0
        for b, target in enumerate(targets):
            if not self._target_is_tn(target, device=pred_logits.device):
                continue
            supervised_this_sample = False
            mask = text_mask[b]
            if not bool(mask.any().item()):
                continue
            phrase_source = phrase_mask[b] if phrase_mask is not None else target.get("phrase_to_token_mask", None)
            phrase_token_count = 0
            if phrase_source is not None:
                phrase_b = phrase_source.to(device=pred_logits.device, dtype=torch.bool)
                if phrase_b.dim() == 1:
                    phrase_b = phrase_b.unsqueeze(0)
                phrase_b = phrase_b[:1, : pred_logits.shape[-1]].any(dim=0) & mask
                phrase_token_count = int(phrase_b.sum().item())
            neg_source = None
            if negative_mask is not None:
                neg_source = negative_mask[b]
            else:
                neg_source = target.get("negative_to_token_mask", target.get("attr_neg_to_token_mask", None))
            if neg_source is None:
                skipped_no_mask_count += 1
            else:
                neg_b = neg_source.to(device=pred_logits.device, dtype=torch.bool)
                if neg_b.dim() == 1:
                    neg_b = neg_b.unsqueeze(0)
                neg_b = neg_b[:1, : pred_logits.shape[-1]].any(dim=0) & mask
                if neg_b.any():
                    logits = pred_logits[b][:, neg_b]
                    raw = F.binary_cross_entropy_with_logits(
                        logits,
                        torch.zeros_like(logits),
                        reduction="mean",
                    )
                    neg_token_count = int(neg_b.sum().item())
                    tn_token_count = phrase_token_count if phrase_token_count > 0 else neg_token_count
                    neg_weight = (
                        float(tn_token_count)
                        if self.gdino_tn_token_neg_weight_mode == "token_count"
                        else 1.0
                    )
                    losses.append(raw * neg_weight)
                    neg_raw_losses.append(raw)
                    neg_effective_weights.append(neg_weight)
                    valid_token_count += neg_token_count * int(pred_logits.shape[1])
                    supervised_this_sample = True
            content_source = content_mask[b] if content_mask is not None else target.get("content_to_token_mask", None)
            if self.gdino_tn_token_content_weight > 0 and content_source is not None:
                content_b = content_source.to(device=pred_logits.device, dtype=torch.bool)
                if content_b.dim() == 1:
                    content_b = content_b.unsqueeze(0)
                content_b = content_b[:1, : pred_logits.shape[-1]].any(dim=0) & mask
                if content_b.any():
                    content_logits = pred_logits[b][:, content_b]
                    content_losses.append(
                        F.binary_cross_entropy_with_logits(
                            content_logits,
                            torch.zeros_like(content_logits),
                            reduction="mean",
                        )
                    )
                    supervised_this_sample = True
            canonical_source = canonical_mask[b] if canonical_mask is not None else target.get("canonical_to_token_mask", None)
            if self.gdino_tn_token_canonical_weight > 0 and canonical_source is not None:
                canonical_b = canonical_source.to(device=pred_logits.device, dtype=torch.bool)
                if canonical_b.dim() == 1:
                    canonical_b = canonical_b.unsqueeze(0)
                canonical_b = canonical_b[:1, : pred_logits.shape[-1]].any(dim=0) & mask
                if canonical_b.any():
                    canonical_logits = pred_logits[b][:, canonical_b]
                    canonical_losses.append(
                        F.binary_cross_entropy_with_logits(
                            canonical_logits,
                            torch.zeros_like(canonical_logits),
                            reduction="mean",
                        )
                    )
                    supervised_this_sample = True
            if supervised_this_sample:
                sample_count += 1

        if losses:
            neg_weighted_raw = torch.stack([x.reshape(()) for x in losses]).mean()
        else:
            neg_weighted_raw = zero
        neg_raw = torch.stack([x.reshape(()) for x in neg_raw_losses]).mean() if neg_raw_losses else zero
        content_raw = torch.stack([x.reshape(()) for x in content_losses]).mean() if content_losses else zero
        canonical_raw = torch.stack([x.reshape(()) for x in canonical_losses]).mean() if canonical_losses else zero
        if neg_effective_weights:
            neg_effective_weight = torch.as_tensor(
                float(sum(neg_effective_weights) / len(neg_effective_weights)),
                device=pred_logits.device,
            )
        else:
            neg_effective_weight = zero.detach()

        loss = (
            neg_weighted_raw * self.gdino_tn_token_neg_weight
            + content_raw * self.gdino_tn_token_content_weight
            + canonical_raw * self.gdino_tn_token_canonical_weight
        )
        return {
            "loss_tn_tokens": loss,
            "tn_token_neg_loss_raw": neg_raw.detach(),
            "tn_token_neg_loss_weighted_raw": neg_weighted_raw.detach(),
            "tn_token_content_loss_raw": content_raw.detach(),
            "tn_token_canonical_loss_raw": canonical_raw.detach(),
            "tn_token_sample_count": torch.as_tensor(float(sample_count), device=pred_logits.device),
            "tn_token_valid_count": torch.as_tensor(float(valid_token_count), device=pred_logits.device),
            "tn_token_skipped_no_mask_count": torch.as_tensor(float(skipped_no_mask_count), device=pred_logits.device),
            "tn_token_neg_weight": torch.as_tensor(float(self.gdino_tn_token_neg_weight), device=pred_logits.device),
            "tn_token_neg_effective_weight": neg_effective_weight,
            "tn_token_neg_weight_mode_token_count": torch.as_tensor(
                float(self.gdino_tn_token_neg_weight_mode == "token_count"),
                device=pred_logits.device,
            ),
            "tn_token_content_weight": torch.as_tensor(float(self.gdino_tn_token_content_weight), device=pred_logits.device),
            "tn_token_canonical_weight": torch.as_tensor(float(self.gdino_tn_token_canonical_weight), device=pred_logits.device),
        }


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.token_sigmoid_binary_focal_loss,
            'tn_alltn': self.gdino_alltn00625_tn_loss,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, cat_list, caption, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        device=next(iter(outputs.values())).device
        one_hot = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64) # torch.Size([bs, 900, 256])
        token = outputs['token'] 
        
        label_map_list = []
        indices = []
        for j in range(len(cat_list)): # bs
            label_map=[]
            for i in range(len(cat_list[j])):
                label_id=torch.tensor([i])
                per_label=create_positive_map(token[j], label_id, cat_list[j], caption[j])
                label_map.append(per_label)
            label_map=torch.stack(label_map,dim=0).squeeze(1)
            label_map_list.append(label_map)
        for j in range(len(cat_list)): # bs
            for_match = {
                "pred_logits" : outputs['pred_logits'][j].unsqueeze(0),
                "pred_boxes" : outputs['pred_boxes'][j].unsqueeze(0)
            }
            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)
        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)

        # import pdb; pdb.set_trace()
        tgt_ids = [v["labels"].cpu() for v in targets]
        # len(tgt_ids) == bs
        for i in range(len(indices)):
            tgt_ids[i]=tgt_ids[i][indices[i][1]]
            one_hot[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
        outputs['one_hot'] = one_hot
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        num_boxes = sum(num_boxes_list)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if self.gdino_tn_loss_type == "alltn00625" and self.gdino_tn_alltn_weight > 0:
            losses.update(self.gdino_alltn00625_tn_loss(outputs, targets, indices, num_boxes))
        if (
            self.gdino_tn_token_neg_weight > 0
            or self.gdino_tn_token_content_weight > 0
            or self.gdino_tn_token_canonical_weight > 0
        ):
            losses.update(self.gdino_tn_token_loss(outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = []
                for j in range(len(cat_list)): # bs
                    aux_output_single = {
                        'pred_logits' : aux_outputs['pred_logits'][j].unsqueeze(0),
                        'pred_boxes': aux_outputs['pred_boxes'][j].unsqueeze(0)
                    }
                    inds = self.matcher(aux_output_single, [targets[j]], label_map_list[j])
                    indices.extend(inds)
                one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
                tgt_ids = [v["labels"].cpu() for v in targets]
                for i in range(len(indices)):
                    tgt_ids[i]=tgt_ids[i][indices[i][1]]
                    one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
                aux_outputs['one_hot'] = one_hot_aux
                aux_outputs['text_mask'] = outputs['text_mask']
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)                
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if self.gdino_tn_loss_type == "alltn00625" and self.gdino_tn_alltn_weight > 0:
                    l_dict = self.gdino_alltn00625_tn_loss(aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if (
                    self.gdino_tn_token_neg_weight > 0
                    or self.gdino_tn_token_content_weight > 0
                    or self.gdino_tn_token_canonical_weight > 0
                ):
                    l_dict = self.gdino_tn_token_loss(aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = []
            for j in range(len(cat_list)): # bs
                interm_output_single = {
                    'pred_logits' : interm_outputs['pred_logits'][j].unsqueeze(0),
                    'pred_boxes': interm_outputs['pred_boxes'][j].unsqueeze(0)
                }
                inds = self.matcher(interm_output_single, [targets[j]], label_map_list[j])
                indices.extend(inds)
            one_hot_aux = torch.zeros(outputs['pred_logits'].size(),dtype=torch.int64)
            tgt_ids = [v["labels"].cpu() for v in targets]
            for i in range(len(indices)):
                tgt_ids[i]=tgt_ids[i][indices[i][1]]
                one_hot_aux[i,indices[i][0]] = label_map_list[i][tgt_ids[i]].to(torch.long)
            interm_outputs['one_hot'] = one_hot_aux
            interm_outputs['text_mask'] = outputs['text_mask']
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)
            if self.gdino_tn_loss_type == "alltn00625" and self.gdino_tn_alltn_weight > 0:
                l_dict = self.gdino_alltn00625_tn_loss(interm_outputs, targets, indices, num_boxes)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)
            if (
                self.gdino_tn_token_neg_weight > 0
                or self.gdino_tn_token_content_weight > 0
                or self.gdino_tn_token_canonical_weight > 0
            ):
                l_dict = self.gdino_tn_token_loss(interm_outputs, targets, indices, num_boxes)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100,text_encoder_type='text_encoder_type', nms_iou_threshold=-1,use_coco_eval=False,args=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        if args.use_coco_eval:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_val_path)
            category_dict = coco.loadCats(coco.getCatIds())
            cat_list = [item['name'] for item in category_dict]
        else:
            cat_list=args.label_list
        caption = " . ".join(cat_list) + ' .'
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        label_list = torch.arange(len(cat_list))
        pos_map=create_positive_map(tokenized,label_list,cat_list,caption)
        # build a mapping from label_id to pos_map
        if args.use_coco_eval:
            id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                    41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
            new_pos_map = torch.zeros((91, 256))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map


        self.nms_iou_threshold=nms_iou_threshold
        self.positive_map = pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']


        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind]=pos_maps[label_ind]/pos_maps[label_ind].sum()

        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')
        labels = topk_indexes % prob.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


class PostProcessStageB(nn.Module):
    """Stage B slot-level patch/text fusion for patch-episode inference."""

    def __init__(
        self,
        num_select=100,
        nms_iou_threshold=-1,
        *,
        beta: float = 1.0,
        canonical_weight: float = 1.0,
        text_agg: str = "mean",
        softmin_tau: float = 0.7,
        mean_softmin_alpha: float = 0.5,
        output_sigmoid_scores: bool = False,
        normalize_fused_score: bool = True,
        score_mode: str = "patch_text",
    ) -> None:
        super().__init__()
        self.num_select = int(num_select)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.beta = float(beta)
        self.canonical_weight = float(canonical_weight)
        self.text_agg = str(text_agg).lower().strip()
        self.softmin_tau = float(softmin_tau)
        self.mean_softmin_alpha = float(mean_softmin_alpha)
        self.output_sigmoid_scores = bool(output_sigmoid_scores)
        self.normalize_fused_score = bool(normalize_fused_score)
        self.score_mode = str(score_mode).lower().replace("-", "_").strip()

    def _aggregate_tokens(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        from .stage_b_score import aggregate_stage_b_tokens

        return aggregate_stage_b_tokens(
            logits,
            mask,
            text_agg=self.text_agg,
            softmin_tau=self.softmin_tau,
            mean_softmin_alpha=self.mean_softmin_alpha,
        )

    def compute_slot_logits(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_stage_b_slot_logits(
            outputs,
            beta=self.beta,
            canonical_weight=self.canonical_weight,
            text_agg=self.text_agg,
            softmin_tau=self.softmin_tau,
            mean_softmin_alpha=self.mean_softmin_alpha,
            detach_patch=False,
            normalize_fused_score=self.normalize_fused_score,
            score_mode=self.score_mode,
        )

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        slot_logits = self.compute_slot_logits(outputs)
        out_bbox = outputs["pred_boxes"]

        assert len(out_bbox) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        B, Q, K = slot_logits.shape
        num_select = min(int(self.num_select), Q * K)
        topk_logits, topk_indexes = torch.topk(slot_logits.reshape(B, -1), num_select, dim=1)
        query_ids = torch.div(topk_indexes, K, rounding_mode="trunc")
        slot_ids = topk_indexes % K
        scores = topk_logits.sigmoid() if self.output_sigmoid_scores else topk_logits

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, query_ids.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        dense_scores = slot_logits.sigmoid() if self.output_sigmoid_scores else slot_logits
        for b in range(B):
            s = scores[b]
            q = query_ids[b]
            l = slot_ids[b]
            box = boxes[b]
            if self.nms_iou_threshold > 0:
                keep = nms(box, s, iou_threshold=self.nms_iou_threshold)
                s = s[keep]
                q = q[keep]
                l = l[keep]
                box = box[keep]
            results.append(
                {
                    "scores": s,
                    "labels": l,
                    "slot_ids": l,
                    "query_ids": q,
                    "boxes": box,
                    "slot_scores": dense_scores[b],
                    "slot_logits": slot_logits[b],
                }
            )
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    patch_only = bool(getattr(args, "patch_only", False))
    stage_b = bool(getattr(args, "stage_b", False))
    stage_b_v7 = bool(getattr(args, "stage_b_v7", False))
    stage_b_v11 = bool(getattr(args, "stage_b_v11_fixed_text", False))
    stage_b_dense_duty = bool(getattr(args, "stage_b_dense_duty", False))
    if stage_b_dense_duty and not stage_b_v11:
        raise ValueError(
            "stage_b_dense_duty requires stage_b_v11_fixed_text=True so the "
            "existing fixed-candidate training/evaluation contract remains active"
        )
    stage_b_legacy_global_gate = bool(
        getattr(args, "stage_b_legacy_global_gate", False)
    )
    stage_b_gdino_score_adapter = bool(
        getattr(args, "stage_b_gdino_score_adapter", False)
    )
    stage_b_u0_patch_rank = bool(
        getattr(args, "stage_b_u0_patch_rank", False)
    )
    stage_b_u0_gate_aligned_d10 = bool(
        getattr(args, "stage_b_u0_gate_aligned_d10", False)
    )
    stage_b_u0_gate_aligned_d11 = bool(
        getattr(args, "stage_b_u0_gate_aligned_d11", False)
    )
    stage_b_u0_gate_aligned_rank_residual = bool(
        getattr(args, "stage_b_u0_gate_aligned_rank_residual", False)
    )
    stage_b_u0_gate_aligned_d12 = bool(
        getattr(args, "stage_b_u0_gate_aligned_d12", False)
    )
    stage_b_u0_gate_aligned_patch_residual = bool(
        getattr(args, "stage_b_u0_gate_aligned_patch_residual", False)
    )
    stage_b_u0_gate_aligned_d13 = bool(
        getattr(args, "stage_b_u0_gate_aligned_d13", False)
    )
    stage_b_data_driven_score = bool(
        getattr(args, "stage_b_data_driven_score", False)
    )
    stage_b_native_patch_category = bool(
        getattr(args, "stage_b_native_patch_category", False)
    )
    if stage_b_u0_patch_rank and not stage_b_gdino_score_adapter:
        raise ValueError(
            "stage_b_u0_patch_rank requires stage_b_gdino_score_adapter=True"
        )
    if stage_b_u0_gate_aligned_d10 and not stage_b_u0_patch_rank:
        raise ValueError(
            "stage_b_u0_gate_aligned_d10 requires stage_b_u0_patch_rank=True"
        )
    if stage_b_u0_gate_aligned_d11 and not stage_b_u0_patch_rank:
        raise ValueError(
            "stage_b_u0_gate_aligned_d11 requires stage_b_u0_patch_rank=True"
        )
    if stage_b_u0_gate_aligned_rank_residual and not stage_b_u0_patch_rank:
        raise ValueError(
            "gate-aligned rank residual requires stage_b_u0_patch_rank=True"
        )
    if stage_b_u0_gate_aligned_d12 and not stage_b_u0_gate_aligned_rank_residual:
        raise ValueError("D12 training requires its rank-residual architecture")
    if stage_b_u0_gate_aligned_patch_residual and not stage_b_u0_patch_rank:
        raise ValueError(
            "gate-aligned patch residual requires stage_b_u0_patch_rank=True"
        )
    if stage_b_u0_gate_aligned_d13 and not stage_b_u0_gate_aligned_patch_residual:
        raise ValueError("D13 training requires its patch-residual architecture")
    if (
        stage_b_u0_gate_aligned_rank_residual
        and stage_b_u0_gate_aligned_patch_residual
    ):
        raise ValueError("D12 and D13 residual architectures are mutually exclusive")
    if sum(
        int(value)
        for value in (
            stage_b_u0_gate_aligned_d10,
            stage_b_u0_gate_aligned_d11,
            stage_b_u0_gate_aligned_d12,
            stage_b_u0_gate_aligned_d13,
        )
    ) > 1:
        raise ValueError(
            "D10, D11, D12, and D13 training objectives are mutually exclusive"
        )
    if stage_b_u0_gate_aligned_d10 and bool(
        getattr(args, "stage_b_u0_category_preserving_patch_gate", False)
    ):
        raise ValueError(
            "D10 training requires the inference-only hard category gate disabled"
        )
    if stage_b_u0_gate_aligned_d11 and not bool(
        getattr(args, "stage_b_u0_category_preserving_patch_gate", False)
    ):
        raise ValueError(
            "D11 training requires the exact hard category gate enabled"
        )
    if stage_b_u0_gate_aligned_rank_residual and not bool(
        getattr(args, "stage_b_u0_category_preserving_patch_gate", False)
    ):
        raise ValueError(
            "gate-aligned rank residual requires the exact hard category gate"
        )
    if stage_b_u0_gate_aligned_patch_residual and not bool(
        getattr(args, "stage_b_u0_category_preserving_patch_gate", False)
    ):
        raise ValueError(
            "gate-aligned patch residual requires the exact hard category gate"
        )
    if stage_b_gdino_score_adapter and (
        patch_only
        or stage_b
        or stage_b_v7
        or stage_b_v11
        or stage_b_legacy_global_gate
        or (
            bool(getattr(args, "enable_patch_branch", False))
            and not stage_b_u0_patch_rank
        )
    ):
        raise ValueError(
            "stage_b_gdino_score_adapter requires ordinary non-patch GroundingDINO "
            "with every patch/legacy/fixed-scorer mode disabled"
        )
    if stage_b_data_driven_score and (
        patch_only
        or stage_b
        or stage_b_v7
        or stage_b_v11
        or stage_b_legacy_global_gate
        or stage_b_gdino_score_adapter
        or stage_b_u0_patch_rank
    ):
        raise ValueError(
            "stage_b_data_driven_score is an independent absolute-score path and "
            "cannot be combined with legacy, fixed-text, or teacher adapters"
        )
    if stage_b_native_patch_category and (
        patch_only
        or stage_b
        or stage_b_v7
        or stage_b_v11
        or stage_b_legacy_global_gate
        or stage_b_gdino_score_adapter
        or stage_b_u0_patch_rank
        or stage_b_data_driven_score
    ):
        raise ValueError(
            "stage_b_native_patch_category is an independent b58-native full-text "
            "path and cannot be combined with legacy or adapter score paths"
        )
    if stage_b_legacy_global_gate and (stage_b_v7 or stage_b_v11):
        raise ValueError(
            "stage_b_legacy_global_gate is only compatible with the legacy Stage-B scorer"
        )
    patch_gate_with_text = bool(
        getattr(
            args,
            "patch_gate_with_text",
            not patch_only and not stage_b_native_patch_category,
        )
    )
    if patch_only:
        patch_gate_with_text = False
    if stage_b_native_patch_category and patch_gate_with_text:
        raise ValueError(
            "stage_b_native_patch_category requires patch_gate_with_text=False"
        )
    enable_patch_branch = bool(
        getattr(
            args,
            "enable_patch_branch",
            patch_only
            or stage_b
            or stage_b_v7
            or stage_b_v11
            or stage_b_legacy_global_gate
            or stage_b_data_driven_score
            or stage_b_native_patch_category,
        )
    )
    if stage_b_native_patch_category and not enable_patch_branch:
        raise ValueError(
            "stage_b_native_patch_category requires enable_patch_branch=True"
        )

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        patch_gate_with_text=patch_gate_with_text,
        patch_only=patch_only,
        enable_patch_branch=enable_patch_branch,
        patch_only_compute_text_logits=bool(getattr(args, "patch_only_compute_text_logits", False) or stage_b),
        patch_logit_scale_init=float(getattr(args, "patch_logit_scale_init", 14.2857142857)),
        patch_logit_scale_max=float(getattr(args, "patch_logit_scale_max", 100.0)),
        patch_dn_num_queries=int(getattr(args, "patch_dn_num_queries", 0)),
        patch_dn_box_noise_scale=float(getattr(args, "patch_dn_box_noise_scale", getattr(args, "dn_box_noise_scale", 0.4))),
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )
    model.stage_b_native_patch_category = stage_b_native_patch_category

    if stage_b_gdino_score_adapter:
        from .stage_b_gdino_score_adapter import StageBGDINOScoreAdapter

        model.stage_b_gdino_score_adapter = StageBGDINOScoreAdapter(
            hidden_dim=model.hidden_dim,
            adapter_dim=int(
                getattr(args, "stage_b_gdino_adapter_dim", 128)
            ),
            gate_hidden_dim=int(
                getattr(args, "stage_b_gdino_gate_hidden_dim", 128)
            ),
            gate_pool_temperature=float(
                getattr(args, "stage_b_gdino_gate_pool_temperature", 0.1)
            ),
            gate_topk=int(getattr(args, "stage_b_gdino_gate_topk", 10)),
        )
        if stage_b_u0_gate_aligned_rank_residual:
            from .stage_b_u0_gate_aligned_d12 import (
                StageBU0GateAlignedD12RankResidual,
            )

            model.stage_b_u0_gate_aligned_rank_residual = (
                StageBU0GateAlignedD12RankResidual(
                    feature_dim=model.stage_b_gdino_score_adapter.adapter_dim,
                    hidden_dim=int(
                        getattr(args, "stage_b_u0_d12_hidden_dim", 64)
                    ),
                    residual_limit=float(
                        getattr(args, "stage_b_u0_d12_residual_limit", 0.1)
                    ),
                )
            )
        if stage_b_u0_gate_aligned_patch_residual:
            from .stage_b_u0_gate_aligned_d13 import (
                StageBU0GateAlignedD13PatchResidual,
            )

            model.stage_b_u0_gate_aligned_patch_residual = (
                StageBU0GateAlignedD13PatchResidual(
                    feature_dim=model.hidden_dim,
                    hidden_dim=int(
                        getattr(args, "stage_b_u0_d13_hidden_dim", 64)
                    ),
                    residual_limit=float(
                        getattr(args, "stage_b_u0_d13_residual_limit", 0.25)
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_u0_category_gate_max_gap", 2.0)
                    ),
                )
            )
        if stage_b_u0_patch_rank:
            if not model.enable_patch_branch:
                raise ValueError(
                    "stage_b_u0_patch_rank requires enable_patch_branch=True"
                )
            from .stage_b_u0_patch_rank import StageBU0PatchRankAdapter

            model.stage_b_u0_patch_rank_adapter = StageBU0PatchRankAdapter(
                query_count=int(getattr(args, "num_queries", 900)),
                hidden_dim=int(
                    getattr(args, "stage_b_u0_patch_rank_hidden_dim", 64)
                ),
                score_clip=float(
                    getattr(args, "stage_b_u0_patch_rank_score_clip", 5.0)
                ),
                direct_patch_skip=bool(
                    getattr(args, "stage_b_u1_direct_patch_skip", False)
                ),
                direct_patch_gain_limit=float(
                    getattr(args, "stage_b_u1_direct_patch_gain_limit", 0.5)
                ),
                category_preserving_gate=bool(
                    getattr(
                        args,
                        "stage_b_u0_category_preserving_patch_gate",
                        False,
                    )
                ),
                category_gate_max_gap=float(
                    getattr(args, "stage_b_u0_category_gate_max_gap", 1.0)
                ),
            )

    if stage_b_data_driven_score:
        if not model.enable_patch_branch:
            raise ValueError(
                "stage_b_data_driven_score requires enable_patch_branch=True"
            )
        from .stage_b_data_driven_score import StageBDataDrivenScoreHeads

        model.stage_b_data_driven_score_heads = StageBDataDrivenScoreHeads(
            hidden_dim=model.hidden_dim,
            rank_dim=int(getattr(args, "stage_b_data_driven_rank_dim", 128)),
            rank_architecture=str(
                getattr(
                    args,
                    "stage_b_data_driven_rank_architecture",
                    "absolute_token",
                )
            ),
            rank_num_heads=int(
                getattr(args, "stage_b_data_driven_rank_num_heads", 4)
            ),
            rank_image_level_policy=str(
                getattr(args, "stage_b_data_driven_rank_image_level_policy", "last")
            ),
            rank_image_levels=int(
                getattr(args, "stage_b_data_driven_rank_image_levels", 2)
            ),
            rank_image_pool_size=int(
                getattr(args, "stage_b_data_driven_rank_image_pool_size", 8)
            ),
            rank_image_pool_policy=str(
                getattr(
                    args,
                    "stage_b_data_driven_rank_image_pool_policy",
                    "valid_extent_masked_adaptive_avg_v1",
                )
            ),
            rank_box_fourier_bands=int(
                getattr(args, "stage_b_data_driven_rank_box_fourier_bands", 16)
            ),
            rank_ffn_dim=int(
                getattr(args, "stage_b_data_driven_rank_ffn_dim", 512)
            ),
            rank_dropout=float(
                getattr(args, "stage_b_data_driven_rank_dropout", 0.0)
            ),
            head_init_seed=int(
                getattr(args, "stage_b_data_driven_head_init_seed", 42)
            ),
            confidence_dim=int(
                getattr(args, "stage_b_data_driven_confidence_dim", 128)
            ),
            token_temperature=float(
                getattr(args, "stage_b_data_driven_token_temperature", 0.07)
            ),
            gate_hidden_dim=int(
                getattr(args, "stage_b_data_driven_gate_hidden_dim", 128)
            ),
            gate_pool_temperature=float(
                getattr(args, "stage_b_data_driven_gate_pool_temperature", 0.1)
            ),
            gate_topk=int(
                getattr(args, "stage_b_data_driven_gate_topk", 10)
            ),
            category_gate=bool(
                getattr(args, "stage_b_data_driven_category_gate", False)
            ),
            category_gate_max_gap=float(
                getattr(args, "stage_b_data_driven_category_gate_max_gap", 3.0)
            ),
            patch_score_clip=float(
                getattr(args, "stage_b_data_driven_patch_score_clip", 5.0)
            ),
        )
        if bool(
            getattr(args, "stage_b_data_driven_patch_residual", False)
        ):
            from .stage_b_data_driven_patch_residual import (
                DATA_DRIVEN_PATCH_RESIDUAL_TOPK_SEMANTIC_CONTRACT,
                StageBDataDrivenPatchResidualMatcher,
                StageBDataDrivenTopKPatchResidualMatcher,
            )

            residual_contract = getattr(
                args, "stage_b_data_driven_patch_residual_contract", None
            )
            if (
                residual_contract
                == DATA_DRIVEN_PATCH_RESIDUAL_TOPK_SEMANTIC_CONTRACT
            ):
                residual_matcher = StageBDataDrivenTopKPatchResidualMatcher(
                    feature_dim=model.hidden_dim,
                    hidden_dim=int(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_hidden_dim",
                            128,
                        )
                    ),
                    context_dim=int(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_context_dim",
                            16,
                        )
                    ),
                    topk=int(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_context_topk",
                            10,
                        )
                    ),
                    residual_limit=float(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_limit",
                            0.25,
                        )
                    ),
                    init_seed=int(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_init_seed",
                            42,
                        )
                    ),
                )
            else:
                residual_matcher = StageBDataDrivenPatchResidualMatcher(
                    feature_dim=model.hidden_dim,
                    hidden_dim=int(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_hidden_dim",
                            128,
                        )
                    ),
                    residual_limit=float(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_limit",
                            0.25,
                        )
                    ),
                    init_seed=int(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_init_seed",
                            42,
                        )
                    ),
                    center_raw=bool(
                        getattr(
                            args,
                            "stage_b_data_driven_patch_residual_center_raw",
                            False,
                        )
                    ),
                )
            model.stage_b_data_driven_patch_residual = residual_matcher

    if stage_b_legacy_global_gate:
        from .stage_b_legacy_global_gate import LegacyStageBGlobalGate

        model.stage_b_legacy_global_gate = LegacyStageBGlobalGate(
            hidden_dim=model.hidden_dim,
            gate_hidden_dim=int(
                getattr(args, "stage_b_legacy_global_gate_hidden_dim", 128)
            ),
            score_pool_temperature=float(
                getattr(args, "stage_b_legacy_global_gate_pool_temperature", 0.1)
            ),
            score_topk=int(
                getattr(args, "stage_b_legacy_global_gate_score_topk", 10)
            ),
        )
        model.stage_b_legacy_global_gate_score_kwargs = {
            "beta": float(getattr(args, "stage_b_infer_text_beta", 1.0)),
            "canonical_weight": float(
                getattr(args, "stage_b_infer_canonical_weight", 1.0)
            ),
            "text_agg": str(getattr(args, "stage_b_infer_text_agg", "mean")),
            "softmin_tau": float(
                getattr(args, "stage_b_infer_softmin_tau", getattr(args, "softmin_tau", 0.7))
            ),
            "mean_softmin_alpha": float(
                getattr(args, "stage_b_infer_mean_softmin_alpha", 0.5)
            ),
            "normalize_fused_score": bool(
                getattr(args, "stage_b_infer_normalize_fused_score", True)
            ),
            "score_mode": str(getattr(args, "stage_b_score_mode", "patch_text")),
        }

    if stage_b_v11:
        model.stage_b_v11_candidate_topk = int(
            getattr(args, "stage_b_v11_candidate_topk", 50)
        )
        model.stage_b_v15_exclude_canonical_from_score = bool(
            getattr(args, "stage_b_v15_exclude_canonical_from_score", False)
        )
        if stage_b_dense_duty:
            from .stage_b_dense_duty_scorer import StageBDenseDutyScorer

            source_decoder_layers = int(
                getattr(model.transformer.decoder, "num_layers", 0)
            )
            requested_layers = int(
                getattr(args, "stage_b_v11_num_layers", source_decoder_layers)
            )
            if requested_layers != source_decoder_layers:
                raise ValueError(
                    "dense-duty Stage B must replay the complete source decoder: "
                    f"requested {requested_layers}, source has {source_decoder_layers}"
                )
            if bool(getattr(args, "stage_b_v15_patch_rank_fusion", False)):
                raise ValueError(
                    "dense-duty Stage B forbids additive patch/text rank fusion; "
                    "patch owns category admission/category-only ranking, while "
                    "modifier-bearing expressions are ranked only by text"
                )
            model.stage_b_dense_duty = True
            confidence_phrase_aggregation = str(
                getattr(
                    args,
                    "stage_b_dense_duty_confidence_phrase_aggregation",
                    "legacy_prob_mean_add_v1",
                )
            ).strip().lower()
            model.stage_b_dense_duty_confidence_word_groups = (
                confidence_phrase_aggregation
                in {
                    "trace_activated_word_veto_product_v1",
                    "trace_activated_word_veto_penalty_v2",
                    "trace_activated_word_veto_absolute_cap_v4",
                    "trace_activated_word_veto_gated_pool_absolute_cap_v5",
                }
            )
            model.stage_b_dense_duty_allow_incidental_trace_edits = bool(
                getattr(
                    args,
                    "stage_b_dense_duty_allow_incidental_trace_edits",
                    False,
                )
            )
            model.stage_b_fixed_text_scorer = StageBDenseDutyScorer(
                model.feat_map,
                model.transformer.encoder,
                model.transformer.decoder,
                getattr(model.transformer, "level_embed", None),
                max_text_len=int(getattr(args, "max_text_len", 256)),
                candidate_topk=model.stage_b_v11_candidate_topk,
                category_gate_max_gap=float(
                    getattr(args, "stage_b_dense_duty_category_gate_max_gap", 3.0)
                ),
                patch_score_clip=float(
                    getattr(args, "stage_b_dense_duty_patch_score_clip", 5.0)
                ),
                confidence_adapter_dim=int(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_adapter_dim",
                        64,
                    )
                ),
                confidence_init_seed=int(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_init_seed",
                        42,
                    )
                ),
                confidence_hidden_dim=int(
                    getattr(args, "stage_b_dense_duty_confidence_hidden_dim", 256)
                ),
                confidence_pool_temperature=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_pool_temperature",
                        0.2,
                    )
                ),
                confidence_pool_topk=int(
                    getattr(args, "stage_b_dense_duty_confidence_pool_topk", 10)
                ),
                confidence_phrase_aggregation=confidence_phrase_aggregation,
                confidence_word_softmin_temperature=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_word_softmin_temperature",
                        0.1,
                    )
                ),
                confidence_veto_gate_scale=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_gate_scale",
                        1.0,
                    )
                ),
                confidence_veto_gate_offset=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_gate_offset",
                        0.0,
                    )
                ),
                confidence_veto_coverage_offset=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_coverage_offset",
                        0.1,
                    )
                ),
                confidence_veto_coverage_ramp=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_coverage_ramp",
                        0.8,
                    )
                ),
                confidence_veto_cap_temperature=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_cap_temperature",
                        0.1,
                    )
                ),
                confidence_veto_cap_initial_ceiling=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_cap_initial_ceiling",
                        -0.1,
                    )
                ),
                confidence_rank_evidence_contract=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_rank_evidence_contract",
                        "off_v1",
                    )
                ),
                confidence_pool_feature_contract=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_pool_feature_contract",
                        "patch_statistics_only_v1",
                    )
                ),
                confidence_residual_parameterization_gain=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_residual_parameterization_gain",
                        1.0,
                    )
                ),
                confidence_gate_gradient_contract=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_gate_gradient_contract",
                        "hard_detached_v1",
                    )
                ),
                confidence_head_gradient_contract=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_head_gradient_contract",
                        "shared_token_veto_global_absolute_v1",
                    )
                ),
                confidence_full_decoder_verifier=bool(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_full_decoder_verifier",
                        False,
                    )
                ),
                confidence_veto_only_patch_softmin=bool(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_only_patch_softmin",
                        False,
                    )
                ),
                confidence_candidate_trace_contract=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_candidate_trace_contract",
                        "off_v1",
                    )
                ),
                confidence_token_depth_base_scale=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_token_depth_base_scale",
                        1.0,
                    )
                ),
                confidence_rank_decoder_unfreeze_last_n=int(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_rank_decoder_unfreeze_last_n",
                        0,
                    )
                ),
                expression_microbatch=int(
                    getattr(args, "stage_b_v11_expression_microbatch", 1)
                ),
                phase=str(getattr(args, "stage_b_dense_duty_phase", "rank")),
            )
        else:
            from .stage_b_fixed_text_scorer import FixedBoxFullTextScorer

            model.stage_b_fixed_text_scorer = FixedBoxFullTextScorer(
                model.transformer.decoder,
                num_layers=int(getattr(args, "stage_b_v11_num_layers", 3)),
                max_text_len=int(getattr(args, "max_text_len", 256)),
                expression_microbatch=int(
                    getattr(args, "stage_b_v11_expression_microbatch", 8)
                ),
                use_validity_head=bool(
                    getattr(args, "stage_b_v14_validity_head", False)
                ),
                decouple_validity_from_ranking=bool(
                    getattr(args, "stage_b_v15_decoupled_confidence", False)
                ),
                validity_pool_temperature=float(
                    getattr(args, "stage_b_v15_validity_pool_temperature", 0.2)
                ),
                patch_rank_fusion=bool(
                    getattr(args, "stage_b_v15_patch_rank_fusion", False)
                ),
                patch_rank_weight=float(
                    getattr(args, "stage_b_v15_patch_rank_weight", 1.0)
                ),
                exclude_canonical_from_score=(
                    model.stage_b_v15_exclude_canonical_from_score
                ),
                candidate_topk=model.stage_b_v11_candidate_topk,
                confidence_output_mode=str(
                    getattr(
                        args,
                        "stage_b_v16_confidence_output_mode",
                        "base_plus_gate",
                    )
                ),
                explicit_confidence_output_contract=bool(
                    getattr(
                        args,
                        "stage_b_v19_explicit_confidence_output_contract",
                        False,
                    )
                ),
                score_ownership=str(
                    getattr(args, "stage_b_v22_score_ownership", "")
                ),
            )
    elif stage_b_v7:
        from .stage_b_v7 import StageBVerifier

        model.stage_b_verifier = StageBVerifier.from_groundingdino(
            model,
            canonical_token_weight=float(getattr(args, "stage_b_v7_canonical_token_weight", 0.15)),
            candidate_residual_init=bool(getattr(args, "stage_b_v7_candidate_residual_init", True)),
            phrase_agg=str(getattr(args, "stage_b_v7_phrase_agg", "mean")),
            phrase_mean_weight=float(getattr(args, "stage_b_v7_phrase_mean_weight", 0.5)),
            phrase_softmin_tau=float(getattr(args, "stage_b_v7_phrase_softmin_tau", 0.5)),
            use_joint_phrase_head=bool(getattr(args, "stage_b_v7_use_joint_phrase_head", True)),
            candidate_topk=int(getattr(args, "stage_b_v7_candidate_topk", 50)),
            patch_prior_weight=float(getattr(args, "stage_b_v7_patch_prior_weight", 0.0)),
            context_scale=float(getattr(args, "stage_b_v7_context_scale", 2.0)),
            use_neighbor_geometry=bool(getattr(args, "stage_b_v7_use_neighbor_geometry", False)),
        )

    if patch_only:
        patch_matching = str(getattr(args, "patch_matching", "hungarian")).lower().strip()
        if (stage_b or stage_b_v7 or stage_b_v11) and patch_matching != "hungarian":
            raise ValueError("Stage B requires patch_matching='hungarian'.")

        if stage_b_legacy_global_gate:
            from .stage_b_legacy_global_gate import LegacyStageBGlobalGateCriterion

            criterion = LegacyStageBGlobalGateCriterion(
                absolute_weight=float(
                    getattr(args, "stage_b_legacy_global_gate_absolute_weight", 1.0)
                ),
                pair_weight=float(
                    getattr(args, "stage_b_legacy_global_gate_pair_weight", 1.0)
                ),
                tail_weight=float(
                    getattr(args, "stage_b_legacy_global_gate_tail_weight", 1.0)
                ),
                pair_margin=float(
                    getattr(args, "stage_b_legacy_global_gate_pair_margin", 0.3)
                ),
                tail_margin=float(
                    getattr(args, "stage_b_legacy_global_gate_tail_margin", 0.3)
                ),
                loss_temperature=float(
                    getattr(args, "stage_b_legacy_global_gate_loss_temperature", 0.1)
                ),
                tail_fraction=float(
                    getattr(args, "stage_b_legacy_global_gate_tail_fraction", 0.05)
                ),
                tail_objective=str(
                    getattr(args, "stage_b_legacy_global_gate_tail_objective", "cvar")
                ),
                require_proposalset_proxy_verified=bool(
                    getattr(
                        args,
                        "stage_b_legacy_global_gate_require_proposalset_proxy_verified",
                        True,
                    )
                ),
            )
        elif stage_b_v11:
            from .stage_b_fixed_text_criterion import StageBFixedTextCriterion

            criterion = StageBFixedTextCriterion(
                positive_iou_threshold=float(
                    getattr(args, "stage_b_v11_positive_iou_threshold", 0.5)
                ),
                negative_iou_threshold=float(
                    getattr(args, "stage_b_v11_negative_iou_threshold", 0.3)
                ),
                listwise_temperature=float(
                    getattr(args, "stage_b_v11_listwise_temperature", 0.2)
                ),
                listwise_weight=float(
                    getattr(args, "stage_b_v11_listwise_weight", 1.0)
                ),
                local_tn_rank_margin=float(
                    getattr(args, "stage_b_v11_local_tn_rank_margin", 0.3)
                ),
                local_tn_rank_weight=float(
                    getattr(args, "stage_b_v11_local_tn_rank_weight", 1.0)
                ),
                predicate_tn_rank_margin=float(
                    getattr(args, "stage_b_v11_predicate_tn_rank_margin", 0.3)
                ),
                predicate_tn_rank_weight=float(
                    getattr(args, "stage_b_v11_predicate_tn_rank_weight", 0.0)
                ),
                local_anchor_weight=float(
                    getattr(args, "stage_b_v11_local_anchor_weight", 0.5)
                ),
                positive_anchor_logit=float(
                    getattr(args, "stage_b_v11_positive_anchor_logit", 0.5)
                ),
                negative_anchor_logit=float(
                    getattr(args, "stage_b_v11_negative_anchor_logit", -0.5)
                ),
                global_tn_negative_weight=float(
                    getattr(args, "stage_b_v11_global_tn_negative_weight", 0.0)
                ),
                global_tn_tail_weight=float(
                    getattr(args, "stage_b_v11_global_tn_tail_weight", 0.0)
                ),
                global_tn_tail_topk=int(
                    getattr(args, "stage_b_v11_global_tn_tail_topk", 10)
                ),
                global_tn_tail_temperature=float(
                    getattr(args, "stage_b_v11_global_tn_tail_temperature", 0.2)
                ),
                global_tn_tail_target_logit=float(
                    getattr(args, "stage_b_v11_global_tn_tail_target_logit", 0.0)
                ),
                batch_tail_separation_weight=float(
                    getattr(args, "stage_b_v11_batch_tail_separation_weight", 0.0)
                ),
                batch_positive_quantile=float(
                    getattr(args, "stage_b_v11_batch_positive_quantile", 0.05)
                ),
                batch_negative_quantile=float(
                    getattr(args, "stage_b_v11_batch_negative_quantile", 0.95)
                ),
                batch_tail_margin=float(
                    getattr(args, "stage_b_v11_batch_tail_margin", 0.3)
                ),
                balance_local_anchor_classes=bool(
                    getattr(
                        args,
                        "stage_b_v11_balance_local_anchor_classes",
                        False,
                    )
                ),
                batch_tail_ddp_global=bool(
                    getattr(args, "stage_b_v11_batch_tail_ddp_global", False)
                ),
                local_absolute_weight=float(
                    getattr(args, "stage_b_v14_local_absolute_weight", 0.0)
                ),
                local_absolute_gamma=float(
                    getattr(args, "stage_b_v14_local_absolute_gamma", 0.0)
                ),
                deployed_global_absolute_weight=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_weight",
                        0.0,
                    )
                ),
                deployed_global_absolute_gamma=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_global_absolute_gamma",
                        0.0,
                    )
                ),
                predicate_absolute_weight=float(
                    getattr(args, "stage_b_v14_predicate_absolute_weight", 0.0)
                ),
                predicate_absolute_gamma=float(
                    getattr(args, "stage_b_v14_predicate_absolute_gamma", 0.0)
                ),
                tail_queue_weight=float(
                    getattr(args, "stage_b_v14_tail_queue_weight", 0.0)
                ),
                tail_queue_size=int(
                    getattr(args, "stage_b_v14_tail_queue_size", 0)
                ),
                tail_queue_min_count=int(
                    getattr(args, "stage_b_v14_tail_queue_min_count", 0)
                ),
                tail_queue_positive_quantile=float(
                    getattr(args, "stage_b_v14_tail_queue_positive_quantile", 0.05)
                ),
                tail_queue_negative_quantile=float(
                    getattr(args, "stage_b_v14_tail_queue_negative_quantile", 0.95)
                ),
                tail_queue_temperature=float(
                    getattr(args, "stage_b_v14_tail_queue_temperature", 0.1)
                ),
                tail_queue_margin=float(
                    getattr(args, "stage_b_v14_tail_queue_margin", 0.3)
                ),
                tail_queue_global_scores=bool(
                    getattr(args, "stage_b_v15_tail_queue_global_scores", False)
                ),
                tail_queue_objective=str(
                    getattr(args, "stage_b_v15_tail_queue_objective", "cvar")
                ),
                tail_queue_pair_weight=float(
                    getattr(args, "stage_b_v15_tail_queue_pair_weight", 0.0)
                ),
                tail_queue_pair_margin=float(
                    getattr(args, "stage_b_v15_tail_queue_pair_margin", 0.0)
                ),
                tail_queue_positive_trust_weight=float(
                    getattr(
                        args,
                        "stage_b_v15_tail_queue_positive_trust_weight",
                        0.0,
                    )
                ),
                tail_queue_positive_trust_margin=float(
                    getattr(
                        args,
                        "stage_b_v15_tail_queue_positive_trust_margin",
                        0.02,
                    )
                ),
                tail_queue_positive_trust_reduction_contract=str(
                    getattr(
                        args,
                        "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                        "mean_v1",
                    )
                ),
                tail_queue_positive_gradient_contract=str(
                    getattr(
                        args,
                        "stage_b_v15_tail_queue_positive_gradient_contract",
                        "mean_translation_v1",
                    )
                ),
                token_objective=str(
                    getattr(args, "stage_b_v21_token_objective", "off")
                ),
                token_weight=float(
                    getattr(args, "stage_b_v21_token_weight", 0.0)
                ),
                token_positive_weight=float(
                    getattr(args, "stage_b_v21_token_positive_weight", 1.0)
                ),
                token_shared_weight=float(
                    getattr(args, "stage_b_v21_token_shared_weight", 0.25)
                ),
                token_edit_weight=float(
                    getattr(args, "stage_b_v21_token_edit_weight", 1.0)
                ),
                token_edit_query_scope=str(
                    getattr(
                        args,
                        "stage_b_v21_token_edit_query_scope",
                        "target_iou_v1",
                    )
                ),
                candidate_depth_all_weight=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_all_weight",
                        0.0,
                    )
                ),
                candidate_depth_escape_weight=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_escape_weight",
                        0.0,
                    )
                ),
                candidate_depth_positive_weight=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_positive_weight",
                        0.0,
                    )
                ),
                candidate_depth_tn_margin=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_tn_margin",
                        0.5,
                    )
                ),
                candidate_depth_escape_margin=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_escape_margin",
                        0.5,
                    )
                ),
                candidate_depth_positive_max=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_positive_max",
                        0.05,
                    )
                ),
                candidate_depth_temperature=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_candidate_depth_temperature",
                        0.1,
                    )
                ),
                token_focal_alpha=float(
                    getattr(args, "stage_b_v21_token_focal_alpha", 0.25)
                ),
                token_focal_gamma=float(
                    getattr(args, "stage_b_v21_token_focal_gamma", 2.0)
                ),
                allow_legacy_token_diff_fallback=bool(
                    getattr(
                        args,
                        "stage_b_v21_allow_legacy_token_diff_fallback",
                        False,
                    )
                ),
                raw_veto_gate_weight=float(
                    getattr(args, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
                ),
                raw_veto_positive_margin=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_positive_margin",
                        0.1,
                    )
                ),
                raw_veto_tn_margin=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_tn_margin",
                        0.1,
                    )
                ),
                raw_veto_query_scope=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_query_scope",
                        "target_iou_v1",
                    )
                ),
                raw_veto_tn_carrier_balance=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_tn_carrier_balance",
                        0.0,
                    )
                ),
                raw_veto_positive_carrier_balance=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_positive_carrier_balance",
                        0.0,
                    )
                ),
                raw_veto_carrier_pair_weight=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_carrier_pair_weight",
                        0.0,
                    )
                ),
                raw_veto_carrier_pair_margin=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_carrier_pair_margin",
                        0.25,
                    )
                ),
                raw_veto_carrier_pair_gradient_contract=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                        "bidirectional_v1",
                    )
                ),
                raw_veto_gate_offset=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_gate_offset",
                        0.0,
                    )
                ),
                raw_veto_gate_scale=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_confidence_veto_gate_scale",
                        1.0,
                    )
                ),
                raw_veto_tail_quantile=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_tail_quantile",
                        0.95,
                    )
                ),
                raw_veto_tail_temperature=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_tail_temperature",
                        0.1,
                    )
                ),
                raw_veto_tail_min_count=int(
                    getattr(
                        args,
                        "stage_b_dense_duty_raw_veto_tail_min_count",
                        256,
                    )
                ),
                deployed_veto_routing_weight=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_veto_routing_weight",
                        0.0,
                    )
                ),
                deployed_veto_positive_max=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_veto_positive_max",
                        0.1,
                    )
                ),
                deployed_veto_tn_min=float(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_veto_tn_min",
                        0.9,
                    )
                ),
                deployed_veto_routing_reduction_contract=str(
                    getattr(
                        args,
                        "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                        "balanced_mean_v1",
                    )
                ),
                tail_queue_negative_reduction_contract=str(
                    getattr(
                        args,
                        "stage_b_v15_tail_queue_negative_reduction_contract",
                        "all_mean_v1",
                    )
                ),
            )
        elif stage_b_v7:
            from .patch_hungarian_criterion import PatchHungarianCriterion
            from .stage_b_v7 import StageBV7Criterion

            matcher = build_matcher(args)
            patch_criterion = PatchHungarianCriterion(
                matcher=matcher,
                weight_dict={
                    "loss_patch_ce": 0.0,
                    "loss_bbox": 0.0,
                    "loss_giou": 0.0,
                },
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                patch_ce_reduction=str(getattr(args, "patch_ce_reduction", "legacy")),
                patch_lambda_neg=float(getattr(args, "patch_lambda_neg", 0.25)),
                patch_ce_neg_topk=int(getattr(args, "patch_ce_neg_topk", 0)),
                patch_ce_neg_topk_ratio=float(getattr(args, "patch_ce_neg_topk_ratio", 0.0)),
                patch_rank_margin=float(getattr(args, "patch_rank_margin", 0.3)),
                patch_rank_hard_negatives=int(getattr(args, "patch_rank_hard_negatives", 16)),
                patch_rank_include_wrong_slots=bool(getattr(args, "patch_rank_include_wrong_slots", True)),
                patch_rank_wrong_slot_weight=float(getattr(args, "patch_rank_wrong_slot_weight", 0.5)),
                patch_ce_positive_only_for_datasets=getattr(args, "patch_ce_positive_only_for_datasets", ()),
            )
            criterion = StageBV7Criterion(
                patch_criterion=patch_criterion,
                min_matched_iou=float(getattr(args, "stage_b_v7_min_matched_iou", 0.5)),
                canonical_token_weight=float(getattr(args, "stage_b_v7_canonical_token_weight", 0.15)),
                tn_token_weight=float(getattr(args, "stage_b_v7_tn_token_weight", getattr(args, "lambda_tn_neg", 1.0))),
                tn_shared_token_weight=float(getattr(args, "stage_b_v7_tn_shared_token_weight", 0.25)),
                phrase_focal_alpha=float(getattr(args, "stage_b_v7_phrase_focal_alpha", getattr(args, "focal_alpha", 0.25))),
                phrase_focal_gamma=float(getattr(args, "stage_b_v7_phrase_focal_gamma", getattr(args, "focal_gamma", 2.0))),
                phrase_focal_coef=float(getattr(args, "stage_b_v7_phrase_focal_coef", 1.0)),
                token_focal_alpha=float(getattr(args, "stage_b_v7_token_focal_alpha", getattr(args, "focal_alpha", 0.25))),
                token_focal_gamma=float(getattr(args, "stage_b_v7_token_focal_gamma", getattr(args, "focal_gamma", 2.0))),
                token_focal_coef=float(getattr(args, "stage_b_v7_token_focal_coef", 0.25)),
                pair_rank_loss_coef=float(getattr(args, "stage_b_v7_pair_rank_loss_coef", 0.0)),
                pair_rank_margin=float(getattr(args, "stage_b_v7_pair_rank_margin", 0.18)),
                pair_rank_topk=int(getattr(args, "stage_b_v7_pair_rank_topk", 10)),
                pair_rank_lse_tau=float(getattr(args, "stage_b_v7_pair_rank_lse_tau", 0.1)),
                tn_pair_rank_loss_coef=float(getattr(args, "stage_b_v7_tn_pair_rank_loss_coef", 0.0)),
                tn_pair_rank_margin=float(getattr(args, "stage_b_v7_tn_pair_rank_margin", 0.3)),
                tn_pair_rank_topk=int(getattr(args, "stage_b_v7_tn_pair_rank_topk", 10)),
                pair_pos_weight=float(getattr(args, "stage_b_v7_pair_pos_weight", 0.0)),
                pair_neg_weight=float(getattr(args, "stage_b_v7_pair_neg_weight", 0.0)),
                negative_iou_max=float(getattr(args, "stage_b_v7_negative_iou_max", 0.3)),
                phrase_hard_negative_topk=int(getattr(args, "stage_b_v7_phrase_hard_negative_topk", 10)),
            )
        elif stage_b:
            from .patch_hungarian_criterion import PatchHungarianCriterion
            from .stage_b_criterion import StageBCriterion

            matcher = build_matcher(args)
            patch_criterion = PatchHungarianCriterion(
                matcher=matcher,
                weight_dict={
                    "loss_patch_ce": 1.0,
                    "loss_bbox": float(getattr(args, "bbox_loss_coef", 0.0)),
                    "loss_giou": float(getattr(args, "giou_loss_coef", 0.0)),
                },
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                patch_ce_reduction=str(getattr(args, "patch_ce_reduction", "legacy")),
                patch_lambda_neg=float(getattr(args, "patch_lambda_neg", 0.25)),
                patch_ce_neg_topk=int(getattr(args, "patch_ce_neg_topk", 0)),
                patch_ce_neg_topk_ratio=float(getattr(args, "patch_ce_neg_topk_ratio", 0.0)),
                patch_rank_margin=float(getattr(args, "patch_rank_margin", 0.3)),
                patch_rank_hard_negatives=int(getattr(args, "patch_rank_hard_negatives", 16)),
                patch_rank_include_wrong_slots=bool(getattr(args, "patch_rank_include_wrong_slots", True)),
                patch_rank_wrong_slot_weight=float(getattr(args, "patch_rank_wrong_slot_weight", 0.5)),
                patch_ce_positive_only_for_datasets=getattr(args, "patch_ce_positive_only_for_datasets", ()),
            )
            criterion = StageBCriterion(
                patch_criterion=patch_criterion,
                lambda_patch=float(getattr(args, "lambda_patch", 1.0)),
                lambda_text=float(getattr(args, "lambda_text", 0.25)),
                canonical_pos_weight=float(getattr(args, "canonical_pos_weight", 1.0)),
                stage_b_text_loss_type=str(getattr(args, "stage_b_text_loss_type", "matched_bce")),
                stage_b_text_focal_alpha=float(getattr(args, "stage_b_text_focal_alpha", args.focal_alpha)),
                stage_b_text_focal_gamma=float(getattr(args, "stage_b_text_focal_gamma", args.focal_gamma)),
                stage_b_extra_iou_match_thr=float(getattr(args, "stage_b_extra_iou_match_thr", 0.5)),
                stage_b_tn_neg_weight=float(getattr(args, "lambda_tn_neg", getattr(args, "stage_b_tn_neg_weight", 1.0))),
                stage_b_tn_content_weight=float(getattr(args, "lambda_tn_content", getattr(args, "stage_b_tn_content_weight", 1.0))),
                stage_b_tn_canonical_weight=float(getattr(args, "lambda_tn_canonical", getattr(args, "stage_b_tn_canonical_weight", 1.0))),
                stage_b_tn_neg_weight_mode=getattr(args, "stage_b_tn_neg_weight_mode", "fixed"),
                stage_b_tn_content_target=float(getattr(args, "tn_content_target", getattr(args, "stage_b_tn_content_target", 1.0))),
                stage_b_tn_canonical_target=float(getattr(args, "tn_canonical_target", getattr(args, "stage_b_tn_canonical_target", 1.0))),
                stage_b_rank_margin=float(getattr(args, "stage_b_rank_margin", 0.3)),
                stage_b_rank_loss_coef=float(getattr(args, "stage_b_rank_loss_coef", 0.0)),
                stage_b_rank_detach_patch=bool(getattr(args, "stage_b_rank_detach_patch", True)),
                stage_b_rank_beta=float(getattr(args, "stage_b_infer_text_beta", 1.0)),
                stage_b_rank_canonical_weight=float(getattr(args, "stage_b_infer_canonical_weight", 1.0)),
                stage_b_rank_text_agg=str(getattr(args, "stage_b_infer_text_agg", "mean")),
                stage_b_rank_softmin_tau=float(getattr(args, "stage_b_infer_softmin_tau", getattr(args, "softmin_tau", 0.7))),
                stage_b_rank_mean_softmin_alpha=float(getattr(args, "stage_b_infer_mean_softmin_alpha", 0.5)),
                stage_b_score_mode=str(getattr(args, "stage_b_score_mode", "patch_text")),
                stage_b_score_calib_loss_coef=float(getattr(args, "stage_b_score_calib_loss_coef", 0.0)),
                stage_b_score_calib_tau_pos=float(getattr(args, "stage_b_score_calib_tau_pos", 0.1)),
                stage_b_score_calib_tau_neg=float(getattr(args, "stage_b_score_calib_tau_neg", 1.4)),
                stage_b_score_calib_margin=float(getattr(args, "stage_b_score_calib_margin", 0.3)),
                stage_b_score_calib_topk=int(getattr(args, "stage_b_score_calib_topk", 10)),
                stage_b_score_calib_pos_weight=float(getattr(args, "stage_b_score_calib_pos_weight", 0.1)),
                stage_b_score_calib_neg_weight=float(getattr(args, "stage_b_score_calib_neg_weight", 0.5)),
                stage_b_score_calib_gap_weight=float(getattr(args, "stage_b_score_calib_gap_weight", 0.1)),
                stage_b_score_calib_pos_query_weight=float(getattr(args, "stage_b_score_calib_pos_query_weight", 0.1)),
                stage_b_score_calib_all_tn_neg_weight=float(getattr(args, "stage_b_score_calib_all_tn_neg_weight", 0.0)),
                stage_b_score_calib_detach_patch=bool(getattr(args, "stage_b_score_calib_detach_patch", True)),
                stage_b_score_calib_neg_agg=str(getattr(args, "stage_b_score_calib_neg_agg", "mean")),
                stage_b_score_calib_neg_lse_tau=float(getattr(args, "stage_b_score_calib_neg_lse_tau", 0.5)),
                stage_b_score_calib_aux_loss=bool(getattr(args, "stage_b_score_calib_aux_loss", False)),
                stage_b_aux_loss_start_idx=int(getattr(args, "stage_b_aux_loss_start_idx", 0)),
            )
        else:
            weight_dict = {
                "loss_patch_ce": float(getattr(args, "patch_ce_coef", 1.0)),
                "loss_bbox": float(getattr(args, "bbox_loss_coef", 0.0)),
                "loss_giou": float(getattr(args, "giou_loss_coef", 0.0)),
            }
            patch_rank_loss_coef = float(getattr(args, "patch_rank_loss_coef", 0.0))
            if patch_rank_loss_coef > 0:
                weight_dict["loss_patch_rank"] = patch_rank_loss_coef
            if bool(getattr(args, "aux_loss", False)):
                aux_weight_dict = {}
                for i in range(args.dec_layers - 1):
                    for key, value in list(weight_dict.items()):
                        aux_weight_dict[f"{key}_{i}"] = value
                weight_dict.update(aux_weight_dict)
            if patch_matching == "hungarian":
                from .patch_hungarian_criterion import PatchHungarianCriterion

                matcher = build_matcher(args)
                criterion = PatchHungarianCriterion(
                    matcher=matcher,
                    weight_dict=weight_dict,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                    patch_ce_reduction=str(getattr(args, "patch_ce_reduction", "legacy")),
                    patch_lambda_neg=float(getattr(args, "patch_lambda_neg", 0.25)),
                    patch_ce_neg_topk=int(getattr(args, "patch_ce_neg_topk", 0)),
                    patch_ce_neg_topk_ratio=float(getattr(args, "patch_ce_neg_topk_ratio", 0.0)),
                    patch_rank_margin=float(getattr(args, "patch_rank_margin", 0.3)),
                    patch_rank_hard_negatives=int(getattr(args, "patch_rank_hard_negatives", 16)),
                    patch_rank_include_wrong_slots=bool(getattr(args, "patch_rank_include_wrong_slots", True)),
                    patch_rank_wrong_slot_weight=float(getattr(args, "patch_rank_wrong_slot_weight", 0.5)),
                    patch_ce_positive_only_for_datasets=getattr(args, "patch_ce_positive_only_for_datasets", ()),
                )
            elif patch_matching == "iou":
                from .patch_only_criterion import PatchOnlyCriterion

                criterion = PatchOnlyCriterion(
                    weight_dict=weight_dict,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                    patch_iou_thr=float(getattr(args, "patch_iou_thr", 0.5)),
                    patch_lambda_neg=float(getattr(args, "patch_lambda_neg", 0.25)),
                    patch_labeling_mode=str(getattr(args, "patch_labeling_mode", "iou_thr")),
                    patch_topk=int(getattr(args, "patch_topk", 50)),
                    patch_topk_iou_thr=float(getattr(args, "patch_topk_iou_thr", 0.05)),
                )
            else:
                raise ValueError("patch_matching must be 'hungarian' or 'iou'.")
        criterion.to(device)
        if stage_b:
            postprocessors = {
                "bbox": PostProcessStageB(
                    num_select=args.num_select,
                    nms_iou_threshold=args.nms_iou_threshold,
                    beta=float(getattr(args, "stage_b_infer_text_beta", 1.0)),
                    canonical_weight=float(getattr(args, "stage_b_infer_canonical_weight", 1.0)),
                    text_agg=str(getattr(args, "stage_b_infer_text_agg", "mean")),
                    softmin_tau=float(getattr(args, "stage_b_infer_softmin_tau", getattr(args, "softmin_tau", 0.7))),
                    mean_softmin_alpha=float(getattr(args, "stage_b_infer_mean_softmin_alpha", 0.5)),
                    output_sigmoid_scores=bool(getattr(args, "stage_b_infer_sigmoid_scores", False)),
                    normalize_fused_score=bool(getattr(args, "stage_b_infer_normalize_fused_score", True)),
                    score_mode=str(getattr(args, "stage_b_score_mode", "patch_text")),
                )
            }
        else:
            postprocessors = {}
    else:
        if stage_b_native_patch_category:
            native_patch_objective = str(
                getattr(args, "stage_b_native_patch_objective", "d1_raw_margin")
                or ""
            ).strip().lower()
            if native_patch_objective == "d1_raw_margin":
                from .stage_b_native_patch_category import (
                    StageBNativePatchCategoryCriterion,
                )

                criterion = StageBNativePatchCategoryCriterion(
                    patch_weight=float(
                        getattr(args, "stage_b_native_patch_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    margin=float(
                        getattr(args, "stage_b_native_patch_margin", 0.1)
                    ),
                    temperature=float(
                        getattr(args, "stage_b_native_patch_temperature", 0.1)
                    ),
                )
            elif native_patch_objective == "d2_gate_aligned":
                from .stage_b_native_patch_category_d2 import (
                    StageBNativePatchCategoryD2Criterion,
                )

                criterion = StageBNativePatchCategoryD2Criterion(
                    weight=float(
                        getattr(args, "stage_b_native_patch_d2_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d2_keep_gap", 2.75)
                    ),
                    drop_gap=float(
                        getattr(args, "stage_b_native_patch_d2_drop_gap", 3.25)
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d2_temperature",
                            0.25,
                        )
                    ),
                    native_hard_negatives=int(
                        getattr(
                            args,
                            "stage_b_native_patch_d2_native_hard_negatives",
                            16,
                        )
                    ),
                    patch_hard_negatives=int(
                        getattr(
                            args,
                            "stage_b_native_patch_d2_patch_hard_negatives",
                            4,
                        )
                    ),
                    keep_weight=float(
                        getattr(
                            args, "stage_b_native_patch_d2_keep_weight", 2.0
                        )
                    ),
                    drop_weight=float(
                        getattr(
                            args, "stage_b_native_patch_d2_drop_weight", 1.0
                        )
                    ),
                    coverage_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d2_coverage_weight",
                            0.25,
                        )
                    ),
                )
            elif native_patch_objective == "d3_critical_winner":
                from .stage_b_native_patch_category_d3 import (
                    StageBNativePatchCategoryD3Criterion,
                )

                criterion = StageBNativePatchCategoryD3Criterion(
                    weight=float(
                        getattr(args, "stage_b_native_patch_d3_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d3_keep_gap", 2.75)
                    ),
                    separation_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d3_separation_gap",
                            3.25,
                        )
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d3_temperature",
                            0.25,
                        )
                    ),
                    critical_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d3_critical_weight",
                            2.0,
                        )
                    ),
                    critical_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d3_critical_keep_weight",
                            1.0,
                        )
                    ),
                    positive_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d3_positive_keep_weight",
                            1.0,
                        )
                    ),
                )
            elif (
                native_patch_objective
                == "d4_positive_protected_critical_winner"
            ):
                from .stage_b_native_patch_category_d4 import (
                    StageBNativePatchCategoryD4Criterion,
                )

                criterion = StageBNativePatchCategoryD4Criterion(
                    weight=float(
                        getattr(args, "stage_b_native_patch_d4_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d4_keep_gap", 2.75)
                    ),
                    separation_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d4_separation_gap",
                            3.25,
                        )
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d4_temperature",
                            0.25,
                        )
                    ),
                    critical_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d4_critical_weight",
                            2.0,
                        )
                    ),
                    critical_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d4_critical_keep_weight",
                            1.0,
                        )
                    ),
                    positive_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d4_positive_keep_weight",
                            32.0,
                        )
                    ),
                )
            elif native_patch_objective == "d5_active_tail_positive_barrier":
                from .stage_b_native_patch_category_d5 import (
                    StageBNativePatchCategoryD5Criterion,
                )

                criterion = StageBNativePatchCategoryD5Criterion(
                    weight=float(
                        getattr(args, "stage_b_native_patch_d5_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d5_keep_gap", 2.75)
                    ),
                    separation_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d5_separation_gap",
                            3.25,
                        )
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d5_temperature",
                            0.25,
                        )
                    ),
                    critical_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d5_critical_weight",
                            2.0,
                        )
                    ),
                    critical_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d5_critical_keep_weight",
                            1.0,
                        )
                    ),
                    active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d5_active_gap",
                            2.0,
                        )
                    ),
                    target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d5_target_gap",
                            2.5,
                        )
                    ),
                    positive_barrier_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d5_positive_barrier_weight",
                            2.0,
                        )
                    ),
                )
            elif native_patch_objective == "d6_direct_deployment_gap":
                from .stage_b_native_patch_category_d6 import (
                    StageBNativePatchCategoryD6Criterion,
                )

                criterion = StageBNativePatchCategoryD6Criterion(
                    weight=float(
                        getattr(args, "stage_b_native_patch_d6_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d6_keep_gap", 2.75)
                    ),
                    drop_gap=float(
                        getattr(args, "stage_b_native_patch_d6_drop_gap", 3.25)
                    ),
                    drop_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d6_drop_active_gap",
                            3.75,
                        )
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d6_temperature",
                            0.25,
                        )
                    ),
                    drop_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d6_drop_weight",
                            2.0,
                        )
                    ),
                    critical_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d6_critical_keep_weight",
                            1.0,
                        )
                    ),
                    positive_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d6_positive_active_gap",
                            2.0,
                        )
                    ),
                    positive_target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d6_positive_target_gap",
                            2.5,
                        )
                    ),
                    positive_barrier_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d6_positive_barrier_weight",
                            2.0,
                        )
                    ),
                )
            elif native_patch_objective == "d7_all_state_positive_anchor":
                from .stage_b_native_patch_category_d7 import (
                    StageBNativePatchCategoryD7Criterion,
                )

                criterion = StageBNativePatchCategoryD7Criterion(
                    weight=float(
                        getattr(args, "stage_b_native_patch_d7_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d7_keep_gap", 2.75)
                    ),
                    drop_gap=float(
                        getattr(args, "stage_b_native_patch_d7_drop_gap", 3.25)
                    ),
                    drop_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_drop_active_gap",
                            3.75,
                        )
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_temperature",
                            0.25,
                        )
                    ),
                    drop_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_drop_weight",
                            2.0,
                        )
                    ),
                    critical_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_critical_keep_weight",
                            1.0,
                        )
                    ),
                    positive_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_positive_active_gap",
                            2.0,
                        )
                    ),
                    positive_target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_positive_target_gap",
                            2.5,
                        )
                    ),
                    positive_barrier_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_positive_barrier_weight",
                            2.0,
                        )
                    ),
                    anchor_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_anchor_active_gap",
                            2.0,
                        )
                    ),
                    anchor_target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_anchor_target_gap",
                            2.5,
                        )
                    ),
                    anchor_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d7_anchor_weight",
                            2.0,
                        )
                    ),
                )
            elif native_patch_objective == "d8_state_class_macro_anchor":
                from .stage_b_native_patch_category_d8 import (
                    StageBNativePatchCategoryD8Criterion,
                )

                criterion = StageBNativePatchCategoryD8Criterion(
                    weight=float(
                        getattr(args, "stage_b_native_patch_d8_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d8_keep_gap", 2.75)
                    ),
                    drop_gap=float(
                        getattr(args, "stage_b_native_patch_d8_drop_gap", 3.25)
                    ),
                    drop_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_drop_active_gap",
                            3.75,
                        )
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_temperature",
                            0.25,
                        )
                    ),
                    drop_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_drop_weight",
                            2.0,
                        )
                    ),
                    critical_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_critical_keep_weight",
                            1.0,
                        )
                    ),
                    positive_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_positive_active_gap",
                            2.0,
                        )
                    ),
                    positive_target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_positive_target_gap",
                            2.5,
                        )
                    ),
                    positive_barrier_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_positive_barrier_weight",
                            2.0,
                        )
                    ),
                    anchor_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_active_gap",
                            2.0,
                        )
                    ),
                    anchor_target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_target_gap",
                            2.5,
                        )
                    ),
                    negative_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_negative_weight",
                            1.0,
                        )
                    ),
                    neutral_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_neutral_weight",
                            2.0,
                        )
                    ),
                    positive_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_positive_weight",
                            4.0,
                        )
                    ),
                )
            elif native_patch_objective == "d9_loss_gradient_localized":
                from .stage_b_native_patch_category_d9 import (
                    StageBNativePatchCategoryD9Criterion,
                )

                criterion = StageBNativePatchCategoryD9Criterion(
                    detach_row_stats=getattr(
                        args,
                        "stage_b_native_patch_d9_detach_row_stats",
                        None,
                    ),
                    weight=float(
                        getattr(args, "stage_b_native_patch_d8_weight", 1.0)
                    ),
                    positive_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_positive_iou_threshold",
                            0.5,
                        )
                    ),
                    negative_iou_threshold=float(
                        getattr(
                            args,
                            "stage_b_native_patch_negative_iou_threshold",
                            0.3,
                        )
                    ),
                    gate_max_gap=float(
                        getattr(args, "stage_b_native_patch_gate_max_gap", 3.0)
                    ),
                    patch_score_clip=float(
                        getattr(args, "stage_b_native_patch_score_clip", 5.0)
                    ),
                    keep_gap=float(
                        getattr(args, "stage_b_native_patch_d8_keep_gap", 2.75)
                    ),
                    drop_gap=float(
                        getattr(args, "stage_b_native_patch_d8_drop_gap", 3.25)
                    ),
                    drop_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_drop_active_gap",
                            3.75,
                        )
                    ),
                    temperature=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_temperature",
                            0.25,
                        )
                    ),
                    drop_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_drop_weight",
                            2.0,
                        )
                    ),
                    critical_keep_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_critical_keep_weight",
                            1.0,
                        )
                    ),
                    positive_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_positive_active_gap",
                            2.0,
                        )
                    ),
                    positive_target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_positive_target_gap",
                            2.5,
                        )
                    ),
                    positive_barrier_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_positive_barrier_weight",
                            2.0,
                        )
                    ),
                    anchor_active_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_active_gap",
                            2.0,
                        )
                    ),
                    anchor_target_gap=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_target_gap",
                            2.5,
                        )
                    ),
                    negative_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_negative_weight",
                            1.0,
                        )
                    ),
                    neutral_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_neutral_weight",
                            2.0,
                        )
                    ),
                    positive_weight=float(
                        getattr(
                            args,
                            "stage_b_native_patch_d8_anchor_positive_weight",
                            4.0,
                        )
                    ),
                )
            else:
                raise ValueError(
                    "stage_b_native_patch_objective must be exactly "
                    "'d1_raw_margin', 'd2_gate_aligned', or "
                    "'d3_critical_winner', or "
                    "'d4_positive_protected_critical_winner', or "
                    "'d5_active_tail_positive_barrier', or "
                    "'d6_direct_deployment_gap', or "
                    "'d7_all_state_positive_anchor', or "
                    "'d8_state_class_macro_anchor', or "
                    "'d9_loss_gradient_localized'"
                )
            criterion.to(device)
            postprocessors = {
                "bbox": PostProcess(
                    num_select=args.num_select,
                    text_encoder_type=args.text_encoder_type,
                    nms_iou_threshold=args.nms_iou_threshold,
                    args=args,
                )
            }
            return model, criterion, postprocessors
        if stage_b_data_driven_score:
            from .stage_b_data_driven_score import StageBDataDrivenCriterion

            criterion = StageBDataDrivenCriterion(
                train_mode=str(
                    getattr(
                        args,
                        "stage_b_data_driven_train_mode",
                        "rank_patch_only",
                    )
                ),
                category_complete=bool(
                    getattr(
                        args,
                        "stage_b_data_driven_category_complete",
                        False,
                    )
                ),
                rank_supervision=str(
                    getattr(
                        args,
                        "stage_b_data_driven_rank_supervision",
                        "all_nonpositive_negative_v1",
                    )
                ),
                tokenizer=model.tokenizer,
                max_text_len=int(getattr(args, "max_text_len", 256)),
                rank_weight=float(
                    getattr(args, "stage_b_data_driven_rank_weight", 1.0)
                ),
                assignment_weight=float(
                    getattr(
                        args,
                        "stage_b_data_driven_assignment_weight",
                        0.0,
                    )
                ),
                deployment_weight=float(
                    getattr(
                        args,
                        "stage_b_data_driven_deployment_weight",
                        0.0,
                    )
                ),
                patch_weight=float(
                    getattr(args, "stage_b_data_driven_patch_weight", 1.0)
                ),
                confidence_weight=float(
                    getattr(
                        args, "stage_b_data_driven_confidence_weight", 1.0
                    )
                ),
                token_weight=float(
                    getattr(args, "stage_b_data_driven_token_weight", 0.0)
                ),
                shared_token_weight=float(
                    getattr(
                        args, "stage_b_data_driven_shared_token_weight", 0.25
                    )
                ),
                positive_iou_threshold=float(
                    getattr(
                        args,
                        "stage_b_data_driven_positive_iou_threshold",
                        0.5,
                    )
                ),
                rank_negative_iou_threshold=float(
                    getattr(
                        args,
                        "stage_b_data_driven_rank_negative_iou_threshold",
                        0.3,
                    )
                ),
                patch_negative_iou_threshold=float(
                    getattr(
                        args,
                        "stage_b_data_driven_patch_negative_iou_threshold",
                        0.3,
                    )
                ),
                temperature=float(
                    getattr(args, "stage_b_data_driven_temperature", 0.1)
                ),
                rank_margin=float(
                    getattr(args, "stage_b_data_driven_rank_margin", 0.1)
                ),
                category_margin=float(
                    getattr(args, "stage_b_data_driven_category_margin", 0.1)
                ),
                category_gate_max_gap=float(
                    getattr(
                        args,
                        "stage_b_data_driven_category_gate_max_gap",
                        3.0,
                    )
                ),
                category_gate_boundary_margin=float(
                    getattr(
                        args,
                        "stage_b_data_driven_category_gate_boundary_margin",
                        0.25,
                    )
                ),
                patch_active_unsafe_auxiliary_weight=float(
                    getattr(
                        args,
                        "stage_b_data_driven_patch_active_unsafe_auxiliary_weight",
                        1.0,
                    )
                ),
                patch_dense_category_focal_weight=float(
                    getattr(
                        args,
                        "stage_b_data_driven_patch_dense_category_focal_weight",
                        1.0,
                    )
                ),
                patch_dense_category_focal_alpha=float(
                    getattr(
                        args,
                        "stage_b_data_driven_patch_dense_category_focal_alpha",
                        0.25,
                    )
                ),
                patch_dense_category_focal_gamma=float(
                    getattr(
                        args,
                        "stage_b_data_driven_patch_dense_category_focal_gamma",
                        2.0,
                    )
                ),
                patch_dense_category_focal_negative_weight=float(
                    getattr(
                        args,
                        "stage_b_data_driven_patch_dense_category_focal_negative_weight",
                        0.25,
                    )
                ),
                patch_drop_positive_anchor_gradient_policy=str(
                    getattr(
                        args,
                        "stage_b_data_driven_patch_drop_positive_anchor_gradient_policy",
                        "global_max_positive_v1",
                    )
                ),
                patch_score_clip=float(
                    getattr(args, "stage_b_data_driven_patch_score_clip", 5.0)
                ),
                fpr_temperature=float(
                    getattr(
                        args, "stage_b_data_driven_fpr_temperature", 0.1
                    )
                ),
                fpr_margin=float(
                    getattr(args, "stage_b_data_driven_fpr_margin", 0.0)
                ),
                target_tpr=float(
                    getattr(args, "stage_b_data_driven_target_tpr", 0.95)
                ),
                positive_queue_size=int(
                    getattr(
                        args, "stage_b_data_driven_positive_queue_size", 4096
                    )
                ),
            )
            criterion.to(device)
            postprocessors = {
                "bbox": PostProcess(
                    num_select=args.num_select,
                    text_encoder_type=args.text_encoder_type,
                    nms_iou_threshold=args.nms_iou_threshold,
                    args=args,
                )
            }
            return model, criterion, postprocessors
        if stage_b_gdino_score_adapter:
            if stage_b_u0_patch_rank:
                if bool(getattr(args, "stage_b_u0_gate_aligned_d13", False)):
                    from .stage_b_u0_gate_aligned_d13 import (
                        StageBU0GateAlignedD13Criterion,
                    )

                    criterion = StageBU0GateAlignedD13Criterion(
                        weight=float(
                            getattr(args, "stage_b_u0_d13_weight", 1.0)
                        ),
                        positive_iou_threshold=float(
                            getattr(
                                args,
                                "stage_b_u0_d13_positive_iou_threshold",
                                0.5,
                            )
                        ),
                        negative_iou_threshold=float(
                            getattr(
                                args,
                                "stage_b_u0_d13_negative_iou_threshold",
                                0.3,
                            )
                        ),
                        gate_max_gap=float(
                            getattr(args, "stage_b_u0_category_gate_max_gap", 2.0)
                        ),
                        keep_gap=float(
                            getattr(args, "stage_b_u0_d13_keep_gap", 1.95)
                        ),
                        drop_gap=float(
                            getattr(args, "stage_b_u0_d13_drop_gap", 2.05)
                        ),
                        preserve_tolerance=float(
                            getattr(
                                args,
                                "stage_b_u0_d13_preserve_tolerance",
                                0.02,
                            )
                        ),
                        temperature=float(
                            getattr(args, "stage_b_u0_d13_temperature", 0.05)
                        ),
                        keep_weight=float(
                            getattr(args, "stage_b_u0_d13_keep_weight", 1.0)
                        ),
                        drop_weight=float(
                            getattr(args, "stage_b_u0_d13_drop_weight", 1.0)
                        ),
                        preserve_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d13_preserve_weight",
                                4.0,
                            )
                        ),
                        residual_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d13_residual_weight",
                                0.05,
                            )
                        ),
                    )
                elif bool(getattr(args, "stage_b_u0_gate_aligned_d12", False)):
                    from .stage_b_u0_gate_aligned_d12 import (
                        StageBU0GateAlignedD12Criterion,
                    )

                    criterion = StageBU0GateAlignedD12Criterion(
                        weight=float(
                            getattr(args, "stage_b_u0_d12_weight", 1.0)
                        ),
                        positive_iou_threshold=float(
                            getattr(
                                args,
                                "stage_b_u0_d12_positive_iou_threshold",
                                0.5,
                            )
                        ),
                        fix_margin=float(
                            getattr(args, "stage_b_u0_d12_fix_margin", 0.05)
                        ),
                        preserve_tolerance=float(
                            getattr(
                                args,
                                "stage_b_u0_d12_preserve_tolerance",
                                0.01,
                            )
                        ),
                        preserve_floor=float(
                            getattr(
                                args,
                                "stage_b_u0_d12_preserve_floor",
                                0.005,
                            )
                        ),
                        temperature=float(
                            getattr(args, "stage_b_u0_d12_temperature", 0.05)
                        ),
                        fix_weight=float(
                            getattr(args, "stage_b_u0_d12_fix_weight", 1.0)
                        ),
                        preserve_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d12_preserve_weight",
                                1.0,
                            )
                        ),
                        residual_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d12_residual_weight",
                                0.01,
                            )
                        ),
                    )
                elif bool(getattr(args, "stage_b_u0_gate_aligned_d11", False)):
                    from .stage_b_u0_gate_aligned_d11 import (
                        StageBU0GateAlignedD11Criterion,
                    )

                    criterion = StageBU0GateAlignedD11Criterion(
                        weight=float(
                            getattr(args, "stage_b_u0_d11_weight", 1.0)
                        ),
                        positive_iou_threshold=float(
                            getattr(
                                args,
                                "stage_b_u0_d11_positive_iou_threshold",
                                0.5,
                            )
                        ),
                        fix_margin=float(
                            getattr(args, "stage_b_u0_d11_fix_margin", 0.05)
                        ),
                        preserve_margin=float(
                            getattr(
                                args,
                                "stage_b_u0_d11_preserve_margin",
                                0.02,
                            )
                        ),
                        temperature=float(
                            getattr(args, "stage_b_u0_d11_temperature", 0.05)
                        ),
                        fix_weight=float(
                            getattr(args, "stage_b_u0_d11_fix_weight", 1.0)
                        ),
                        preserve_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d11_preserve_weight",
                                1.0,
                            )
                        ),
                    )
                elif bool(getattr(args, "stage_b_u0_gate_aligned_d10", False)):
                    from .stage_b_u0_gate_aligned_d10 import (
                        StageBU0GateAlignedD10Criterion,
                    )

                    criterion = StageBU0GateAlignedD10Criterion(
                        weight=float(
                            getattr(args, "stage_b_u0_d10_weight", 1.0)
                        ),
                        positive_iou_threshold=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_positive_iou_threshold",
                                0.5,
                            )
                        ),
                        negative_iou_threshold=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_negative_iou_threshold",
                                0.3,
                            )
                        ),
                        gate_max_gap=float(
                            getattr(args, "stage_b_u0_d10_gate_max_gap", 2.0)
                        ),
                        patch_score_clip=float(
                            getattr(args, "stage_b_u0_d10_patch_score_clip", 5.0)
                        ),
                        keep_gap=float(
                            getattr(args, "stage_b_u0_d10_keep_gap", 1.75)
                        ),
                        drop_gap=float(
                            getattr(args, "stage_b_u0_d10_drop_gap", 2.25)
                        ),
                        drop_active_gap=float(
                            getattr(args, "stage_b_u0_d10_drop_active_gap", 2.75)
                        ),
                        temperature=float(
                            getattr(args, "stage_b_u0_d10_temperature", 0.25)
                        ),
                        max_rank_blockers=int(
                            getattr(args, "stage_b_u0_d10_max_rank_blockers", 4)
                        ),
                        drop_weight=float(
                            getattr(args, "stage_b_u0_d10_drop_weight", 2.0)
                        ),
                        critical_keep_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_critical_keep_weight",
                                1.0,
                            )
                        ),
                        positive_active_gap=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_positive_active_gap",
                                1.25,
                            )
                        ),
                        positive_target_gap=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_positive_target_gap",
                                1.5,
                            )
                        ),
                        positive_barrier_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_positive_barrier_weight",
                                2.0,
                            )
                        ),
                        instance_active_gap=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_instance_active_gap",
                                1.25,
                            )
                        ),
                        instance_target_gap=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_instance_target_gap",
                                1.5,
                            )
                        ),
                        instance_coverage_weight=float(
                            getattr(
                                args,
                                "stage_b_u0_d10_instance_coverage_weight",
                                2.0,
                            )
                        ),
                    )
                else:
                    from .stage_b_u0_patch_rank import StageBU0PatchRankCriterion

                    criterion = StageBU0PatchRankCriterion(
                        weight=float(
                            getattr(args, "stage_b_u0_patch_rank_weight", 1.0)
                        ),
                        iou_threshold=float(
                            getattr(args, "stage_b_u0_positive_iou_threshold", 0.5)
                        ),
                        fix_margin=float(
                            getattr(args, "stage_b_u0_fix_margin", 0.05)
                        ),
                        preserve_margin=float(
                            getattr(args, "stage_b_u0_preserve_margin", 0.02)
                        ),
                        temperature=float(
                            getattr(args, "stage_b_u0_rank_temperature", 0.1)
                        ),
                        residual_weight=float(
                            getattr(args, "stage_b_u0_residual_weight", 1e-3)
                        ),
                        category_complete_supervision=bool(
                            getattr(
                                args,
                                "stage_b_u2_category_complete_supervision",
                                False,
                            )
                        ),
                        category_loss_weight=float(
                            getattr(args, "stage_b_u2_category_loss_weight", 0.0)
                        ),
                        category_negative_iou_threshold=float(
                            getattr(
                                args,
                                "stage_b_u2_category_negative_iou_threshold",
                                0.3,
                            )
                        ),
                        category_margin=float(
                            getattr(args, "stage_b_u2_category_margin", 0.1)
                        ),
                        target_preserve_weight=float(
                            getattr(args, "stage_b_u2_target_preserve_weight", 1.0)
                        ),
                    )
                criterion.to(device)
                postprocessors = {
                    "bbox": PostProcess(
                        num_select=args.num_select,
                        text_encoder_type=args.text_encoder_type,
                        nms_iou_threshold=args.nms_iou_threshold,
                        args=args,
                    )
                }
                return model, criterion, postprocessors
            from .stage_b_gdino_score_adapter import (
                StageBGDINOScoreAdapterCriterion,
            )

            criterion = StageBGDINOScoreAdapterCriterion(
                tn_scope=str(getattr(args, "stage_b_gdino_tn_scope", "")),
                train_mode=str(
                    getattr(args, "stage_b_gdino_adapter_train_mode", "joint")
                ),
                confidence_objective=str(
                    getattr(
                        args,
                        "stage_b_gdino_confidence_objective",
                        "queue_q05_st",
                    )
                ),
                positive_iou_threshold=float(
                    getattr(args, "stage_b_gdino_positive_iou_threshold", 0.5)
                ),
                negative_iou_threshold=float(
                    getattr(args, "stage_b_gdino_negative_iou_threshold", 0.5)
                ),
                listwise_temperature=float(
                    getattr(args, "stage_b_gdino_listwise_temperature", 0.2)
                ),
                rank_fix_margin=float(
                    getattr(args, "stage_b_gdino_rank_fix_margin", 0.05)
                ),
                rank_preserve_margin=float(
                    getattr(args, "stage_b_gdino_rank_preserve_margin", 0.02)
                ),
                rank_residual_weight=float(
                    getattr(args, "stage_b_gdino_rank_residual_weight", 1e-3)
                ),
                rank_weight=float(
                    getattr(args, "stage_b_gdino_rank_weight", 1.0)
                ),
                confidence_weight=float(
                    getattr(args, "stage_b_gdino_confidence_weight", 1.0)
                ),
                fpr_temperature=float(
                    getattr(args, "stage_b_gdino_fpr_temperature", 0.1)
                ),
                fpr_margin=float(
                    getattr(args, "stage_b_gdino_fpr_margin", 0.0)
                ),
                paired_margin_weight=float(
                    getattr(args, "stage_b_gdino_paired_margin_weight", 0.25)
                ),
                paired_margin=float(
                    getattr(args, "stage_b_gdino_paired_margin", 0.05)
                ),
                positive_trust_margin=float(
                    getattr(args, "stage_b_gdino_positive_trust_margin", 0.02)
                ),
                positive_trust_weight=float(
                    getattr(args, "stage_b_gdino_positive_trust_weight", 1.0)
                ),
                queue_size=int(
                    getattr(args, "stage_b_gdino_queue_size", 4096)
                ),
                queue_min_count=int(
                    getattr(args, "stage_b_gdino_queue_min_count", 256)
                ),
            )
            criterion.to(device)
            postprocessors = {
                "bbox": PostProcess(
                    num_select=args.num_select,
                    text_encoder_type=args.text_encoder_type,
                    nms_iou_threshold=args.nms_iou_threshold,
                    args=args,
                )
            }
            return model, criterion, postprocessors
        matcher = build_matcher(args)

        # prepare weight dict
        weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        clean_weight_dict = copy.deepcopy(weight_dict)

        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if args.two_stage_type != 'no':
            interm_weight_dict = {}
            try:
                no_interm_box_loss = args.no_interm_box_loss
            except:
                no_interm_box_loss = False
            _coeff_weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
                'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            }
            try:
                interm_loss_coef = args.interm_loss_coef
            except:
                interm_loss_coef = 1.0
            interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
            weight_dict.update(interm_weight_dict)

        gdino_tn_alltn_weight = float(getattr(args, "gdino_tn_alltn_weight", 0.0))
        if gdino_tn_alltn_weight > 0:
            weight_dict["loss_tn_alltn"] = gdino_tn_alltn_weight
            if args.aux_loss:
                for i in range(args.dec_layers - 1):
                    weight_dict[f"loss_tn_alltn_{i}"] = gdino_tn_alltn_weight
            if args.two_stage_type != 'no':
                try:
                    interm_loss_coef = args.interm_loss_coef
                except:
                    interm_loss_coef = 1.0
                weight_dict["loss_tn_alltn_interm"] = gdino_tn_alltn_weight * interm_loss_coef

        gdino_tn_token_neg_weight = float(getattr(args, "gdino_tn_token_neg_weight", getattr(args, "lambda_tn_neg", 0.0)))
        gdino_tn_token_content_weight = float(getattr(args, "gdino_tn_token_content_weight", getattr(args, "lambda_tn_content", 0.0)))
        gdino_tn_token_canonical_weight = float(getattr(args, "gdino_tn_token_canonical_weight", getattr(args, "lambda_tn_canonical", 0.0)))
        gdino_tn_token_enabled = (
            gdino_tn_token_neg_weight > 0
            or gdino_tn_token_content_weight > 0
            or gdino_tn_token_canonical_weight > 0
        )
        if gdino_tn_token_enabled:
            weight_dict["loss_tn_tokens"] = 1.0
            if args.aux_loss:
                for i in range(args.dec_layers - 1):
                    weight_dict[f"loss_tn_tokens_{i}"] = 1.0
            if args.two_stage_type != 'no':
                try:
                    interm_loss_coef = args.interm_loss_coef
                except:
                    interm_loss_coef = 1.0
                weight_dict["loss_tn_tokens_interm"] = interm_loss_coef

        # losses = ['labels', 'boxes', 'cardinality']
        losses = ['labels', 'boxes']

        criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                                 focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,losses=losses,
                                 gdino_tn_loss_type=getattr(args, "gdino_tn_loss_type", "dense_focal"),
                                 gdino_tn_alltn_weight=gdino_tn_alltn_weight,
                                 gdino_tn_alltn_topk=getattr(args, "gdino_tn_alltn_topk", 10),
                                 gdino_tn_alltn_tau_neg=getattr(args, "gdino_tn_alltn_tau_neg", -2.4),
                                 gdino_tn_alltn_lse_tau=getattr(args, "gdino_tn_alltn_lse_tau", 0.2),
                                 gdino_tn_alltn_text_agg=getattr(args, "gdino_tn_alltn_text_agg", "mean"),
                                 gdino_tn_token_neg_weight=gdino_tn_token_neg_weight,
                                 gdino_tn_token_content_weight=gdino_tn_token_content_weight,
                                 gdino_tn_token_canonical_weight=gdino_tn_token_canonical_weight,
                                 gdino_tn_token_neg_weight_mode=getattr(args, "gdino_tn_token_neg_weight_mode", "fixed"),
                                 )
        criterion.to(device)
        postprocessors = {'bbox': PostProcess(num_select=args.num_select  , text_encoder_type=args.text_encoder_type,nms_iou_threshold=args.nms_iou_threshold,args=args)}

    return model, criterion, postprocessors

def create_positive_map(tokenized, tokens_positive,cat_list,caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j,label in enumerate(tokens_positive):

        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:
                end_pos = None
        # except Exception as e:
        #     print("beg:", beg, "end:", end)
        #     print("token_positive:", tokens_positive)
        #     # print("beg_pos:", beg_pos, "end_pos:", end_pos)
        #     raise e
        # if beg_pos is None:
        #     try:
        #         beg_pos = tokenized.char_to_token(beg + 1)
        #         if beg_pos is None:
        #             beg_pos = tokenized.char_to_token(beg + 2)
        #     except:
        #         beg_pos = None
        # if end_pos is None:
        #     try:
        #         end_pos = tokenized.char_to_token(end - 2)
        #         if end_pos is None:
        #             end_pos = tokenized.char_to_token(end - 3)
        #     except:
        #         end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j,beg_pos: end_pos + 1].fill_(1)
    return positive_map 
