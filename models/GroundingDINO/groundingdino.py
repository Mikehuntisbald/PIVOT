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
from typing import Dict, List, Optional, Union

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

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        
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
                score_patch = logit_scale * torch.einsum("bqd,bd->bq", query_proj, patch_global)  # (B,Q)
                score_for_fuse = score_patch
                alpha_base = patch_gate  # (B,) or None
            elif patch_global.dim() == 3:
                score_patch = logit_scale * torch.einsum("bqd,bkd->bqk", query_proj, patch_global)  # (B,Q,K)
                if patch_mask_in is not None:
                    if (not torch.is_tensor(patch_mask_in)) or patch_mask_in.dim() != 2:
                        raise ValueError("patch_mask must be a bool tensor of shape (B,K) for multi-patch.")
                    score_patch = score_patch.masked_fill(~patch_mask_in[:, None, :].to(torch.bool), -100.0)
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
                if alpha_base is None:
                    alpha = torch.full((score_for_fuse.shape[0], 1), 0.5, device=score_for_fuse.device)
                else:
                    alpha = alpha_base.unsqueeze(-1)
                fused_logits = (1 - alpha) * text_logits + alpha * score_for_fuse.unsqueeze(-1)

        out = {
            "pred_logits": fused_logits,
            "pred_logits_text": text_logits,
            "pred_logits_patch": score_patch,
            "pred_boxes": outputs_coord_list[-1],
        }
        if patch_mask_in is not None:
            out["patch_mask"] = patch_mask_in
        if (score_patch is not None) and (not patch_only) and ((patches is not None) or (patch_global_in is not None)):
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
            "phrase_semantic_token_mask",
            "tn_group_ids",
        ):
            mask_value = kw.get(mask_key, None)
            if mask_value is not None:
                if torch.is_tensor(mask_value):
                    mask_value = mask_value.to(samples.device)
                    if mask_value.dim() == 2 and bs == 1:
                        mask_value = mask_value.unsqueeze(0)
                out[mask_key] = mask_value
        if patch_mask_in is not None and isinstance(text_dict, dict) and ("phrase_mask" in text_dict):
            pm = patch_mask_in.to(torch.bool)
            tm = text_dict["phrase_mask"].to(torch.bool)
            if pm.shape[0] == tm.shape[0]:
                if pm.shape[1] != tm.shape[1]:
                    raise ValueError(
                        f"patch_mask (B,K)={tuple(pm.shape)} and phrase_mask (B,P)={tuple(tm.shape)} mismatch; "
                        "make sure Stage A captions repeat 'object .' per patch."
                    )
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
        if self.aux_loss and (not patch_only):
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




class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha,focal_gamma, losses):
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
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(outputs['text_mask'].shape[0],-1,outputs['text_mask'].shape[1])
            pred_logits = torch.masked_select(pred_logits, text_mask)
            new_targets = torch.masked_select(new_targets, text_mask)

        new_targets=new_targets.float()
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
        canonical_weight: float = 0.15,
        text_agg: str = "mean",
        softmin_tau: float = 0.7,
        mean_softmin_alpha: float = 0.5,
        output_sigmoid_scores: bool = True,
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
    patch_gate_with_text = bool(getattr(args, "patch_gate_with_text", not patch_only))
    if patch_only:
        patch_gate_with_text = False
    enable_patch_branch = bool(getattr(args, "enable_patch_branch", patch_only or stage_b))

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


    if patch_only:
        patch_matching = str(getattr(args, "patch_matching", "hungarian")).lower().strip()
        if stage_b and patch_matching != "hungarian":
            raise ValueError("Stage B requires patch_matching='hungarian'.")

        if stage_b:
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
            )
            criterion = StageBCriterion(
                patch_criterion=patch_criterion,
                lambda_patch=float(getattr(args, "lambda_patch", 1.0)),
                lambda_text=float(getattr(args, "lambda_text", 0.25)),
                canonical_pos_weight=float(getattr(args, "canonical_pos_weight", 0.15)),
                stage_b_rank_margin=float(getattr(args, "stage_b_rank_margin", 0.3)),
                stage_b_rank_loss_coef=float(getattr(args, "stage_b_rank_loss_coef", 0.0)),
                stage_b_rank_detach_patch=bool(getattr(args, "stage_b_rank_detach_patch", True)),
                stage_b_rank_beta=float(getattr(args, "stage_b_infer_text_beta", 1.0)),
                stage_b_rank_canonical_weight=float(getattr(args, "stage_b_infer_canonical_weight", 0.15)),
                stage_b_rank_text_agg=str(getattr(args, "stage_b_infer_text_agg", "mean")),
                stage_b_rank_softmin_tau=float(getattr(args, "stage_b_infer_softmin_tau", getattr(args, "softmin_tau", 0.7))),
                stage_b_rank_mean_softmin_alpha=float(getattr(args, "stage_b_infer_mean_softmin_alpha", 0.5)),
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
                    canonical_weight=float(getattr(args, "stage_b_infer_canonical_weight", 0.15)),
                    text_agg=str(getattr(args, "stage_b_infer_text_agg", "mean")),
                    softmin_tau=float(getattr(args, "stage_b_infer_softmin_tau", getattr(args, "softmin_tau", 0.7))),
                    mean_softmin_alpha=float(getattr(args, "stage_b_infer_mean_softmin_alpha", 0.5)),
                    output_sigmoid_scores=bool(getattr(args, "stage_b_infer_sigmoid_scores", True)),
                )
            }
        else:
            postprocessors = {}
    else:
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

        # losses = ['labels', 'boxes', 'cardinality']
        losses = ['labels', 'boxes']

        criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                                 focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,losses=losses
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
