import copy
import types
import unittest

import torch
from torch import nn

from models.GroundingDINO.stage_b_dense_duty_scorer import (
    AbsoluteConfidencePool,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT,
    CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_SHARED,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO,
    CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_JOINT_CLIP,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETOES,
    CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_PENALTY,
    CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC,
    CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION,
    CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE,
    CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE,
    CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH,
    DenseExpressionTower,
    StageBDenseDutyScorer,
    TokenAwareConfidenceAdapter,
    _frozen_reference_carrier_index,
    _frozen_reference_carrier_gate,
    _masked_score_statistics,
    _word_normalized_softmin_probability,
)
from models.GroundingDINO.transformer import TransformerDecoder


class _FakeVisualLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, value):
        return value + 0.05 * self.proj(value)


class _FakeTextLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, value):
        return value + 0.1 * self.proj(value)


class _FakeFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.visual = nn.Linear(hidden_dim, hidden_dim)
        self.text = nn.Linear(hidden_dim, hidden_dim)

    def forward_for_test(self, visual, text):
        text_summary = self.text(text).mean(dim=1, keepdim=True)
        visual_summary = self.visual(visual).mean(dim=1, keepdim=True)
        return visual + 0.1 * text_summary, text + 0.1 * visual_summary


class _FakeEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeVisualLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.text_layers = nn.ModuleList(
            [_FakeTextLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.fusion_layers = nn.ModuleList(
            [_FakeFusionLayer(hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        src,
        *,
        pos,
        level_start_index,
        spatial_shapes,
        valid_ratios,
        key_padding_mask,
        memory_text,
        text_attention_mask,
        position_ids,
        text_self_attention_masks,
    ):
        del (
            pos,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            key_padding_mask,
            text_attention_mask,
            position_ids,
            text_self_attention_masks,
        )
        visual = src
        text = memory_text
        for visual_layer, text_layer, fusion_layer in zip(
            self.layers, self.text_layers, self.fusion_layers
        ):
            visual, text = fusion_layer.forward_for_test(visual, text)
            text = text_layer(text)
            visual = visual_layer(visual)
        return visual, text


class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, scale: float) -> None:
        super().__init__()
        self.use_text_cross_attention = True
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward_for_test(self, target, memory, memory_text):
        text_summary = memory_text.mean(dim=1).unsqueeze(0)
        image_summary = memory.mean(dim=1).unsqueeze(0)
        return target + self.scale * (
            0.1 * self.query(target) + text_summary + 0.2 * image_summary
        )


class _FakeDecoder(TransformerDecoder):
    def __init__(self, hidden_dim: int = 4, num_layers: int = 2) -> None:
        nn.Module.__init__(self)
        self.layers = nn.ModuleList(
            [
                _FakeDecoderLayer(hidden_dim, float(index + 1) / 10.0)
                for index in range(num_layers)
            ]
        )
        self.num_layers = int(num_layers)
        self.d_model = int(hidden_dim)
        self.return_intermediate = True
        self.norm = nn.LayerNorm(hidden_dim)
        self.ref_point_head = nn.Linear(4, hidden_dim)
        self.bbox_embed = nn.ModuleList(
            [nn.Linear(hidden_dim, 4) for _ in range(num_layers)]
        )
        self.class_embed = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        refpoints_unsigmoid=None,
        level_start_index=None,
        spatial_shapes=None,
        valid_ratios=None,
        memory_text=None,
        text_attention_mask=None,
    ):
        del (
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            text_attention_mask,
        )
        output = tgt
        outputs = []
        for layer in self.layers:
            output = layer.forward_for_test(output, memory.transpose(0, 1), memory_text)
            outputs.append(self.norm(output).transpose(0, 1))
        reference = refpoints_unsigmoid.sigmoid().transpose(0, 1)
        references = [reference]
        if self.bbox_embed is not None:
            references.append((reference + 0.01).clamp(max=1.0))
        return outputs, references


class _ContextProvider:
    def __init__(self, hidden_dim: int = 4, bert_dim: int = 6, token_count: int = 3):
        self.hidden_dim = int(hidden_dim)
        self.bert_dim = int(bert_dim)
        self.token_count = int(token_count)
        self.leaves = []
        self.call_sizes = []

    def __call__(self, captions, owners):
        count = len(captions)
        self.call_sizes.append(count)
        caption_signal = torch.tensor(
            [sum(ord(char) for char in caption) % 17 for caption in captions],
            device=owners.device,
            dtype=torch.float32,
        )
        bert_leaf = (
            torch.arange(
                self.token_count * self.bert_dim,
                device=owners.device,
                dtype=torch.float32,
            )
            .view(1, self.token_count, self.bert_dim)
            .expand(count, -1, -1)
            .clone()
            .requires_grad_(True)
        )
        bert = bert_leaf + caption_signal[:, None, None] / 100.0
        image_pattern = torch.arange(
            self.hidden_dim * 4,
            device=owners.device,
            dtype=torch.float32,
        ).view(1, self.hidden_dim, 2, 2)
        src_leaf = (
            image_pattern.expand(count, -1, -1, -1)
            .clone()
            .requires_grad_(True)
        )
        src = src_leaf + owners.float()[:, None, None, None] / 10.0
        pos_leaf = (
            (image_pattern / 100.0)
            .expand(count, -1, -1, -1)
            .clone()
            .requires_grad_(True)
        )
        pos = pos_leaf + owners.float()[:, None, None, None] / 1000.0
        self.leaves.extend((bert_leaf, src_leaf, pos_leaf))
        text_mask = torch.ones(
            count, self.token_count, dtype=torch.bool, device=owners.device
        )
        return {
            "bert_hidden": bert,
            "text_token_mask": text_mask,
            "position_ids": torch.arange(
                self.token_count, device=owners.device
            )[None].expand(count, -1),
            "text_self_attention_masks": torch.ones(
                count,
                self.token_count,
                self.token_count,
                dtype=torch.bool,
                device=owners.device,
            ),
            "phrase_token_mask": text_mask,
            "srcs": [src],
            "masks": [
                torch.zeros(count, 2, 2, dtype=torch.bool, device=owners.device)
            ],
            "poss": [pos],
        }


class _FakeGroundingDINO(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.feat_map = nn.Linear(6, 4)
            self.transformer = nn.Module()
            self.transformer.encoder = _FakeEncoder()
            self.transformer.decoder = _FakeDecoder()
            self.transformer.level_embed = nn.Parameter(torch.randn(1, 4))


class StageBDenseDutyScorerTest(unittest.TestCase):
    def _scorer(
        self,
        *,
        phase="rank",
        source_seed=1,
        category_gate_max_gap=100.0,
        confidence_phrase_aggregation="legacy_prob_mean_add_v1",
        **confidence_options,
    ):
        source = _FakeGroundingDINO(source_seed)
        scorer = StageBDenseDutyScorer(
            source.feat_map,
            source.transformer.encoder,
            source.transformer.decoder,
            source.transformer.level_embed,
            max_text_len=8,
            candidate_topk=3,
            category_gate_max_gap=category_gate_max_gap,
            patch_score_clip=5.0,
            confidence_adapter_dim=3,
            confidence_hidden_dim=7,
            confidence_pool_topk=2,
            confidence_phrase_aggregation=confidence_phrase_aggregation,
            phase=phase,
            **confidence_options,
        )
        return scorer, source

    def _inputs(self, *, requires_grad=False):
        torch.manual_seed(7)
        batch_size, candidate_count, hidden_dim, slots = 2, 3, 4, 2
        candidate_hs = torch.randn(
            batch_size,
            candidate_count,
            hidden_dim,
            requires_grad=requires_grad,
        )
        boxes = torch.rand(batch_size, candidate_count, 4).clamp(0.05, 0.95)
        boxes.requires_grad_(requires_grad)
        indices = torch.tensor([[4, 0, 2], [5, 2, 1]], dtype=torch.long)
        patch = torch.tensor(
            [[4.0, 3.0, 2.0], [2.0, 3.0, 1.0]],
            requires_grad=requires_grad,
        )
        captions = [["red person", "blue person"], ["left cup", "right cup"]]
        valid = torch.tensor([[True, True], [True, False]])
        score_mask = torch.zeros(batch_size, slots, 8, dtype=torch.bool)
        score_mask[:, :, 0] = True
        score_mask[1, 0].zero_()  # Canonical-only/fallback contract.
        return {
            "candidate_hs": candidate_hs,
            "candidate_boxes": boxes,
            "candidate_indices": indices,
            "candidate_patch_logits": patch,
            "expression_captions": captions,
            "expression_valid_mask": valid,
            "expression_score_token_mask": score_mask,
        }

    def _forward(self, scorer, *, requires_grad=False):
        inputs = self._inputs(requires_grad=requires_grad)
        if scorer.confidence_adapter.phrase_aggregation in (
            CONFIDENCE_PHRASE_AGGREGATION_WORD_VETOES
        ):
            score_mask = inputs["expression_score_token_mask"]
            word_groups = torch.full(score_mask.shape, -1, dtype=torch.long)
            word_groups[score_mask] = 0
            inputs["expression_score_word_group_ids"] = word_groups
        provider = _ContextProvider()
        output = scorer(raw_context_provider=provider, **inputs)
        return output, inputs, provider

    def test_eval_shapes_and_fixed_candidate_contract(self):
        scorer, _source = self._scorer(phase="eval")
        scorer.eval()
        output, inputs, _provider = self._forward(scorer)
        self.assertTrue(scorer.is_dense_duty)
        self.assertEqual(tuple(output["candidate_idx"].shape), (2, 3))
        self.assertEqual(tuple(output["candidate_boxes"].shape), (2, 3, 4))
        self.assertEqual(tuple(output["candidate_eligible_mask"].shape), (2, 3, 2))
        self.assertEqual(tuple(output["layer_token_logits"].shape), (2, 2, 3, 2, 8))
        self.assertEqual(tuple(output["final_phrase_logits"].shape), (2, 3, 2))
        self.assertEqual(tuple(output["confidence_token_logits"].shape), (2, 3, 2, 8))
        self.assertEqual(
            tuple(output["final_confidence_token_residual_logits"].shape),
            (2, 3, 2, 8),
        )
        self.assertEqual(tuple(output["final_validity_logits"].shape), (2, 3, 2))
        self.assertEqual(
            tuple(output["final_reference_base_logits"].shape), (2, 3, 2)
        )
        self.assertEqual(tuple(output["final_confidence_global_logits"].shape), (2, 2))
        self.assertEqual(
            tuple(
                output[
                    "final_frozen_rank_full_expression_global_logits"
                ].shape
            ),
            (2, 2),
        )
        for batch_index in range(2):
            for slot_index in range(2):
                if not bool(output["expression_valid_mask"][batch_index, slot_index]):
                    continue
                full_phrase = DenseExpressionTower.aggregate_phrase_logits(
                    output["final_rank_token_logits"][
                        batch_index, :, slot_index
                    ][None],
                    output["expression_token_mask"][
                        batch_index, slot_index
                    ][None],
                )[0]
                eligible = output["candidate_eligible_mask"][
                    batch_index, :, slot_index
                ]
                expected = full_phrase.masked_fill(~eligible, -torch.inf).max()
                self.assertTrue(
                    torch.equal(
                        output[
                            "final_frozen_rank_full_expression_global_logits"
                        ][batch_index, slot_index],
                        expected,
                    )
                )
        self.assertTrue(torch.equal(output["candidate_idx"], inputs["candidate_indices"]))
        self.assertTrue(
            torch.equal(output["candidate_boxes"], inputs["candidate_boxes"].detach())
        )
        self.assertFalse(output["final_phrase_logits"].requires_grad)
        self.assertFalse(output["final_validity_logits"].requires_grad)
        self.assertFalse(output["candidate_eligible_mask"][1, :, 1].any())
        fallback = output["candidate_patch_standardized"][1]
        self.assertTrue(
            torch.equal(output["final_phrase_logits"][1, :, 0], fallback)
        )
        self.assertTrue(output["category_only_patch_fallback_mask"][1, 0])
        self.assertFalse(output["category_only_patch_fallback_mask"][0].any())

    def test_missing_score_mask_fails_closed(self):
        scorer, _source = self._scorer(phase="rank")
        scorer.train()
        inputs = self._inputs()
        inputs["expression_score_token_mask"] = None
        with self.assertRaisesRegex(
            ValueError, "requires a noncanonical score-token mask"
        ):
            scorer(raw_context_provider=_ContextProvider(), **inputs)

    def test_rank_and_confidence_parameters_are_disjoint(self):
        scorer, _source = self._scorer()
        rank_ids = {id(parameter) for parameter in scorer.rank_parameters()}
        confidence_ids = {
            id(parameter) for parameter in scorer.confidence_parameters()
        }
        self.assertTrue(rank_ids)
        self.assertTrue(confidence_ids)
        self.assertFalse(rank_ids & confidence_ids)
        self.assertFalse(hasattr(scorer, "confidence_tower"))

    def test_zero_init_rank_evidence_preserves_u0_and_receives_gradient(self):
        baseline, _source = self._scorer(phase="eval", source_seed=19)
        evidence, _source = self._scorer(
            phase="eval",
            source_seed=19,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE
            ),
        )
        baseline.eval()
        evidence.eval()
        baseline_output, _inputs, _provider = self._forward(baseline)
        evidence_output, _inputs, _provider = self._forward(evidence)
        scale = evidence.confidence_adapter.rank_evidence_residual_scale
        self.assertIsNotNone(scale)
        self.assertEqual(float(scale.detach()), 0.0)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
        ):
            self.assertTrue(
                torch.equal(evidence_output[key], baseline_output[key]), key
            )

        evidence.set_phase("confidence")
        evidence.train()
        output, inputs, provider = self._forward(evidence, requires_grad=True)
        rank_token = output["layer_token_logits"][-1].detach().float()
        rank_token = torch.where(
            torch.isfinite(rank_token), rank_token, torch.zeros_like(rank_token)
        ).clamp(min=-20.0, max=20.0)
        residual = output["final_confidence_token_residual_logits"]
        (residual * rank_token).sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertGreater(float(scale.grad), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in evidence.rank_parameters())
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_zero_init_rank_affine_preserves_u0_and_trains_both_parameters(self):
        baseline, _source = self._scorer(phase="eval", source_seed=23)
        affine, _source = self._scorer(
            phase="eval",
            source_seed=23,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_AFFINE
            ),
        )
        baseline.eval()
        affine.eval()
        baseline_output, _inputs, _provider = self._forward(baseline)
        affine_output, _inputs, _provider = self._forward(affine)
        scale = affine.confidence_adapter.rank_evidence_residual_scale
        bias = affine.confidence_adapter.rank_evidence_residual_bias
        self.assertIsNotNone(scale)
        self.assertIsNotNone(bias)
        self.assertEqual(float(scale.detach()), 0.0)
        self.assertEqual(float(bias.detach()), 0.0)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
        ):
            self.assertTrue(
                torch.equal(affine_output[key], baseline_output[key]), key
            )

        affine.set_phase("confidence")
        affine.train()
        output, inputs, provider = self._forward(affine, requires_grad=True)
        residual = output["final_confidence_token_residual_logits"]
        residual.sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertIsNotNone(bias.grad)
        self.assertNotEqual(float(scale.grad), 0.0)
        self.assertNotEqual(float(bias.grad), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in affine.rank_parameters())
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_gate_margin_parameterization_preserves_u0_and_scales_residual(self):
        baseline, _source = self._scorer(phase="eval", source_seed=29)
        gain = 0.25 / 0.03
        conditioned, _source = self._scorer(
            phase="eval",
            source_seed=29,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_GATE_MARGIN
            ),
            confidence_residual_parameterization_gain=gain,
        )
        baseline.eval()
        conditioned.eval()
        baseline_output, _inputs, _provider = self._forward(baseline)
        conditioned_output, _inputs, _provider = self._forward(conditioned)
        scale = conditioned.confidence_adapter.rank_evidence_residual_scale
        self.assertIsNotNone(scale)
        self.assertIsNone(
            conditioned.confidence_adapter.rank_evidence_residual_bias
        )
        self.assertEqual(float(scale.detach()), 0.0)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
        ):
            self.assertTrue(
                torch.equal(conditioned_output[key], baseline_output[key]), key
            )

        conditioned.set_phase("confidence")
        conditioned.train()
        with torch.no_grad():
            scale.fill_(0.125)
        conditioned_output, inputs, provider = self._forward(
            conditioned, requires_grad=True
        )
        unit, _source = self._scorer(
            phase="confidence",
            source_seed=29,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SCALE
            ),
        )
        unit.train()
        with torch.no_grad():
            unit.confidence_adapter.rank_evidence_residual_scale.fill_(0.125)
        unit_output, _inputs, _provider = self._forward(unit, requires_grad=True)
        self.assertTrue(
            torch.allclose(
                conditioned_output["final_confidence_token_residual_logits"],
                unit_output["final_confidence_token_residual_logits"] * gain,
                rtol=1e-5,
                atol=1e-6,
            )
        )
        conditioned_output[
            "final_confidence_token_residual_logits"
        ].sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertNotEqual(float(scale.grad), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in conditioned.rank_parameters())
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_carrier_token_slope_preserves_u0_and_prior_initialization(self):
        baseline, _source = self._scorer(
            phase="eval",
            source_seed=33,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
        )
        conditioned, _source = self._scorer(
            phase="eval",
            source_seed=33,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE
            ),
            confidence_residual_parameterization_gain=0.25 / 0.03,
        )
        baseline.eval()
        conditioned.eval()
        baseline_state = baseline.state_dict()
        conditioned_state = conditioned.state_dict()
        added_names = sorted(set(conditioned_state).difference(baseline_state))
        self.assertEqual(
            added_names,
            ["confidence_adapter.carrier_rank_slope.weight"],
        )
        self.assertTrue(
            torch.equal(
                conditioned_state[added_names[0]],
                torch.zeros_like(conditioned_state[added_names[0]]),
            )
        )
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(conditioned_state[name], value), name)

        baseline_output, _inputs, _provider = self._forward(baseline)
        conditioned_output, _inputs, _provider = self._forward(conditioned)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_mismatch_gate",
            "final_confidence_veto_sample_gate",
            "final_confidence_veto_carrier_index",
        ):
            self.assertTrue(
                torch.equal(conditioned_output[key], baseline_output[key]), key
            )

    def test_carrier_token_slope_only_changes_carrier_modifier_tokens(self):
        gain = 0.25 / 0.03
        adapter = TokenAwareConfidenceAdapter(
            4,
            adapter_dim=3,
            max_text_len=3,
            patch_hidden_dim=3,
            score_topk=2,
            rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE
            ),
            residual_parameterization_gain=gain,
        )
        rank_token = torch.tensor(
            [
                [
                    [
                        [2.0, -1.0, 4.0],
                        [6.0, 3.0, -2.0],
                        [-4.0, 5.0, 1.0],
                    ],
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ],
                ]
            ]
        )
        query = torch.randn(1, 2, 3, 4, requires_grad=True)
        text = torch.tensor(
            [
                [[1.0, 2.0, 4.0, 8.0], [2.0, 1.0, 3.0, 7.0], [5.0, 4.0, 2.0, 1.0]],
                [[3.0, 1.0, 4.0, 2.0], [6.0, 5.0, 2.0, 1.0], [1.0, 7.0, 3.0, 2.0]],
            ],
            requires_grad=True,
        )
        phrase_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        modifier_mask = torch.tensor(
            [[True, False, True], [False, False, False]]
        )
        patch = torch.tensor([[0.0, 4.0, 1.0], [3.0, 2.0, 1.0]])
        eligible = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        with torch.no_grad():
            adapter.carrier_rank_slope.weight.fill_(0.5)
        output = adapter(
            rank_token_layers=rank_token,
            query_layers=query,
            text_features=text,
            phrase_token_mask=phrase_mask,
            score_token_mask=modifier_mask,
            patch_logits=patch,
            patch_standardized=torch.zeros_like(patch),
            candidate_mask=eligible,
        )

        carrier_index = output["reference_carrier_index_layers"]
        text_latent = adapter.text_projection(adapter.text_norm(text.detach()))
        slope = adapter.carrier_rank_slope(text_latent.detach()).squeeze(-1).float()
        evidence = rank_token.float().clamp(-20.0, 20.0) / 20.0
        carrier_mask = torch.nn.functional.one_hot(
            carrier_index, num_classes=3
        ).float()
        expected = (
            gain
            * carrier_mask[..., None]
            * modifier_mask[None, :, None, :].float()
            * slope[None, :, None, :]
            * evidence
        )
        torch.testing.assert_close(
            output["token_residual_layers"], expected, rtol=1e-6, atol=1e-7
        )
        changed = output["token_residual_layers"].ne(0.0)
        allowed = carrier_mask[..., None].bool() & modifier_mask[
            None, :, None, :
        ]
        self.assertFalse(bool((changed & ~allowed).any().item()))
        self.assertFalse(bool(changed[:, 1].any().item()))

        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.carrier_rank_slope.weight.zero_()
        output = adapter(
            rank_token_layers=rank_token.requires_grad_(),
            query_layers=query,
            text_features=text,
            phrase_token_mask=phrase_mask,
            score_token_mask=modifier_mask,
            patch_logits=patch,
            patch_standardized=torch.zeros_like(patch),
            candidate_mask=eligible,
        )
        carrier_weight = carrier_mask[..., None] * modifier_mask[
            None, :, None, :
        ].float()
        (output["token_residual_layers"] * carrier_weight).sum().backward()
        self.assertIsNotNone(adapter.carrier_rank_slope.weight.grad)
        self.assertTrue(
            bool(adapter.carrier_rank_slope.weight.grad.ne(0.0).any().item())
        )
        self.assertIsNone(rank_token.grad)
        self.assertIsNone(query.grad)
        self.assertIsNone(text.grad)

    def test_carrier_token_affine_adds_only_zero_intercept_at_u0(self):
        slope, _source = self._scorer(
            phase="eval",
            source_seed=35,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_SLOPE
            ),
            confidence_residual_parameterization_gain=0.25 / 0.03,
        )
        affine, _source = self._scorer(
            phase="eval",
            source_seed=35,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE
            ),
            confidence_residual_parameterization_gain=0.25 / 0.03,
        )
        slope.eval()
        affine.eval()
        slope_state = slope.state_dict()
        affine_state = affine.state_dict()
        self.assertEqual(
            sorted(set(affine_state).difference(slope_state)),
            ["confidence_adapter.carrier_rank_slope.bias"],
        )
        for name, value in slope_state.items():
            self.assertTrue(torch.equal(affine_state[name], value), name)
        self.assertTrue(
            torch.equal(
                affine.confidence_adapter.carrier_rank_slope.bias,
                torch.zeros((1,)),
            )
        )
        slope_output, _inputs, _provider = self._forward(slope)
        affine_output, _inputs, _provider = self._forward(affine)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_global_logits",
            "final_confidence_veto_sample_gate",
            "final_confidence_veto_carrier_index",
        ):
            self.assertTrue(torch.equal(affine_output[key], slope_output[key]), key)

        affine.set_phase("confidence")
        affine.train()
        output, inputs, provider = self._forward(affine, requires_grad=True)
        output["final_confidence_token_residual_logits"].sum().backward()
        bias = affine.confidence_adapter.carrier_rank_slope.bias
        self.assertIsNotNone(bias.grad)
        self.assertNotEqual(float(bias.grad), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in affine.rank_parameters())
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_sparse_rank_channel_adds_only_zero_output_surface_at_u0(self):
        gain = 0.25 / 0.03
        affine, _source = self._scorer(
            phase="eval",
            source_seed=37,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_CARRIER_TOKEN_AFFINE
            ),
            confidence_residual_parameterization_gain=gain,
        )
        rank_channel, _source = self._scorer(
            phase="eval",
            source_seed=37,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_residual_parameterization_gain=gain,
        )
        affine.eval()
        rank_channel.eval()
        affine_state = affine.state_dict()
        rank_channel_state = rank_channel.state_dict()
        self.assertEqual(
            sorted(set(rank_channel_state).difference(affine_state)),
            [
                "confidence_adapter.rank_channel_logit_projection.bias",
                "confidence_adapter.rank_channel_logit_projection.weight",
                "confidence_adapter.rank_channel_norm.bias",
                "confidence_adapter.rank_channel_norm.weight",
                "confidence_adapter.rank_channel_output.weight",
                "confidence_adapter.rank_channel_projection.bias",
                "confidence_adapter.rank_channel_projection.weight",
            ],
        )
        for name, value in affine_state.items():
            self.assertTrue(torch.equal(rank_channel_state[name], value), name)
        self.assertEqual(
            int(torch.count_nonzero(rank_channel.confidence_adapter.rank_channel_output.weight)),
            0,
        )

        affine_output, _inputs, _provider = self._forward(affine)
        rank_channel_output, _inputs, _provider = self._forward(rank_channel)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_mismatch_gate",
            "final_confidence_veto_sample_gate",
            "final_confidence_veto_carrier_index",
        ):
            self.assertTrue(
                torch.equal(rank_channel_output[key], affine_output[key]), key
            )

    def test_signed_rank_query_pool_adds_only_six_parameters_and_is_u0_exact(self):
        gain = 0.25 / 0.03
        common = {
            "phase": "eval",
            "source_seed": 41,
            "confidence_phrase_aggregation": (
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            "confidence_rank_evidence_contract": (
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            "confidence_residual_parameterization_gain": gain,
        }
        v19, _source = self._scorer(**common)
        v20, _source = self._scorer(
            **common,
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY
            ),
        )
        v19.eval()
        v20.eval()

        v19_state = v19.state_dict()
        v20_state = v20.state_dict()
        self.assertEqual(
            sorted(set(v20_state).difference(v19_state)),
            [
                "confidence_adapter.global_query_norm.bias",
                "confidence_adapter.global_query_norm.weight",
                "confidence_adapter.global_query_trunk.0.bias",
                "confidence_adapter.global_query_trunk.0.weight",
                "confidence_adapter.global_query_trunk.2.bias",
                "confidence_adapter.global_query_trunk.2.weight",
            ],
        )
        self.assertFalse(set(v19_state).difference(v20_state))
        for name, value in v19_state.items():
            self.assertTrue(torch.equal(v20_state[name], value), name)

        v19_output, _inputs, _provider = self._forward(v19)
        v20_output, _inputs, _provider = self._forward(v20)
        self.assertEqual(tuple(v20_output), tuple(v19_output))
        for key, value in v19_output.items():
            self.assertTrue(torch.equal(v20_output[key], value), key)

    def test_sparse_rank_channel_is_modifier_only_and_stop_gradient(self):
        gain = 0.25 / 0.03
        adapter = TokenAwareConfidenceAdapter(
            4,
            adapter_dim=3,
            max_text_len=3,
            patch_hidden_dim=3,
            score_topk=2,
            rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            residual_parameterization_gain=gain,
        )
        rank_token = torch.randn(1, 2, 3, 3, requires_grad=True)
        query = torch.randn(1, 2, 3, 4, requires_grad=True)
        text = torch.randn(2, 3, 4, requires_grad=True)
        phrase_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        modifier_mask = torch.tensor(
            [[True, False, True], [False, False, False]]
        )
        patch = torch.tensor([[4.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
        eligible = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        with torch.no_grad():
            adapter.rank_channel_output.weight.fill_(0.25)
        output = adapter(
            rank_token_layers=rank_token,
            query_layers=query,
            text_features=text,
            phrase_token_mask=phrase_mask,
            score_token_mask=modifier_mask,
            patch_logits=patch,
            patch_standardized=torch.zeros_like(patch),
            candidate_mask=eligible,
        )
        residual = output["token_residual_layers"]
        allowed = modifier_mask[None, :, None, :].expand_as(residual)
        self.assertFalse(bool(residual.masked_select(~allowed).ne(0.0).any().item()))
        self.assertTrue(bool(residual.masked_select(allowed).ne(0.0).any().item()))

        changed_text = text.detach().clone()
        changed_text[0, 1].add_(1000.0)
        changed_output = adapter(
            rank_token_layers=rank_token,
            query_layers=query,
            text_features=changed_text,
            phrase_token_mask=phrase_mask,
            score_token_mask=modifier_mask,
            patch_logits=patch,
            patch_standardized=torch.zeros_like(patch),
            candidate_mask=eligible,
        )
        self.assertTrue(
            torch.equal(
                residual,
                changed_output["token_residual_layers"],
            )
        )

        residual.sum().backward()
        self.assertIsNotNone(adapter.rank_channel_output.weight.grad)
        self.assertTrue(
            bool(adapter.rank_channel_output.weight.grad.ne(0.0).any().item())
        )
        self.assertIsNotNone(adapter.rank_channel_projection.weight.grad)
        self.assertTrue(
            bool(adapter.rank_channel_projection.weight.grad.ne(0.0).any().item())
        )
        self.assertIsNone(rank_token.grad)
        self.assertIsNone(query.grad)
        self.assertIsNone(text.grad)

    def test_rank_loss_detaches_candidates_patch_and_confidence(self):
        scorer, _source = self._scorer(phase="rank")
        scorer.train()
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        loss = output["final_phrase_logits"][0, :, 0].sum()
        loss.backward()
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            any(parameter.grad is not None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.confidence_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_confidence_loss_cannot_reach_rank_or_frozen_inputs(self):
        scorer, _source = self._scorer(phase="confidence")
        scorer.train()
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        loss = output["final_confidence_global_logits"].sum()
        loss.backward()
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in scorer.confidence_parameters()
            )
        )
        self.assertIsNotNone(scorer.confidence_adapter.patch_residual[-1].weight.grad)
        self.assertTrue(
            any(parameter.grad is not None for parameter in scorer.confidence_pool.parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_confidence_pool_has_finite_gradients_at_zero_variance(self):
        pool = AbsoluteConfidencePool(
            hidden_dim=4,
            pool_hidden_dim=7,
            score_topk=2,
        )
        cases = (
            torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True),
            torch.tensor([[2.0, 2.0, 2.0]], requires_grad=True),
        )
        masks = (
            torch.tensor([[True, False, False]]),
            torch.tensor([[True, True, True]]),
        )
        for dense_logit, candidate_mask in zip(cases, masks):
            with self.subTest(mask=candidate_mask.tolist()):
                dense_feature = torch.randn(1, 3, 4, requires_grad=True)
                confidence, _residual = pool(
                    dense_feature,
                    dense_logit,
                    candidate_mask,
                )
                torch.nn.functional.softplus(confidence).sum().backward()
                self.assertTrue(torch.isfinite(dense_logit.grad).all())
                self.assertTrue(torch.isfinite(dense_feature.grad).all())
                self.assertTrue(
                    all(
                        parameter.grad is None
                        or torch.isfinite(parameter.grad).all()
                        for parameter in pool.parameters()
                    )
                )
                pool.zero_grad(set_to_none=True)

    def test_single_candidate_confidence_scorer_backward_is_finite(self):
        scorer, _source = self._scorer(
            phase="confidence",
            category_gate_max_gap=0.0,
        )
        scorer.train()
        output, inputs, _provider = self._forward(scorer)
        self.assertTrue(
            torch.equal(
                output["candidate_eligible_mask"].sum(dim=1),
                inputs["expression_valid_mask"].long(),
            )
        )
        torch.nn.functional.softplus(
            output["final_confidence_global_logits"]
        ).sum().backward()
        gradients = [
            parameter.grad
            for parameter in scorer.confidence_parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_phrase_aggregation_uses_fp32_under_autocast_inputs(self):
        token_logits = torch.tensor(
            [[[12.0, 12.0], [-12.0, -12.0]]], dtype=torch.float16
        )
        phrase_mask = torch.tensor([[True, True]])
        phrase_logits = DenseExpressionTower.aggregate_phrase_logits(
            token_logits, phrase_mask
        )
        self.assertEqual(phrase_logits.dtype, torch.float32)
        self.assertGreater(float(phrase_logits[0, 0]), 8.0)
        self.assertLess(float(phrase_logits[0, 1]), -8.0)

    def test_confidence_adapter_u0_exactly_inherits_rank_and_patch_contract(self):
        torch.manual_seed(19)
        adapter = TokenAwareConfidenceAdapter(
            4,
            adapter_dim=3,
            max_text_len=5,
            patch_hidden_dim=3,
            score_topk=2,
            patch_score_clip=5.0,
        )
        rank_token = torch.randn(2, 2, 3, 5, dtype=torch.float16)
        query = torch.randn(2, 2, 3, 4)
        text = torch.randn(2, 5, 4)
        phrase_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )
        score_mask = torch.tensor(
            [[False, True, True, False, False], [False, False, False, False, False]]
        )
        patch = torch.tensor([[4.0, 2.0, -1.0], [3.0, 1.0, -2.0]])
        standardized = torch.tensor([[1.0, 0.0, -1.0], [1.5, 0.0, -1.5]])
        candidate_mask = torch.tensor([[True, True, True], [True, True, False]])

        output = adapter(
            rank_token_layers=rank_token,
            query_layers=query,
            text_features=text,
            phrase_token_mask=phrase_mask,
            score_token_mask=score_mask,
            patch_logits=patch,
            patch_standardized=standardized,
            candidate_mask=candidate_mask,
        )
        self.assertEqual(output["token_layers"].dtype, torch.float32)
        self.assertTrue(torch.equal(output["token_layers"], rank_token.float()))
        self.assertTrue(
            torch.equal(
                output["token_residual_layers"],
                torch.zeros_like(output["token_residual_layers"]),
            )
        )

        modifier = torch.stack(
            [
                DenseExpressionTower.aggregate_phrase_logits(layer, score_mask)
                for layer in rank_token.float()
            ]
        )
        expected_base = patch.clamp(-5.0, 5.0)[None] + torch.where(
            score_mask.any(dim=-1)[None, :, None],
            modifier,
            torch.zeros_like(modifier),
        )
        self.assertTrue(torch.equal(output["base_layers"], expected_base))
        self.assertTrue(
            torch.equal(
                output["base_layers"][:, 1],
                patch[1].clamp(-5.0, 5.0)[None].expand(2, -1),
            )
        )

        changed_features = adapter(
            rank_token_layers=rank_token,
            query_layers=query + 100.0,
            text_features=text - 100.0,
            phrase_token_mask=phrase_mask,
            score_token_mask=score_mask,
            patch_logits=patch,
            patch_standardized=standardized,
            candidate_mask=candidate_mask,
        )["hidden_layers"]
        self.assertTrue(torch.equal(output["hidden_layers"], changed_features))

        pool = AbsoluteConfidencePool(4, pool_hidden_dim=7, score_topk=2)
        global_logit, pool_residual = pool(
            output["hidden_layers"][0], output["base_layers"][0], candidate_mask
        )
        expected_global, _ = _masked_score_statistics(
            output["base_layers"][0], candidate_mask, topk=2
        )
        self.assertTrue(torch.equal(pool_residual, torch.zeros_like(pool_residual)))
        self.assertTrue(torch.equal(global_logit, expected_global))

    def test_word_softmin_is_invariant_to_wordpiece_count(self):
        one_piece_logits = torch.tensor([[[[3.0, -2.0]]]])
        one_piece_residuals = torch.tensor([[[[0.8, -0.2]]]], requires_grad=True)
        one_piece_probability, one_piece_gate = (
            _word_normalized_softmin_probability(
                one_piece_logits,
                one_piece_residuals,
                torch.tensor([[True, True]]),
                torch.tensor([[0, 1]], dtype=torch.long),
                temperature=0.1,
                gate_scale=1.0,
            )
        )

        three_piece_logits = torch.tensor([[[[3.0, 3.0, 3.0, -2.0]]]])
        three_piece_residuals = torch.tensor(
            [[[[0.8, 0.8, 0.8, -0.2]]]], requires_grad=True
        )
        three_piece_probability, three_piece_gate = (
            _word_normalized_softmin_probability(
                three_piece_logits,
                three_piece_residuals,
                torch.tensor([[True, True, True, True]]),
                torch.tensor([[0, 0, 0, 1]], dtype=torch.long),
                temperature=0.1,
                gate_scale=1.0,
            )
        )
        torch.testing.assert_close(
            one_piece_probability, three_piece_probability, rtol=0.0, atol=1e-7
        )
        torch.testing.assert_close(one_piece_gate, three_piece_gate, rtol=0.0, atol=0.0)
        self.assertFalse(one_piece_gate.requires_grad)
        self.assertFalse(three_piece_gate.requires_grad)

    def test_word_veto_gate_offset_and_ramp_are_exact_and_detached(self):
        token_logits = torch.zeros(1, 1, 3, 1)
        token_residuals = torch.tensor([[[[0.04], [0.10], [0.20]]]])
        _probability, gate = _word_normalized_softmin_probability(
            token_logits,
            token_residuals,
            torch.tensor([[True]]),
            torch.tensor([[0]], dtype=torch.long),
            temperature=0.1,
            gate_scale=0.10,
            gate_offset=0.05,
        )
        torch.testing.assert_close(
            gate,
            torch.tensor([[[0.0, 0.5, 1.0]]]),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertFalse(gate.requires_grad)

    def test_word_veto_beats_high_patch_and_focuses_low_word_gradient(self):
        token_logits = torch.tensor(
            [[[[4.0, 4.0, 4.0, -4.0]]]], requires_grad=True
        )
        token_residuals = torch.tensor([[[[-0.5, -0.5, -0.5, 1.0]]]])
        text_probability, mismatch_gate = _word_normalized_softmin_probability(
            token_logits,
            token_residuals,
            torch.ones((1, 4), dtype=torch.bool),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            temperature=0.1,
            gate_scale=1.0,
        )
        self.assertEqual(float(mismatch_gate), 1.0)
        patch_probability = torch.tensor(5.0).sigmoid()
        joint_logit = torch.logit(
            (patch_probability * text_probability).clamp(1e-6, 1.0 - 1e-6)
        )
        self.assertLess(float(joint_logit.detach()), 0.0)
        torch.nn.functional.softplus(joint_logit).sum().backward()
        shared_gradient = token_logits.grad[0, 0, 0, :3].abs().max()
        edit_gradient = token_logits.grad[0, 0, 0, 3].abs()
        self.assertGreater(float(edit_gradient), 10.0 * float(shared_gradient))

    def test_word_veto_u0_preserves_legacy_confidence_and_has_zero_net_delta(self):
        legacy, _source = self._scorer(phase="eval", source_seed=31)
        word_veto, _source = self._scorer(
            phase="eval",
            source_seed=31,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO
            ),
        )
        legacy.eval()
        word_veto.eval()
        legacy_output, _inputs, _provider = self._forward(legacy)
        veto_output, _inputs, _provider = self._forward(word_veto)

        for key in (
            "final_rank_token_logits",
            "final_phrase_logits",
            "final_rank_score",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
        ):
            self.assertTrue(torch.equal(legacy_output[key], veto_output[key]), key)
        self.assertTrue(
            torch.equal(
                veto_output["final_confidence_delta_logits"],
                torch.zeros_like(veto_output["final_confidence_delta_logits"]),
            )
        )
        self.assertTrue(
            torch.equal(
                veto_output["final_confidence_global_logits"],
                veto_output["final_reference_global_confidence_logits"],
            )
        )

    def test_word_veto_penalty_preserves_inherited_evidence_and_only_subtracts(self):
        adapter = TokenAwareConfidenceAdapter(
            4,
            adapter_dim=3,
            max_text_len=2,
            patch_hidden_dim=3,
            score_topk=1,
            phrase_aggregation=CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_PENALTY,
        )
        with torch.no_grad():
            adapter.token_bias.bias.fill_(1.0)
        rank_token = torch.tensor([[[[2.0, -4.0]]]])
        token_mask = torch.ones((1, 2), dtype=torch.bool)
        output = adapter(
            rank_token_layers=rank_token,
            query_layers=torch.zeros(1, 1, 1, 4),
            text_features=torch.zeros(1, 2, 4),
            phrase_token_mask=token_mask,
            score_token_mask=token_mask,
            patch_logits=torch.tensor([[5.0]]),
            patch_standardized=torch.zeros(1, 1),
            candidate_mask=torch.ones(1, 1, dtype=torch.bool),
            score_word_group_ids=torch.tensor([[0, 1]], dtype=torch.long),
        )
        modifier = DenseExpressionTower.aggregate_phrase_logits(
            output["token_layers"][0], token_mask
        )
        identity = torch.tensor([[5.0]]) + modifier
        veto_probability, gate = _word_normalized_softmin_probability(
            output["token_layers"],
            output["token_residual_layers"],
            token_mask,
            torch.tensor([[0, 1]], dtype=torch.long),
            temperature=0.1,
            gate_scale=1.0,
        )
        expected = identity[None] + gate * veto_probability.clamp_min(1e-6).log()
        torch.testing.assert_close(output["base_layers"], expected)
        self.assertTrue(bool((output["base_layers"] <= identity[None]).all().item()))

    def test_word_veto_penalty_u0_preserves_legacy_confidence(self):
        legacy, _source = self._scorer(phase="eval", source_seed=37)
        penalty, _source = self._scorer(
            phase="eval",
            source_seed=37,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_PENALTY
            ),
        )
        legacy.eval()
        penalty.eval()
        legacy_output, _inputs, _provider = self._forward(legacy)
        penalty_output, _inputs, _provider = self._forward(penalty)
        for key in (
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_delta_logits",
        ):
            self.assertTrue(torch.equal(legacy_output[key], penalty_output[key]), key)

    def test_word_veto_absolute_cap_uses_identity_base_and_u0_is_exact(self):
        adapter = TokenAwareConfidenceAdapter(
            4,
            adapter_dim=3,
            max_text_len=2,
            patch_hidden_dim=3,
            score_topk=1,
            phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP
            ),
            veto_gate_offset=0.05,
            veto_gate_scale=0.10,
        )
        with torch.no_grad():
            adapter.token_bias.bias.fill_(0.2)
        token_mask = torch.ones((1, 2), dtype=torch.bool)
        adapter_output = adapter(
            rank_token_layers=torch.tensor([[[[2.0, -4.0], [1.0, 3.0]]]]),
            query_layers=torch.zeros(1, 1, 2, 4),
            text_features=torch.zeros(1, 2, 4),
            phrase_token_mask=token_mask,
            score_token_mask=token_mask,
            patch_logits=torch.tensor([[5.0, 4.0]]),
            patch_standardized=torch.zeros(1, 2),
            candidate_mask=torch.ones(1, 2, dtype=torch.bool),
            score_word_group_ids=torch.tensor([[0, 1]], dtype=torch.long),
        )
        modifier = DenseExpressionTower.aggregate_phrase_logits(
            adapter_output["token_layers"][0], token_mask
        )
        identity_base = torch.tensor([[5.0, 4.0]]) + modifier
        self.assertTrue(
            torch.equal(adapter_output["base_layers"], identity_base[None])
        )
        self.assertTrue(
            torch.equal(
                adapter_output["mismatch_gate_layers"],
                torch.ones_like(adapter_output["mismatch_gate_layers"]),
            )
        )

        legacy, _source = self._scorer(phase="eval", source_seed=43)
        absolute_cap, _source = self._scorer(
            phase="eval",
            source_seed=43,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.05,
            confidence_veto_gate_scale=0.10,
        )
        legacy.eval()
        absolute_cap.eval()
        legacy_output, _inputs, _provider = self._forward(legacy)
        cap_output, _inputs, _provider = self._forward(absolute_cap)
        for key in (
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_delta_logits",
        ):
            self.assertTrue(torch.equal(legacy_output[key], cap_output[key]), key)
        self.assertTrue(
            torch.equal(
                cap_output["final_confidence_veto_coverage"],
                torch.zeros_like(cap_output["final_confidence_veto_coverage"]),
            )
        )
        self.assertTrue(
            torch.equal(
                cap_output["final_confidence_veto_sample_gate"],
                torch.zeros_like(cap_output["final_confidence_veto_sample_gate"]),
            )
        )
        torch.testing.assert_close(
            cap_output["confidence_veto_absolute_ceiling"],
            torch.tensor(-0.1),
            rtol=0.0,
            atol=1e-7,
        )

    def test_word_veto_absolute_cap_zero_gate_keeps_post_pool_score_exact(self):
        scorer, _source = self._scorer(
            phase="eval",
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.05,
            confidence_veto_gate_scale=0.10,
        )
        scorer.eval()
        with torch.no_grad():
            scorer.confidence_pool.residual[-1].bias.fill_(7.0)
        output, _inputs, _provider = self._forward(scorer)
        valid = output["expression_valid_mask"]
        expected = output["final_reference_global_confidence_logits"] + 7.0
        self.assertTrue(
            torch.equal(
                output["final_confidence_global_logits"][valid], expected[valid]
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_sample_gate"],
                torch.zeros_like(output["final_confidence_veto_sample_gate"]),
            )
        )
        self.assertGreater(
            float(output["final_confidence_global_logits"][valid].min()),
            float(output["confidence_veto_absolute_ceiling"]),
        )

    def test_word_veto_absolute_cap_is_post_pool_and_gradient_is_isolated(self):
        scorer, _source = self._scorer(
            phase="confidence",
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.05,
            confidence_veto_gate_scale=0.10,
        )
        scorer.train()
        with torch.no_grad():
            scorer.confidence_adapter.token_bias.bias.fill_(0.2)
            scorer.confidence_pool.residual[-1].bias.fill_(10.0)
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        modifier_valid = (
            output["modifier_valid_mask"] & output["expression_valid_mask"]
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_coverage"][modifier_valid],
                torch.ones_like(
                    output["final_confidence_veto_coverage"][modifier_valid]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_sample_gate"][modifier_valid],
                torch.ones_like(
                    output["final_confidence_veto_sample_gate"][modifier_valid]
                ),
            )
        )
        expected_ceiling = output["final_confidence_veto_absolute_ceiling"][
            modifier_valid
        ]
        torch.testing.assert_close(
            output["final_confidence_global_logits"][modifier_valid],
            expected_ceiling,
            rtol=0.0,
            atol=2e-6,
        )
        uncapped = (
            output["final_reference_global_confidence_logits"][modifier_valid]
            + 10.0
        )
        self.assertTrue(bool((uncapped > 5.0).all().item()))

        output["final_confidence_global_logits"][modifier_valid].sum().backward()
        raw_ceiling = scorer.confidence_adapter.veto_cap_raw_ceiling
        self.assertIsNotNone(raw_ceiling)
        self.assertIsNotNone(raw_ceiling.grad)
        self.assertTrue(torch.isfinite(raw_ceiling.grad).all())
        self.assertTrue(bool(raw_ceiling.grad.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_word_veto_gated_pool_uses_frozen_reference_carrier(self):
        reference = torch.tensor([[5.0, 4.0, 100.0], [1.0, 1.0, 0.0]])
        eligible = torch.tensor([[True, True, False], [True, True, True]])
        mismatch_gate = torch.tensor(
            [[0.0, 1.0, 1.0], [0.7, 0.2, 1.0]], requires_grad=True
        )
        gate, carrier = _frozen_reference_carrier_gate(
            reference, eligible, mismatch_gate
        )
        self.assertTrue(torch.equal(carrier, torch.tensor([0, 0])))
        torch.testing.assert_close(gate, torch.tensor([0.0, 0.7]))
        self.assertFalse(gate.requires_grad)
        self.assertTrue(
            torch.equal(
                _frozen_reference_carrier_index(reference, eligible),
                carrier,
            )
        )

    def test_word_veto_gated_pool_closed_gate_bypasses_trained_pool_exactly(self):
        scorer, _source = self._scorer(
            phase="eval",
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.05,
            confidence_veto_gate_scale=0.10,
        )
        scorer.eval()
        with torch.no_grad():
            scorer.confidence_pool.residual[-1].bias.fill_(7.0)
        output, _inputs, _provider = self._forward(scorer)
        valid = output["expression_valid_mask"]
        expected_carrier = output["final_reference_base_logits"].masked_fill(
            ~output["candidate_eligible_mask"], -torch.inf
        ).argmax(dim=1)
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_carrier_index"][valid],
                expected_carrier[valid],
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_sample_gate"],
                torch.zeros_like(output["final_confidence_veto_sample_gate"]),
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_global_logits"][valid],
                output["final_reference_global_confidence_logits"][valid],
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_delta_logits"][valid],
                torch.zeros_like(output["final_confidence_delta_logits"][valid]),
            )
        )

    def test_signed_rank_query_pool_trains_through_closed_veto_gate_only(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=47,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.05,
            confidence_veto_gate_scale=0.10,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY
            ),
            confidence_residual_parameterization_gain=gain,
        )
        scorer.train()
        with torch.no_grad():
            scorer.confidence_pool.residual[-1].weight.fill_(0.125)
            scorer.confidence_pool.residual[-1].bias.fill_(0.25)

        output, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = output["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_sample_gate"][valid],
                torch.zeros_like(
                    output["final_confidence_veto_sample_gate"][valid]
                ),
            )
        )
        self.assertTrue(
            bool(
                (
                    output["final_confidence_global_logits"][valid]
                    != output["final_reference_global_confidence_logits"][valid]
                ).any().item()
            )
        )

        output["final_confidence_global_logits"][valid].sum().backward()
        trunk_gradients = [
            parameter.grad
            for parameter in scorer.confidence_adapter.global_query_trunk.parameters()
            if parameter.grad is not None
        ]
        pool_gradients = [
            parameter.grad
            for parameter in scorer.confidence_pool.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(trunk_gradients)
        self.assertTrue(
            any(bool(gradient.ne(0.0).any().item()) for gradient in trunk_gradients)
        )
        self.assertTrue(pool_gradients)
        self.assertTrue(
            any(bool(gradient.ne(0.0).any().item()) for gradient in pool_gradients)
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_continuous_modifier_gate_applies_sample_delta_without_cap_plateau(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=53,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS
            ),
        )
        scorer.train()
        with torch.no_grad():
            scorer.confidence_pool.residual[-1].bias.fill_(-2.0)

        output, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = output["expression_valid_mask"] & output["modifier_valid_mask"]
        gate = output["final_confidence_veto_sample_gate"][valid]
        expected_gate = torch.sigmoid(gate.new_tensor(-0.02 / 0.03))
        torch.testing.assert_close(
            gate,
            expected_gate.expand_as(gate),
            rtol=1e-5,
            atol=1e-6,
        )
        expected_delta = -2.0 * gate
        torch.testing.assert_close(
            output["final_confidence_delta_logits"][valid],
            expected_delta,
            rtol=1e-5,
            atol=1e-6,
        )
        # The readout translates each reference logit instead of smooth-minning
        # high scores onto one absolute ceiling.
        torch.testing.assert_close(
            output["final_confidence_global_logits"][valid]
            - output["final_confidence_global_logits"][valid].mean(),
            output["final_reference_global_confidence_logits"][valid]
            - output["final_reference_global_confidence_logits"][valid].mean(),
            rtol=1e-5,
            atol=1e-6,
        )

        output["final_confidence_global_logits"][valid].sum().backward()
        self.assertIsNotNone(scorer.confidence_adapter.token_bias.bias.grad)
        self.assertTrue(
            bool(scorer.confidence_adapter.token_bias.bias.grad.ne(0).any().item())
        )
        self.assertIsNotNone(
            scorer.confidence_adapter.veto_cap_raw_ceiling.grad
        )
        self.assertTrue(
            bool(
                scorer.confidence_adapter.veto_cap_raw_ceiling.grad.ne(0).any().item()
            )
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_monotone_veto_depth_is_u0_exact_trainable_and_never_raises(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=59,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_SIGNED_RANK_QUERY
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH
            ),
        )
        scorer.train()
        u0, _inputs, _provider = self._forward(scorer)
        valid = u0["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                u0["final_reference_global_confidence_logits"][valid],
            )
        )

        loss = u0["final_confidence_global_logits"][valid].sum()
        loss.backward()
        final_bias = scorer.confidence_pool.residual[-1].bias
        self.assertIsNotNone(final_bias.grad)
        self.assertTrue(bool(final_bias.grad.ne(0).any().item()))

        with torch.no_grad():
            scorer.confidence_pool.residual[-1].bias.fill_(2.0)
        lowered, _inputs, _provider = self._forward(scorer)
        self.assertTrue(
            bool(
                (
                    lowered["final_confidence_global_logits"][valid]
                    <= lowered["final_reference_global_confidence_logits"][valid]
                ).all().item()
            )
        )
        with torch.no_grad():
            scorer.confidence_pool.residual[-1].bias.fill_(-2.0)
        closed, _inputs, _provider = self._forward(scorer)
        self.assertTrue(
            torch.equal(
                closed["final_confidence_global_logits"][valid],
                closed["final_reference_global_confidence_logits"][valid],
            )
        )

    def test_token_conditioned_monotone_pool_is_u0_exact_and_text_aware(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=61,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CONTINUOUS_MONOTONE_DEPTH
            ),
        )
        scorer.train()
        u0, _inputs, _provider = self._forward(scorer)
        valid = u0["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                u0["final_reference_global_confidence_logits"][valid],
            )
        )

        adapter = scorer.confidence_adapter
        query = torch.randn(1, 1, 2, adapter.hidden_dim)
        text = torch.randn(1, adapter.max_text_len, adapter.hidden_dim)
        rank_token = torch.zeros(1, 1, 2, adapter.max_text_len)
        rank_token[0, 0, 0, 1] = -4.0
        rank_token[0, 0, 1, 0] = -4.0
        residual = torch.zeros_like(rank_token)
        modifier_mask = torch.zeros(
            1, adapter.max_text_len, dtype=torch.bool
        )
        modifier_mask[0, :2] = True
        feature = adapter._token_conditioned_global_feature(
            query=query,
            text=text,
            rank_token=rank_token,
            token_residual=residual,
            modifier_mask=modifier_mask,
        )
        self.assertEqual(tuple(feature.shape), tuple(query.shape))
        self.assertTrue(bool(torch.isfinite(feature).all().item()))
        self.assertFalse(torch.equal(feature[0, 0, 0], feature[0, 0, 1]))

    def test_complementary_trust_veto_is_u0_exact_and_direction_constrained(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=67,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_COMPLEMENTARY_TRUST_VETO
            ),
        )
        scorer.train()
        u0, _inputs, _provider = self._forward(scorer)
        valid = u0["expression_valid_mask"]
        reference = u0["final_reference_global_confidence_logits"][valid]
        self.assertTrue(
            torch.equal(u0["final_confidence_global_logits"][valid], reference)
        )
        u0["final_confidence_global_logits"][valid].sum().backward()
        final_bias = scorer.confidence_pool.residual[-1].bias
        self.assertIsNotNone(final_bias.grad)
        self.assertTrue(bool(final_bias.grad.ne(0).any().item()))

        with torch.no_grad():
            final_bias.fill_(2.0)
        vetoed, _inputs, _provider = self._forward(scorer)
        self.assertTrue(
            bool(
                (
                    vetoed["final_confidence_global_logits"][valid]
                    <= vetoed["final_reference_global_confidence_logits"][valid]
                ).all().item()
            )
        )
        with torch.no_grad():
            final_bias.fill_(-2.0)
        trusted, _inputs, _provider = self._forward(scorer)
        self.assertTrue(
            bool(
                (
                    trusted["final_confidence_global_logits"][valid]
                    >= trusted["final_reference_global_confidence_logits"][valid]
                ).all().item()
            )
        )

    def test_ungated_monotone_tail_veto_reaches_closed_gate_hard_negatives(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=71,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_UNGATED_MONOTONE_DEPTH
            ),
        )
        scorer.train()
        u0, _inputs, _provider = self._forward(scorer)
        valid = u0["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                u0["final_reference_global_confidence_logits"][valid],
            )
        )

        with torch.no_grad():
            scorer.confidence_adapter.token_bias.bias.fill_(-100.0)
            scorer.confidence_pool.residual[-1].bias.fill_(2.0)
        vetoed, _inputs, _provider = self._forward(scorer)
        closed_gate = (
            valid
            & (vetoed["final_confidence_veto_sample_gate"] < 1e-6)
        )
        self.assertTrue(bool(closed_gate.any().item()))
        self.assertTrue(
            bool(
                (
                    vetoed["final_confidence_global_logits"][closed_gate]
                    < vetoed["final_reference_global_confidence_logits"][
                        closed_gate
                    ]
                ).all().item()
            )
        )

    def test_floor_gated_monotone_veto_keeps_closed_gate_fallback(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=73,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_FLOOR_GATED_MONOTONE_DEPTH
            ),
        )
        scorer.train()
        u0, _inputs, _provider = self._forward(scorer)
        valid = u0["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                u0["final_reference_global_confidence_logits"][valid],
            )
        )

        with torch.no_grad():
            scorer.confidence_adapter.token_bias.bias.fill_(-100.0)
            scorer.confidence_pool.residual[-1].bias.fill_(2.0)
        vetoed, _inputs, _provider = self._forward(scorer)
        closed_gate = (
            valid
            & (vetoed["final_confidence_veto_sample_gate"] < 1e-6)
        )
        self.assertTrue(bool(closed_gate.any().item()))
        observed_drop = (
            vetoed["final_reference_global_confidence_logits"][closed_gate]
            - vetoed["final_confidence_global_logits"][closed_gate]
        )
        expected_drop = torch.full_like(observed_drop, 0.5)
        torch.testing.assert_close(
            observed_drop, expected_drop, rtol=0.0, atol=1e-5
        )

    def test_independent_absolute_confidence_does_not_use_rank_as_carrier(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=79,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_TOKEN_CONDITIONED
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_INDEPENDENT_ABSOLUTE_LOGIT
            ),
        )
        scorer.train()
        u0, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = u0["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                torch.zeros_like(u0["final_confidence_global_logits"][valid]),
            )
        )
        self.assertFalse(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                u0["final_reference_global_confidence_logits"][valid],
            )
        )
        u0["final_confidence_global_logits"][valid].sum().backward()
        final_bias = scorer.confidence_pool.residual[-1].bias
        self.assertIsNotNone(final_bias.grad)
        self.assertTrue(bool(final_bias.grad.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

        with torch.no_grad():
            final_bias.fill_(2.0)
        raised, _inputs, _provider = self._forward(scorer)
        torch.testing.assert_close(
            raised["final_confidence_global_logits"][valid],
            torch.full_like(
                raised["final_confidence_global_logits"][valid], 2.0
            ),
            rtol=0.0,
            atol=1e-6,
        )

    def test_cross_attention_absolute_confidence_preserves_token_identity(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=83,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_CROSS_ATTENTION
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CROSS_ATTENTION_INDEPENDENT_ABSOLUTE_LOGIT
            ),
        )
        scorer.train()
        u0, _inputs, _provider = self._forward(scorer)
        valid = u0["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                torch.zeros_like(u0["final_confidence_global_logits"][valid]),
            )
        )

        adapter = scorer.confidence_adapter
        query = torch.randn(1, 1, 2, adapter.hidden_dim)
        text = torch.randn(1, adapter.max_text_len, adapter.hidden_dim)
        rank_token = torch.zeros(1, 1, 2, adapter.max_text_len)
        rank_token[0, 0, 0, 1] = -4.0
        rank_token[0, 0, 1, 0] = -4.0
        residual = torch.zeros_like(rank_token)
        modifier_mask = torch.zeros(1, adapter.max_text_len, dtype=torch.bool)
        modifier_mask[0, :2] = True
        phrase_mask = modifier_mask.clone()
        feature = adapter._cross_attention_global_feature(
            query=query,
            text=text,
            rank_token=rank_token,
            token_residual=residual,
            modifier_mask=modifier_mask,
            phrase_mask=phrase_mask,
        )
        self.assertEqual(tuple(feature.shape), tuple(query.shape))
        self.assertTrue(bool(torch.isfinite(feature).all().item()))
        self.assertFalse(torch.equal(feature[0, 0, 0], feature[0, 0, 1]))

        scorer.zero_grad(set_to_none=True)
        with torch.no_grad():
            scorer.confidence_pool.residual[-1].weight.fill_(0.01)
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        output["final_confidence_global_logits"][valid].sum().backward()
        cross_gradient = scorer.confidence_adapter.cross_attention.in_proj_weight.grad
        self.assertIsNotNone(cross_gradient)
        self.assertTrue(bool(cross_gradient.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_candidate_absolute_logits_receive_local_supervision_before_pooling(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=89,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ABSOLUTE
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ABSOLUTE_LOGIT
            ),
        )
        scorer.train()
        u0, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = u0["expression_valid_mask"]
        candidate_mask = (
            u0["candidate_eligible_mask"] & valid[:, None, :]
        )
        candidate_logits = u0["final_confidence_base_logits"]
        self.assertTrue(
            torch.equal(
                candidate_logits[candidate_mask],
                torch.zeros_like(candidate_logits[candidate_mask]),
            )
        )
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                torch.zeros_like(u0["final_confidence_global_logits"][valid]),
            )
        )
        candidate_logits[candidate_mask].sum().backward()
        head = scorer.confidence_adapter.candidate_absolute_head
        self.assertIsNotNone(head)
        self.assertIsNotNone(head[-1].bias.grad)
        self.assertTrue(bool(head[-1].bias.grad.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

        scorer.zero_grad(set_to_none=True)
        with torch.no_grad():
            head[-1].weight.fill_(0.01)
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        output["final_confidence_base_logits"][candidate_mask].sum().backward()
        cross_gradient = scorer.confidence_adapter.cross_attention.in_proj_weight.grad
        self.assertIsNotNone(cross_gradient)
        self.assertTrue(bool(cross_gradient.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_candidate_calibration_is_zero_init_monotone_and_stop_gradient(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=97,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_CALIBRATED
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_CALIBRATED_LOGIT
            ),
        )
        scorer.train()
        u0, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = u0["expression_valid_mask"]
        candidate_mask = u0["candidate_eligible_mask"] & valid[:, None, :]
        self.assertTrue(
            torch.equal(
                u0["final_confidence_base_logits"][candidate_mask],
                torch.zeros_like(
                    u0["final_confidence_base_logits"][candidate_mask]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                u0["final_confidence_global_logits"][valid],
                torch.zeros_like(u0["final_confidence_global_logits"][valid]),
            )
        )
        adapter = scorer.confidence_adapter
        self.assertEqual(
            tuple(float(value.item()) for value in adapter.candidate_calibration_depths()),
            (0.0, 0.0, 0.0),
        )

        scorer.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.token_bias.bias.fill_(0.2)
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        output["final_confidence_global_logits"][valid].sum().backward()
        for parameter in (
            adapter.candidate_patch_scale_raw,
            adapter.candidate_veto_depth_raw,
            adapter.candidate_coverage_depth_raw,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(parameter.grad.ne(0).item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_candidate_normalized_patch_is_invariant_and_amplifies_veto(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=101,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_NORMALIZED
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_NORMALIZED_LOGIT
            ),
        )
        adapter = scorer.confidence_adapter
        self.assertIsNone(adapter.candidate_patch_scale_raw)
        patch = torch.tensor([[1.0, 2.0, -3.0]])
        standardized = torch.tensor([[-0.5, 1.0, -0.5]])
        mask = torch.tensor([[True, True, True]])
        first = adapter._patch_statistics(patch, standardized, mask)
        shifted = adapter._patch_statistics(patch + 17.0, standardized, mask)
        self.assertTrue(torch.equal(first, shifted))
        self.assertEqual(
            tuple(float(value.item()) for value in adapter.candidate_calibration_depths()),
            (0.0, 0.0, 0.0),
        )
        with torch.no_grad():
            adapter.candidate_veto_depth_raw.fill_(0.01)
            adapter.candidate_coverage_depth_raw.fill_(0.02)
        patch_depth, veto_depth, coverage_depth = (
            adapter.candidate_calibration_depths()
        )
        self.assertEqual(float(patch_depth.item()), 0.0)
        self.assertAlmostEqual(
            float(veto_depth.item()), 0.01 * adapter.adapter_dim, places=6
        )
        self.assertAlmostEqual(
            float(coverage_depth.item()), 0.02 * adapter.adapter_dim, places=6
        )

    def test_candidate_asymmetric_uses_raw_patch_and_separate_veto_units(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=101,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            ),
        )
        adapter = scorer.confidence_adapter
        self.assertIsNone(adapter.candidate_patch_scale_raw)
        patch = torch.tensor([[1.0, 2.0, -3.0]])
        standardized = torch.tensor([[-0.5, 1.0, -0.5]])
        mask = torch.tensor([[True, True, True]])
        first = adapter._patch_statistics(patch, standardized, mask)
        shifted = adapter._patch_statistics(patch + 17.0, standardized, mask)
        self.assertFalse(torch.equal(first, shifted))
        patch_depth, veto_depth, coverage_depth = (
            adapter.candidate_calibration_depths()
        )
        self.assertAlmostEqual(
            float(patch_depth.item()), 1.0 / adapter.patch_score_clip, places=7
        )
        self.assertEqual(float(veto_depth.item()), 0.0)
        self.assertEqual(float(coverage_depth.item()), 0.0)
        with torch.no_grad():
            adapter.candidate_veto_depth_raw.fill_(0.01)
            adapter.candidate_coverage_depth_raw.fill_(0.02)
        _, veto_depth, coverage_depth = adapter.candidate_calibration_depths()
        self.assertAlmostEqual(
            float(veto_depth.item()),
            0.01 * (adapter.adapter_dim ** 0.5),
            places=6,
        )
        self.assertAlmostEqual(
            float(coverage_depth.item()), 0.02 * adapter.adapter_dim, places=6
        )

    def _v53_fulltext_global_scorer(self):
        gain = 0.25 / 0.03
        scorer, source = self._scorer(
            phase="confidence",
            source_seed=131,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.0,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            ),
            confidence_head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_FULLTEXT_GLOBAL_ABSOLUTE
            ),
        )
        scorer.train()
        return scorer, source

    def test_v53_u0_inherits_frozen_fulltext_candidate_and_global_carriers(self):
        scorer, _source = self._v53_fulltext_global_scorer()
        output, _inputs, _provider = self._forward(scorer)
        eligible = output["candidate_eligible_mask"]
        valid = output["expression_valid_mask"]

        for batch_index in range(int(valid.shape[0])):
            for slot_index in range(int(valid.shape[1])):
                if not bool(valid[batch_index, slot_index]):
                    continue
                expected_candidate = DenseExpressionTower.aggregate_phrase_logits(
                    output["final_rank_token_logits"][
                        batch_index, :, slot_index
                    ][None],
                    output["expression_token_mask"][
                        batch_index, slot_index
                    ][None],
                )[0]
                candidate_mask = eligible[batch_index, :, slot_index]
                torch.testing.assert_close(
                    output["final_confidence_base_logits"][
                        batch_index, :, slot_index
                    ][candidate_mask],
                    expected_candidate[candidate_mask],
                    rtol=0.0,
                    atol=0.0,
                )
                expected_global = expected_candidate.masked_fill(
                    ~candidate_mask, -torch.inf
                ).max()
                torch.testing.assert_close(
                    output["final_confidence_global_logits"][
                        batch_index, slot_index
                    ],
                    expected_global,
                    rtol=0.0,
                    atol=0.0,
                )
                expected_carrier = expected_candidate.masked_fill(
                    ~candidate_mask, -torch.inf
                ).argmax()
                self.assertEqual(
                    int(
                        output["final_confidence_veto_carrier_index"][
                            batch_index, slot_index
                        ].item()
                    ),
                    int(expected_carrier.item()),
                )

    def test_v53_sample_global_loss_reaches_winner_and_complete_global_owner_only(self):
        scorer, _source = self._v53_fulltext_global_scorer()
        residual_outputs = []

        def retain_candidate_residual(_module, _inputs, value):
            value.retain_grad()
            residual_outputs.append(value)

        hook = scorer.confidence_adapter.candidate_absolute_head.register_forward_hook(
            retain_candidate_residual
        )
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        hook.remove()
        valid = output["expression_valid_mask"]
        output["final_confidence_global_logits"][valid].sum().backward()

        self.assertEqual(len(residual_outputs), 1)
        self.assertIsNotNone(residual_outputs[0].grad)
        self.assertTrue(bool(residual_outputs[0].grad.ne(0).any().item()))
        self.assertIsNotNone(
            scorer.confidence_adapter.candidate_absolute_head[-1].bias.grad
        )
        self.assertTrue(
            bool(
                scorer.confidence_adapter.candidate_absolute_head[-1]
                .bias.grad.ne(0)
                .any()
                .item()
            )
        )
        self.assertIsNotNone(scorer.confidence_pool.residual[-1].bias.grad)
        self.assertTrue(
            bool(scorer.confidence_pool.residual[-1].bias.grad.ne(0).any().item())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.token_veto_parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_v53_two_owner_partition_is_complete_and_has_no_dormant_tensors(self):
        scorer, _source = self._v53_fulltext_global_scorer()
        adapter = scorer.confidence_adapter
        self.assertIsNone(adapter.patch_residual)
        self.assertIsNone(adapter.global_query_norm)
        self.assertIsNone(adapter.veto_cap_raw_ceiling)
        self.assertIsNone(adapter.candidate_patch_scale_raw)
        self.assertIsNone(adapter.candidate_veto_depth_raw)
        self.assertIsNone(adapter.candidate_coverage_depth_raw)

        token = scorer.token_veto_parameters()
        global_absolute = scorer.global_absolute_parameters()
        token_ids = {id(parameter) for parameter in token}
        global_ids = {id(parameter) for parameter in global_absolute}
        confidence_ids = {id(parameter) for parameter in scorer.confidence_parameters()}
        self.assertEqual(len(token), 21)
        self.assertEqual(len(global_absolute), 44)
        self.assertFalse(token_ids & global_ids)
        self.assertEqual(token_ids | global_ids, confidence_ids)

        output, _inputs, _provider = self._forward(scorer)
        valid = output["expression_valid_mask"]
        loss = output["final_confidence_global_logits"][valid].sum()
        loss = loss + output["final_confidence_token_logits"][0, :, 0, 0].sum()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in token))
        self.assertTrue(
            all(parameter.grad is not None for parameter in global_absolute)
        )
        self.assertEqual(
            scorer.expected_live_confidence_parameter_tensor_counts(),
            {"token_veto": 21, "global_absolute": 44},
        )

    def test_candidate_deployed_routing_st_is_v13_forward_identical(self):
        gain = 0.25 / 0.03
        common = {
            "phase": "confidence",
            "source_seed": 109,
            "confidence_phrase_aggregation": (
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            "confidence_veto_gate_offset": 0.0,
            "confidence_veto_gate_scale": 0.03,
            "confidence_rank_evidence_contract": (
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            "confidence_pool_feature_contract": (
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
            ),
            "confidence_residual_parameterization_gain": gain,
        }
        v13, _source = self._scorer(
            **common,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_LOGIT
            ),
        )
        v43, _source = self._scorer(
            **common,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
            ),
        )
        v13.train()
        v43.train()
        self.assertEqual(tuple(v13.state_dict()), tuple(v43.state_dict()))
        for name, value in v13.state_dict().items():
            self.assertTrue(torch.equal(value, v43.state_dict()[name]), msg=name)

        compared_keys = (
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_mismatch_gate",
            "final_confidence_veto_coverage",
            "final_confidence_veto_sample_gate",
            "final_confidence_veto_carrier_index",
        )
        for residual_bias in (0.0, 0.02, 0.08):
            with self.subTest(residual_bias=residual_bias), torch.no_grad():
                v13.confidence_adapter.token_bias.bias.fill_(residual_bias)
                v43.confidence_adapter.token_bias.bias.fill_(residual_bias)
                v13_output, _inputs, _provider = self._forward(v13)
                v43_output, _inputs, _provider = self._forward(v43)
                for key in compared_keys:
                    self.assertTrue(
                        torch.equal(v13_output[key], v43_output[key]),
                        msg=f"forward mismatch for {key} at bias={residual_bias}",
                    )

    def test_candidate_deployed_routing_st_exposes_coverage_gradient_only_to_adapter(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=113,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.0,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
            ),
        )
        scorer.train()
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = output["expression_valid_mask"]
        coverage = output["final_confidence_veto_coverage"][valid]
        self.assertTrue(coverage.requires_grad)
        self.assertTrue(torch.equal(coverage, torch.zeros_like(coverage)))
        (1.0 - coverage).mean().backward()
        residual_bias = scorer.confidence_adapter.token_bias.bias
        self.assertIsNotNone(residual_bias.grad)
        self.assertTrue(bool(residual_bias.grad.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_split_confidence_heads_are_forward_identical_and_gradient_disjoint(self):
        gain = 0.25 / 0.03
        common = {
            "phase": "confidence",
            "source_seed": 127,
            "confidence_phrase_aggregation": (
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            "confidence_veto_gate_offset": 0.0,
            "confidence_veto_gate_scale": 0.03,
            "confidence_rank_evidence_contract": (
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            "confidence_pool_feature_contract": (
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
            ),
            "confidence_residual_parameterization_gain": gain,
            "confidence_gate_gradient_contract": (
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
            ),
        }
        shared, _source = self._scorer(
            **common,
            confidence_head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_SHARED
            ),
        )
        split, _source = self._scorer(
            **common,
            confidence_head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_JOINT_CLIP
            ),
        )
        shared.train()
        split.train()
        split.load_state_dict(shared.state_dict(), strict=True)
        with torch.no_grad():
            for scorer in (shared, split):
                scorer.confidence_adapter.token_bias.bias.fill_(0.08)
                scorer.confidence_adapter.candidate_absolute_head[-1].weight.fill_(
                    0.01
                )
                scorer.confidence_pool.residual[-1].weight.fill_(0.01)

        shared_output, _inputs, _provider = self._forward(shared)
        split_output, inputs, provider = self._forward(split, requires_grad=True)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_mismatch_gate",
            "final_confidence_veto_coverage",
        ):
            self.assertTrue(
                torch.equal(shared_output[key], split_output[key]),
                msg=f"split head changed forward value for {key}",
            )

        token_parameters = split.token_veto_parameters()
        global_parameters = split.global_absolute_parameters()
        token_ids = {id(parameter) for parameter in token_parameters}
        global_ids = {id(parameter) for parameter in global_parameters}
        confidence_ids = {
            id(parameter) for parameter in split.confidence_parameters()
        }
        self.assertFalse(token_ids & global_ids)
        self.assertEqual(token_ids | global_ids, confidence_ids)

        valid = split_output["expression_valid_mask"]
        split_output["final_confidence_global_logits"][valid].sum().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or not bool(parameter.grad.detach().ne(0).any().item())
                for parameter in token_parameters
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in global_parameters
            )
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

        split.zero_grad(set_to_none=True)
        token_output, _inputs, _provider = self._forward(split, requires_grad=True)
        token_valid = token_output["expression_valid_mask"]
        token_loss = token_output["final_confidence_token_residual_logits"][
            token_valid[:, None, :, None].expand_as(
                token_output["final_confidence_token_residual_logits"]
            )
        ].sum()
        token_loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in token_parameters
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in global_parameters))

    def test_independent_deployed_router_preserves_v47_u0_and_owns_only_routing(self):
        gain = 0.25 / 0.03
        common = {
            "phase": "eval",
            "source_seed": 129,
            "confidence_phrase_aggregation": (
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            "confidence_veto_gate_offset": 0.0,
            "confidence_veto_gate_scale": 0.03,
            "confidence_rank_evidence_contract": (
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            "confidence_pool_feature_contract": (
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
            ),
            "confidence_residual_parameterization_gain": gain,
            "confidence_gate_gradient_contract": (
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
            ),
        }
        baseline, _source = self._scorer(
            **common,
            confidence_head_gradient_contract=(
                "split_token_veto_global_absolute_v2"
            ),
        )
        routed, _source = self._scorer(
            **common,
            confidence_head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_DEPLOYED_ROUTER
            ),
        )
        baseline.eval()
        routed.eval()

        baseline_state = baseline.state_dict()
        routed_state = routed.state_dict()
        self.assertTrue(
            all(
                torch.equal(value, routed_state[name])
                for name, value in baseline_state.items()
            )
        )
        router_parameters = routed.deployed_router_parameters()
        token_parameters = routed.token_veto_parameters()
        global_parameters = routed.global_absolute_parameters()
        owner_ids = [
            {id(parameter) for parameter in parameters}
            for parameters in (
                token_parameters,
                router_parameters,
                global_parameters,
            )
        ]
        self.assertTrue(all(owner_ids))
        self.assertFalse(owner_ids[0] & owner_ids[1])
        self.assertFalse(owner_ids[0] & owner_ids[2])
        self.assertFalse(owner_ids[1] & owner_ids[2])
        self.assertEqual(
            set().union(*owner_ids),
            {id(parameter) for parameter in routed.confidence_parameters()},
        )

        baseline_output, _inputs, _provider = self._forward(baseline)
        routed_output, _inputs, _provider = self._forward(routed)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_mismatch_gate",
            "final_confidence_veto_coverage",
        ):
            self.assertTrue(
                torch.equal(baseline_output[key], routed_output[key]),
                msg=f"independent router changed V47 U0 output {key}",
            )
        self.assertTrue(
            torch.equal(
                routed_output["final_confidence_raw_mismatch_gate"],
                routed_output["final_confidence_deployed_routing_gate"],
            )
        )
        self.assertTrue(
            torch.equal(
                routed_output["final_confidence_deployed_routing_residual"],
                torch.zeros_like(
                    routed_output[
                        "final_confidence_deployed_routing_residual"
                    ]
                ),
            )
        )

        routed.set_phase("confidence")
        routed.train()
        routing_output, inputs, provider = self._forward(
            routed, requires_grad=True
        )
        valid = routing_output["expression_valid_mask"]
        routing_gate = routing_output[
            "final_confidence_deployed_routing_gate"
        ]
        routing_gate[
            valid[:, None, :].expand_as(routing_gate)
        ].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in router_parameters
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in token_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in global_parameters))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

        from engine import _clip_stage_b_dense_duty_grad_norms

        wrapper = nn.Module()
        wrapper.stage_b_fixed_text_scorer = routed
        routing_clip = _clip_stage_b_dense_duty_grad_norms(wrapper, 0.05)
        self.assertGreater(
            routing_clip["grad_norm_dense_duty_deployed_router_preclip"], 0.0
        )
        self.assertLessEqual(
            routing_clip["grad_norm_dense_duty_deployed_router_postclip"],
            0.050001,
        )
        self.assertGreater(
            routing_clip["grad_tensor_count_dense_duty_deployed_router"], 0.0
        )

        routed.zero_grad(set_to_none=True)
        token_output, _inputs, _provider = self._forward(routed)
        token_valid = token_output["expression_valid_mask"]
        token_residual = token_output[
            "final_confidence_token_residual_logits"
        ]
        token_residual[
            token_valid[:, None, :, None].expand_as(token_residual)
        ].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in token_parameters
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in router_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in global_parameters))

        routed.zero_grad(set_to_none=True)
        global_output, _inputs, _provider = self._forward(routed)
        global_valid = global_output["expression_valid_mask"]
        global_output["final_confidence_global_logits"][
            global_valid
        ].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in global_parameters
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in token_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in router_parameters))

    def test_candidate_sample_split_preserves_forward_and_isolates_three_owners(self):
        gain = 0.25 / 0.03
        common = {
            "phase": "confidence",
            "source_seed": 137,
            "confidence_phrase_aggregation": (
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            "confidence_veto_gate_offset": 0.0,
            "confidence_veto_gate_scale": 0.03,
            "confidence_rank_evidence_contract": (
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            "confidence_pool_feature_contract": (
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
            ),
            "confidence_residual_parameterization_gain": gain,
            "confidence_gate_gradient_contract": (
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
            ),
        }
        baseline, _source = self._scorer(
            **common,
            confidence_head_gradient_contract="split_token_veto_global_absolute_v2",
        )
        split, _source = self._scorer(
            **common,
            confidence_head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_CANDIDATE_SAMPLE
            ),
        )
        baseline.train()
        split.train()
        split.load_state_dict(baseline.state_dict(), strict=True)

        baseline_output, _inputs, _provider = self._forward(baseline)
        split_output, inputs, provider = self._forward(split, requires_grad=True)
        for key in (
            "final_confidence_token_logits",
            "final_confidence_base_logits",
            "final_confidence_global_logits",
            "final_confidence_mismatch_gate",
            "final_confidence_veto_coverage",
        ):
            self.assertTrue(
                torch.equal(baseline_output[key], split_output[key]),
                msg=f"candidate/sample split changed U0 forward value for {key}",
            )

        token_parameters = split.token_veto_parameters()
        candidate_parameters = split.candidate_absolute_parameters()
        sample_parameters = split.sample_calibrator_parameters()
        owner_ids = tuple(
            {id(parameter) for parameter in owner}
            for owner in (token_parameters, candidate_parameters, sample_parameters)
        )
        self.assertTrue(all(owner_ids))
        self.assertFalse(owner_ids[0] & owner_ids[1])
        self.assertFalse(owner_ids[0] & owner_ids[2])
        self.assertFalse(owner_ids[1] & owner_ids[2])
        self.assertEqual(
            set().union(*owner_ids),
            {id(parameter) for parameter in split.confidence_parameters()},
        )

        valid = split_output["expression_valid_mask"]
        candidate_logits = split_output["final_confidence_base_logits"]
        candidate_mask = valid[:, None, :].expand_as(candidate_logits)
        candidate_logits[candidate_mask].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in candidate_parameters
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in token_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in sample_parameters))

        split.zero_grad(set_to_none=True)
        global_output, _inputs, _provider = self._forward(split, requires_grad=True)
        global_output["final_confidence_global_logits"][valid].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in sample_parameters
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in token_parameters))
        self.assertTrue(all(parameter.grad is None for parameter in candidate_parameters))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_global_trust_veto_routes_are_value_identical_and_gradient_isolated(self):
        gain = 0.25 / 0.03
        common = {
            "phase": "confidence",
            "source_seed": 131,
            "confidence_phrase_aggregation": (
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            "confidence_veto_gate_offset": 0.0,
            "confidence_veto_gate_scale": 0.03,
            "confidence_rank_evidence_contract": (
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            "confidence_pool_feature_contract": (
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_ASYMMETRIC
            ),
            "confidence_residual_parameterization_gain": gain,
            "confidence_gate_gradient_contract": (
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_ASYMMETRIC_DEPLOYED_ROUTING_ST
            ),
        }
        baseline, _source = self._scorer(
            **common,
            confidence_head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_SPLIT_JOINT_CLIP
            ),
        )
        split, _source = self._scorer(
            **common,
            confidence_head_gradient_contract=(
                CONFIDENCE_HEAD_GRADIENT_CONTRACT_GLOBAL_TRUST_VETO
            ),
        )
        baseline.train()
        split.train()
        baseline_output, _inputs, _provider = self._forward(baseline)
        split_output, inputs, provider = self._forward(split, requires_grad=True)
        valid = split_output["expression_valid_mask"]

        self.assertTrue(
            torch.equal(
                baseline_output["final_confidence_global_logits"],
                split_output["final_confidence_global_logits"],
            )
        )
        self.assertTrue(
            torch.equal(
                split_output["final_positive_confidence_logits"],
                split_output["final_validity_logits"],
            )
        )
        self.assertTrue(
            torch.equal(
                split_output["final_negative_confidence_logits"],
                split_output["final_validity_logits"],
            )
        )
        self.assertTrue(
            torch.equal(
                split_output["final_global_veto_raw_logits"][valid],
                torch.zeros_like(
                    split_output["final_global_veto_raw_logits"][valid]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                split_output["final_global_veto_depth"][valid],
                torch.zeros_like(split_output["final_global_veto_depth"][valid]),
            )
        )

        token_parameters = split.token_veto_parameters()
        trust_parameters = split.global_trust_parameters()
        veto_parameters = split.global_veto_parameters()
        token_ids = {id(parameter) for parameter in token_parameters}
        trust_ids = {id(parameter) for parameter in trust_parameters}
        veto_ids = {id(parameter) for parameter in veto_parameters}
        confidence_ids = {
            id(parameter) for parameter in split.confidence_parameters()
        }
        self.assertTrue(token_ids and trust_ids and veto_ids)
        self.assertFalse(token_ids & trust_ids)
        self.assertFalse(token_ids & veto_ids)
        self.assertFalse(trust_ids & veto_ids)
        self.assertEqual(token_ids | trust_ids | veto_ids, confidence_ids)

        positive_loss = torch.nn.functional.softplus(
            -split_output["final_positive_global_confidence_logits"][valid]
        ).mean()
        positive_loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in trust_parameters
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in veto_parameters
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                or not bool(parameter.grad.detach().ne(0).any().item())
                for parameter in token_parameters
            )
        )

        split.zero_grad(set_to_none=True)
        negative_output, _inputs, _provider = self._forward(
            split, requires_grad=True
        )
        negative_valid = negative_output["expression_valid_mask"]
        negative_loss = torch.nn.functional.softplus(
            negative_output["final_negative_global_confidence_logits"][
                negative_valid
            ]
        ).mean()
        negative_loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or not bool(parameter.grad.detach().ne(0).any().item())
                for parameter in trust_parameters
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                and bool(parameter.grad.detach().ne(0).any().item())
                for parameter in veto_parameters
            )
        )
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(all(value.grad is None for value in provider.leaves))

        optimizer = torch.optim.SGD(veto_parameters, lr=0.1)
        optimizer.step()
        learned_output, _inputs, _provider = self._forward(split)
        learned_valid = learned_output["expression_valid_mask"]
        learned_depth = learned_output["final_global_veto_depth"][learned_valid]
        self.assertTrue(bool(learned_depth.ge(0.0).all().item()))
        self.assertTrue(bool(learned_depth.gt(0.0).any().item()))

    def test_candidate_set_attention_pool_is_independent_and_trainable(self):
        gain = 0.25 / 0.03
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=101,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.02,
            confidence_veto_gate_scale=0.03,
            confidence_rank_evidence_contract=(
                CONFIDENCE_RANK_EVIDENCE_CONTRACT_SPARSE_RANK_CHANNEL_MISMATCH
            ),
            confidence_pool_feature_contract=(
                CONFIDENCE_POOL_FEATURE_CONTRACT_CANDIDATE_SET_ATTENTION
            ),
            confidence_residual_parameterization_gain=gain,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_CANDIDATE_SET_ATTENTION_LOGIT
            ),
        )
        pool = scorer.confidence_pool
        self.assertTrue(pool.set_attention)
        self.assertEqual(tuple(pool.set_seed.shape), (4, scorer.rank_tower.hidden_dim))
        with torch.no_grad():
            pool.residual[-1].weight.fill_(0.01)
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = output["expression_valid_mask"]
        output["final_confidence_global_logits"][valid].sum().backward()
        self.assertIsNotNone(pool.set_seed.grad)
        self.assertTrue(bool(torch.isfinite(pool.set_seed.grad).all().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_gated_pool_soft_backward_is_bitwise_forward_identical(self):
        common = {
            "phase": "eval",
            "source_seed": 23,
            "confidence_phrase_aggregation": (
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            "confidence_veto_gate_offset": 0.05,
            "confidence_veto_gate_scale": 0.10,
        }
        hard_scorer, _source = self._scorer(
            **common,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED
            ),
        )
        soft_backward_scorer, _source = self._scorer(
            **common,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD
            ),
        )
        hard_scorer.eval()
        soft_backward_scorer.eval()
        self.assertEqual(
            hard_scorer.confidence_adapter.gate_gradient_contract,
            CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_DETACHED,
        )
        self.assertEqual(
            soft_backward_scorer.confidence_adapter.gate_gradient_contract,
            CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD,
        )
        self.assertEqual(
            tuple(hard_scorer.state_dict()),
            tuple(soft_backward_scorer.state_dict()),
        )
        for name, hard_value in hard_scorer.state_dict().items():
            self.assertTrue(
                torch.equal(hard_value, soft_backward_scorer.state_dict()[name])
            )

        compared_keys = (
            "final_confidence_token_logits",
            "final_confidence_token_residual_logits",
            "final_reference_base_logits",
            "final_reference_global_confidence_logits",
            "final_confidence_global_logits",
            "final_confidence_delta_logits",
            "final_confidence_mismatch_gate",
            "final_confidence_veto_sample_gate",
            "final_confidence_veto_carrier_index",
        )
        for residual_bias in (0.0, 0.08, 0.2):
            with self.subTest(residual_bias=residual_bias), torch.no_grad():
                hard_scorer.confidence_adapter.token_bias.bias.fill_(residual_bias)
                soft_backward_scorer.confidence_adapter.token_bias.bias.fill_(
                    residual_bias
                )
                hard_output, _inputs, _provider = self._forward(hard_scorer)
                soft_backward_output, _inputs, _provider = self._forward(
                    soft_backward_scorer
                )
                for key in compared_keys:
                    self.assertTrue(
                        torch.equal(hard_output[key], soft_backward_output[key]),
                        msg=f"forward mismatch for {key} at bias={residual_bias}",
                    )

    def test_gated_pool_soft_backward_reaches_hard_closed_carrier_only(self):
        scorer, _source = self._scorer(
            phase="confidence",
            source_seed=29,
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.05,
            confidence_veto_gate_scale=0.10,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD
            ),
        )
        scorer.train()
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        valid = output["expression_valid_mask"]
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_sample_gate"][valid],
                torch.zeros_like(
                    output["final_confidence_veto_sample_gate"][valid]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_global_logits"][valid],
                output["final_reference_global_confidence_logits"][valid],
            )
        )
        reference_global = output["final_reference_global_confidence_logits"]
        high_score_index = reference_global.masked_fill(
            ~valid, -torch.inf
        ).reshape(-1).argmax()
        self.assertGreater(
            float(reference_global.reshape(-1)[high_score_index].item()), 1.0
        )

        output["final_confidence_global_logits"].reshape(-1)[
            high_score_index
        ].backward()
        residual_bias = scorer.confidence_adapter.token_bias.bias
        self.assertIsNotNone(residual_bias.grad)
        self.assertTrue(torch.isfinite(residual_bias.grad).all())
        self.assertTrue(bool(residual_bias.grad.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_word_veto_gated_pool_open_gate_trains_bounded_veto_depth(self):
        scorer, _source = self._scorer(
            phase="confidence",
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO_GATED_POOL_ABSOLUTE_CAP
            ),
            confidence_veto_gate_offset=0.05,
            confidence_veto_gate_scale=0.10,
            confidence_gate_gradient_contract=(
                CONFIDENCE_GATE_GRADIENT_CONTRACT_HARD_FORWARD_SOFT_BACKWARD
            ),
        )
        scorer.train()
        with torch.no_grad():
            scorer.confidence_adapter.token_bias.bias.fill_(0.2)
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        modifier_valid = (
            output["modifier_valid_mask"] & output["expression_valid_mask"]
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_veto_sample_gate"][modifier_valid],
                torch.ones_like(
                    output["final_confidence_veto_sample_gate"][modifier_valid]
                ),
            )
        )
        ceiling = output["final_confidence_veto_absolute_ceiling"][modifier_valid]
        reference = output["final_reference_global_confidence_logits"][
            modifier_valid
        ]
        temperature = scorer.confidence_adapter.veto_cap_temperature
        expected = reference - temperature * torch.nn.functional.softplus(
            (reference - ceiling) / temperature
        ) - temperature * torch.log(
            torch.tensor(2.0, device=ceiling.device)
        )
        torch.testing.assert_close(
            output["final_confidence_global_logits"][modifier_valid],
            expected,
            rtol=0.0,
            atol=2e-6,
        )
        self.assertTrue(
            bool(
                (
                    output["final_confidence_global_logits"][modifier_valid]
                    <= ceiling
                ).all().item()
            )
        )
        self.assertTrue(
            bool(
                (
                    output["final_confidence_global_logits"][modifier_valid]
                    <= reference
                ).all().item()
            )
        )

        output["final_confidence_global_logits"][modifier_valid].sum().backward()
        raw_ceiling = scorer.confidence_adapter.veto_cap_raw_ceiling
        self.assertIsNotNone(raw_ceiling)
        self.assertIsNotNone(raw_ceiling.grad)
        self.assertTrue(bool(raw_ceiling.grad.ne(0).any().item()))
        pool_bias = scorer.confidence_pool.residual[-1].bias
        self.assertIsNotNone(pool_bias.grad)
        self.assertTrue(torch.isfinite(pool_bias.grad).all())
        self.assertTrue(bool(pool_bias.grad.ne(0).any().item()))
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

    def test_word_veto_category_only_row_is_exact_patch_fallback(self):
        scorer, _source = self._scorer(
            phase="eval",
            confidence_phrase_aggregation=(
                CONFIDENCE_PHRASE_AGGREGATION_WORD_VETO
            ),
        )
        scorer.eval()
        output, _inputs, _provider = self._forward(scorer)
        self.assertTrue(
            torch.equal(
                output["final_confidence_base_logits"][1, :, 0],
                output["candidate_patch_logits"][1],
            )
        )

    def test_confidence_adapter_fp32_subtraction_preserves_small_residual(self):
        adapter = TokenAwareConfidenceAdapter(
            4,
            adapter_dim=3,
            max_text_len=2,
            patch_hidden_dim=3,
            score_topk=1,
        )
        with torch.no_grad():
            adapter.token_bias.bias.fill_(1e-3)
        rank_token = torch.full((1, 1, 1, 2), 12.0, dtype=torch.float16)
        output = adapter(
            rank_token_layers=rank_token,
            query_layers=torch.zeros(1, 1, 1, 4),
            text_features=torch.zeros(1, 2, 4),
            phrase_token_mask=torch.tensor([[True, False]]),
            score_token_mask=torch.tensor([[True, False]]),
            patch_logits=torch.zeros(1, 1),
            patch_standardized=torch.zeros(1, 1),
            candidate_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        self.assertEqual(output["token_layers"].dtype, torch.float32)
        self.assertAlmostEqual(
            float(output["token_residual_layers"][0, 0, 0, 0].detach()),
            1e-3,
            places=7,
        )
        self.assertLess(
            float(output["token_layers"][0, 0, 0, 0].detach()), 12.0
        )
        self.assertEqual(
            float(output["token_layers"][0, 0, 0, 1].detach()), 12.0
        )

    def test_confidence_phase_token_loss_updates_only_token_adapter(self):
        scorer, _source = self._scorer(phase="confidence")
        scorer.train()
        output, inputs, provider = self._forward(scorer, requires_grad=True)
        self.assertTrue(
            torch.equal(
                output["final_rank_token_logits"],
                output["final_confidence_token_logits"],
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_confidence_token_residual_logits"],
                torch.zeros_like(
                    output["final_confidence_token_residual_logits"]
                ),
            )
        )
        output["final_token_logits"][0, :, 0, 0].sum().backward()
        self.assertIsNone(inputs["candidate_hs"].grad)
        self.assertIsNone(inputs["candidate_boxes"].grad)
        self.assertIsNone(inputs["candidate_patch_logits"].grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.rank_parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in scorer.confidence_adapter.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and bool(parameter.grad.ne(0).any().item())
                for parameter in scorer.confidence_adapter.query_projection.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                or not bool(parameter.grad.ne(0).any().item())
                for parameter in scorer.confidence_adapter.text_projection.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None and bool(parameter.grad.ne(0).any().item())
                for parameter in scorer.confidence_adapter.token_bias.parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in scorer.confidence_pool.parameters())
        )
        self.assertTrue(all(value.grad is None for value in provider.leaves))

        rank_before = output["final_rank_token_logits"].detach().clone()
        with torch.no_grad():
            for parameter in scorer.confidence_adapter.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.01)
        updated, _inputs, _provider = self._forward(scorer)
        self.assertTrue(
            torch.equal(updated["final_rank_token_logits"], rank_before)
        )
        self.assertTrue(
            bool(
                updated["final_confidence_token_residual_logits"]
                .ne(0)
                .any()
                .item()
            )
        )
        self.assertFalse(
            torch.equal(
                updated["final_rank_token_logits"],
                updated["final_confidence_token_logits"],
            )
        )
        scorer.zero_grad(set_to_none=True)
        updated["final_token_logits"][0, :, 0, 0].sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool(parameter.grad.ne(0).any().item())
                for parameter in scorer.confidence_adapter.text_projection.parameters()
            )
        )

    def test_all_phases_return_legacy_aliases_and_mark_placeholders(self):
        legacy_keys = {
            "layer_token_logits",
            "final_token_logits",
            "layer_phrase_logits",
            "final_phrase_logits",
            "layer_validity_logits",
            "final_validity_logits",
            "layer_validity_gate_logits",
            "final_validity_gate_logits",
            "layer_confidence_base_logits",
            "final_confidence_base_logits",
            "layer_predicate_logits",
            "final_predicate_logits",
            "predicate_token_mask",
            "predicate_valid_mask",
            "score_token_mask",
            "expression_token_mask",
            "final_rank_score",
            "final_score",
        }
        for phase in ("rank", "confidence", "eval"):
            with self.subTest(phase=phase):
                scorer, _source = self._scorer(phase=phase)
                scorer.eval()
                output, _inputs, _provider = self._forward(scorer)
                self.assertFalse(legacy_keys - set(output))
                self.assertFalse(bool(output["rank_output_is_placeholder"].item()))
                self.assertFalse(
                    bool(output["confidence_output_is_placeholder"].item())
                )

    def test_training_materializes_only_the_owned_tower(self):
        expected = {
            "rank": (False, True),
            "confidence": (False, False),
        }
        for phase, placeholders in expected.items():
            with self.subTest(phase=phase):
                scorer, _source = self._scorer(phase=phase)
                scorer.train()
                output, _inputs, _provider = self._forward(scorer)
                self.assertEqual(
                    bool(output["rank_output_is_placeholder"].item()),
                    placeholders[0],
                )
                self.assertEqual(
                    bool(output["confidence_output_is_placeholder"].item()),
                    placeholders[1],
                )

    def test_expression_microbatch_is_bounded_and_numerically_equivalent(self):
        scorer, _source = self._scorer(phase="eval")
        scorer.eval()
        inputs = self._inputs()
        torch.manual_seed(211)
        full_provider = _ContextProvider()
        full = scorer(
            raw_context_provider=full_provider,
            expression_microbatch=4,
            **inputs,
        )
        torch.manual_seed(211)
        split_provider = _ContextProvider()
        split = scorer(
            raw_context_provider=split_provider,
            expression_microbatch=1,
            **inputs,
        )
        self.assertEqual(full_provider.call_sizes, [4])
        self.assertEqual(split_provider.call_sizes, [1] * 4)
        for key in (
            "final_phrase_logits",
            "final_validity_logits",
            "final_rank_token_logits",
            "final_confidence_token_logits",
            "layer_confidence_token_residual_logits",
            "final_confidence_token_residual_logits",
        ):
            self.assertTrue(torch.allclose(full[key], split[key], atol=1e-6))

    def test_patch_changes_category_confidence_but_not_token_logits(self):
        scorer, _source = self._scorer(phase="confidence")
        scorer.eval()
        first_inputs = self._inputs()
        second_inputs = self._inputs()
        second_inputs["candidate_patch_logits"] = torch.tensor(
            [[-10.0, 8.0, 0.5], [7.5, -2.0, 4.0]]
        )

        torch.manual_seed(101)
        first = scorer(raw_context_provider=_ContextProvider(), **first_inputs)
        torch.manual_seed(101)
        second = scorer(raw_context_provider=_ContextProvider(), **second_inputs)

        self.assertTrue(
            torch.equal(
                first["candidate_eligible_mask"],
                second["candidate_eligible_mask"],
            )
        )
        self.assertTrue(
            torch.equal(
                first["confidence_token_logits"],
                second["confidence_token_logits"],
            )
        )
        self.assertFalse(
            torch.equal(
                first["final_confidence_global_logits"],
                second["final_confidence_global_logits"],
            )
        )

    def test_phase_controls_module_modes_and_trainable_surface(self):
        scorer, _source = self._scorer(phase="rank")
        scorer.train()
        self.assertTrue(scorer.rank_tower.training)
        self.assertFalse(scorer.confidence_adapter.training)
        self.assertTrue(all(parameter.requires_grad for parameter in scorer.rank_parameters()))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in scorer.confidence_parameters()
            )
        )
        self.assertFalse(scorer.rank_tower.encoder.layers.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in scorer.rank_tower.encoder.layers.parameters()
            )
        )

        scorer.set_phase("confidence")
        self.assertFalse(scorer.rank_tower.training)
        self.assertTrue(scorer.confidence_adapter.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in scorer.rank_parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in scorer.confidence_parameters()
            )
        )

        scorer.set_phase("eval")
        self.assertFalse(scorer.rank_tower.training)
        self.assertFalse(scorer.confidence_adapter.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in scorer.parameters())
        )

    def test_load_from_groundingdino_copies_rank_tower_exactly(self):
        scorer, _source = self._scorer(source_seed=1)
        replacement = _FakeGroundingDINO(99)
        audit = scorer.load_from_groundingdino(replacement)
        self.assertTrue(audit["confidence_adapter_initialized_independently"])
        self.assertEqual(
            audit["confidence_token_contract"],
            "detached_rank_token_minus_zero_init_residual_v1",
        )
        self.assertEqual(
            audit["confidence_pool_feature_contract"],
            "patch_statistics_only_v1",
        )
        tower = scorer.rank_tower
        for key, value in replacement.feat_map.state_dict().items():
            self.assertTrue(torch.equal(tower.feat_map.state_dict()[key], value))
        for key, value in replacement.transformer.encoder.state_dict().items():
            self.assertTrue(torch.equal(tower.encoder.state_dict()[key], value))
        source_decoder = copy.deepcopy(replacement.transformer.decoder)
        source_decoder.bbox_embed = None
        source_decoder.class_embed = None
        for key, value in source_decoder.state_dict().items():
            self.assertTrue(torch.equal(tower.decoder.state_dict()[key], value))
        self.assertTrue(
            torch.equal(tower.level_embed, replacement.transformer.level_embed)
        )

    def test_checkpoint_warmstart_is_exact_and_rejects_incomplete_state(self):
        scorer, _source = self._scorer(source_seed=1)
        replacement = _FakeGroundingDINO(123)
        source_decoder = copy.deepcopy(replacement.transformer.decoder)
        source_decoder.bbox_embed = None
        source_decoder.class_embed = None
        state = {}
        state.update(
            {f"feat_map.{key}": value for key, value in replacement.feat_map.state_dict().items()}
        )
        state.update(
            {
                f"transformer.encoder.{key}": value
                for key, value in replacement.transformer.encoder.state_dict().items()
            }
        )
        state.update(
            {
                f"transformer.decoder.{key}": value
                for key, value in source_decoder.state_dict().items()
            }
        )
        state["transformer.level_embed"] = replacement.transformer.level_embed.detach()
        audit = scorer.load_from_full_text_checkpoint_state(
            state, checkpoint_label="unit OGC"
        )
        self.assertGreater(audit["loaded_tensor_count"], 0)
        self.assertEqual(audit["source_decoder_num_layers"], 2)
        self.assertEqual(audit["selected_source_layer_indices"], [0, 1])
        self.assertEqual(audit["loaded_num_layers"], 2)
        self.assertEqual(
            audit["loaded_components"], list(scorer.warmstart_components)
        )
        self.assertTrue(
            torch.equal(
                scorer.rank_tower.feat_map.weight, replacement.feat_map.weight
            )
        )
        self.assertTrue(
            torch.equal(
                scorer.rank_tower.level_embed, replacement.transformer.level_embed
            )
        )
        incomplete = dict(state)
        incomplete.pop("feat_map.weight")
        with self.assertRaisesRegex(ValueError, "incompatible feat_map"):
            scorer.load_from_full_text_checkpoint_state(
                incomplete, checkpoint_label="incomplete OGC"
            )
        state["transformer.decoder.bbox_embed.0.weight"] = torch.randn(4, 4)
        scorer.load_from_full_text_checkpoint_state(
            state, checkpoint_label="OGC with discarded bbox head"
        )


if __name__ == "__main__":
    unittest.main()
