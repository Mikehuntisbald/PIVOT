import runpy
import unittest
from pathlib import Path

import torch
from torch import nn

from models.GroundingDINO.stage_b_fixed_text_scorer import (
    FixedBoxFullTextScorer,
    build_stage_b_noncanonical_token_masks_from_ids,
    validate_stage_b_fixed_text_scorer_checkpoint,
)
from models.GroundingDINO.transformer import TransformerDecoder


class _FakeTextDecoderLayer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.use_text_cross_attention = True
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward_for_test(self, tgt, memory, memory_text):
        text_summary = memory_text.mean(dim=1).unsqueeze(0)
        image_summary = memory.mean(dim=0).unsqueeze(0)
        return tgt + self.scale * text_summary + 0.1 * image_summary


class _FakeTransformerDecoder(TransformerDecoder):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 4) -> None:
        nn.Module.__init__(self)
        self.layers = nn.ModuleList(
            [_FakeTextDecoderLayer(float(idx + 1)) for idx in range(num_layers)]
        )
        self.d_model = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.return_intermediate = True
        self.norm = nn.LayerNorm(hidden_dim)
        self.ref_point_head = nn.Linear(4, hidden_dim)
        self.bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(num_layers)])
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
            output = layer.forward_for_test(output, memory, memory_text)
            outputs.append(self.norm(output).transpose(0, 1))
        reference = refpoints_unsigmoid.sigmoid().transpose(0, 1)
        references = [reference]
        if self.bbox_embed is not None:
            references.append((reference + 0.01).clamp(max=1.0))
        return outputs, references


class FixedBoxFullTextScorerTest(unittest.TestCase):
    def _context_provider(self, calls, hidden_dim, max_text_len):
        def provider(captions, owner_indices):
            calls.append((list(captions), owner_indices.detach().cpu().tolist()))
            batch = len(captions)
            device = owner_indices.device
            text_len = 4
            encoded_text = torch.zeros((batch, text_len, hidden_dim), device=device)
            feature_pattern = torch.linspace(
                0.5, 1.5, hidden_dim, device=device
            )
            for idx, caption in enumerate(captions):
                value = float(sum(ord(ch) for ch in caption) % 17 + 1) / 17.0
                encoded_text[idx, 1:3] = value * feature_pattern
            text_mask = torch.ones((batch, text_len), dtype=torch.bool, device=device)
            phrase_mask = torch.zeros((batch, text_len), dtype=torch.bool, device=device)
            phrase_mask[:, 1:3] = True
            memory = owner_indices.to(torch.float32)[:, None, None].expand(
                batch, 3, hidden_dim
            )
            return {
                "memory": memory,
                "memory_key_padding_mask": torch.zeros(
                    (batch, 3), dtype=torch.bool, device=device
                ),
                "memory_pos": torch.zeros_like(memory),
                "level_start_index": torch.tensor([0], dtype=torch.long, device=device),
                "spatial_shapes": torch.tensor([[1, 3]], dtype=torch.long, device=device),
                "valid_ratios": torch.ones((batch, 1, 2), device=device),
                "text_dict": {
                    "encoded_text": encoded_text,
                    "text_token_mask": text_mask,
                },
                "phrase_token_mask": phrase_mask,
            }

        return provider

    def test_independent_slots_microbatch_and_fixed_box_boundary(self):
        source = _FakeTransformerDecoder(num_layers=4, hidden_dim=4)
        scorer = FixedBoxFullTextScorer(
            source,
            num_layers=3,
            max_text_len=8,
            expression_microbatch=2,
        )
        self.assertIsNone(scorer.decoder.bbox_embed)
        self.assertIsNone(scorer.decoder.class_embed)
        self.assertEqual(scorer.decoder.layers[0].scale.detach().item(), 2.0)

        candidate_hs = torch.randn((2, 2, 4), requires_grad=True)
        candidate_boxes = torch.tensor(
            [
                [[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]],
                [[0.6, 0.6, 0.3, 0.3], [0.75, 0.75, 0.1, 0.1]],
            ],
            requires_grad=True,
        )
        boxes_before = candidate_boxes.detach().clone()
        calls = []
        predicate_mask = torch.zeros((2, 2, 8), dtype=torch.bool)
        predicate_mask[0, 0, 1] = True
        predicate_mask[0, 1, 2] = True
        predicate_mask[1, 0, 1] = True
        outputs = scorer(
            candidate_hs=candidate_hs,
            candidate_boxes=candidate_boxes,
            expression_captions=[["red object", "blue object"], ["small object", ""]],
            expression_valid_mask=torch.tensor([[True, True], [True, False]]),
            expression_predicate_token_mask=predicate_mask,
            context_provider=self._context_provider(calls, hidden_dim=4, max_text_len=8),
        )

        self.assertEqual(tuple(outputs["layer_token_logits"].shape), (3, 2, 2, 2, 8))
        self.assertEqual(tuple(outputs["final_token_logits"].shape), (2, 2, 2, 8))
        self.assertEqual(tuple(outputs["layer_phrase_logits"].shape), (3, 2, 2, 2))
        self.assertEqual(tuple(outputs["layer_validity_logits"].shape), (3, 2, 2, 2))
        self.assertTrue(
            torch.equal(outputs["layer_validity_logits"], outputs["layer_phrase_logits"])
        )
        self.assertTrue(
            torch.equal(outputs["final_validity_logits"], outputs["final_phrase_logits"])
        )
        self.assertEqual(tuple(outputs["layer_predicate_logits"].shape), (3, 2, 2, 2))
        self.assertEqual(tuple(outputs["final_predicate_logits"].shape), (2, 2, 2))
        self.assertEqual(outputs["predicate_valid_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(tuple(outputs["final_score"].shape), (2, 2, 2))
        self.assertEqual(
            calls,
            [
                (["red object", "blue object"], [0, 0]),
                (["small object", "object ."], [1, 1]),
            ],
        )
        self.assertFalse(
            torch.equal(outputs["final_score"][0, :, 0], outputs["final_score"][0, :, 1])
        )
        self.assertEqual(outputs["final_score"][1, :, 1].tolist(), [0.0, 0.0])
        self.assertTrue(torch.equal(candidate_boxes.detach(), boxes_before))

        outputs["final_score"].sum().backward()
        self.assertIsNone(candidate_hs.grad)
        self.assertIsNone(candidate_boxes.grad)
        for layer in scorer.decoder.layers:
            self.assertIsNotNone(layer.scale.grad)
            self.assertTrue(torch.isfinite(layer.scale.grad))

    def test_predicate_outputs_are_microbatch_invariant_without_new_state(self):
        source = _FakeTransformerDecoder(num_layers=4, hidden_dim=4)
        scorer = FixedBoxFullTextScorer(
            source, num_layers=3, max_text_len=8, expression_microbatch=4
        )
        scorer.eval()
        candidate_hs = torch.randn((2, 2, 4))
        candidate_boxes = torch.rand((2, 2, 4))
        captions = [["red object", "blue object"], ["small car", "large car"]]
        valid = torch.ones((2, 2), dtype=torch.bool)
        predicate_mask = torch.zeros((2, 2, 8), dtype=torch.bool)
        predicate_mask[:, :, 1] = True
        state_keys_before = set(scorer.state_dict())

        full = scorer(
            candidate_hs=candidate_hs,
            candidate_boxes=candidate_boxes,
            expression_captions=captions,
            expression_valid_mask=valid,
            expression_predicate_token_mask=predicate_mask,
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
            expression_microbatch=4,
        )
        split = scorer(
            candidate_hs=candidate_hs,
            candidate_boxes=candidate_boxes,
            expression_captions=captions,
            expression_valid_mask=valid,
            expression_predicate_token_mask=predicate_mask,
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
            expression_microbatch=1,
        )

        for key in (
            "final_token_logits",
            "final_phrase_logits",
            "final_predicate_logits",
            "predicate_token_mask",
        ):
            self.assertTrue(torch.equal(full[key], split[key]), key)
        self.assertEqual(state_keys_before, set(scorer.state_dict()))
        self.assertFalse(
            any(key.startswith("validity_head.") for key in scorer.state_dict())
        )

    def test_validity_head_is_zero_initialized_detached_residual(self):
        source = _FakeTransformerDecoder(num_layers=4, hidden_dim=4)
        scorer = FixedBoxFullTextScorer(
            source,
            num_layers=3,
            max_text_len=8,
            expression_microbatch=4,
            use_validity_head=True,
        )
        self.assertIsNotNone(scorer.validity_head)
        self.assertTrue(
            torch.equal(
                scorer.validity_head[-1].weight,
                torch.zeros_like(scorer.validity_head[-1].weight),
            )
        )
        self.assertTrue(
            torch.equal(
                scorer.validity_head[-1].bias,
                torch.zeros_like(scorer.validity_head[-1].bias),
            )
        )

        outputs = scorer(
            candidate_hs=torch.randn((1, 2, 4)),
            candidate_boxes=torch.rand((1, 2, 4)),
            expression_captions=[["red object", "blue object"]],
            expression_valid_mask=torch.ones((1, 2), dtype=torch.bool),
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )
        self.assertTrue(
            torch.equal(
                outputs["layer_validity_logits"],
                outputs["layer_phrase_logits"].detach(),
            )
        )
        outputs["final_validity_logits"].sum().backward()
        self.assertGreater(float(scorer.validity_head[-1].weight.grad.abs().sum()), 0.0)
        for layer in scorer.decoder.layers:
            self.assertIsNotNone(layer.scale.grad)
            self.assertEqual(float(layer.scale.grad.abs().sum()), 0.0)

        head_before = {
            key: value.detach().clone()
            for key, value in scorer.validity_head.state_dict().items()
        }
        replacement = _FakeTransformerDecoder(num_layers=5, hidden_dim=4)
        scorer.load_from_decoder(replacement)
        for key, value in scorer.validity_head.state_dict().items():
            self.assertTrue(torch.equal(value, head_before[key]), key)

    def test_validity_outputs_are_microbatch_invariant(self):
        source = _FakeTransformerDecoder(num_layers=4, hidden_dim=4)
        scorer = FixedBoxFullTextScorer(
            source,
            num_layers=3,
            max_text_len=8,
            use_validity_head=True,
        )
        scorer.eval()
        with torch.no_grad():
            scorer.validity_head[-1].weight.fill_(0.125)
            scorer.validity_head[-1].bias.fill_(-0.25)
        kwargs = {
            "candidate_hs": torch.randn((2, 2, 4)),
            "candidate_boxes": torch.rand((2, 2, 4)),
            "expression_captions": [
                ["red object", "blue object"],
                ["small car", "large car"],
            ],
            "expression_valid_mask": torch.ones((2, 2), dtype=torch.bool),
        }
        full = scorer(
            **kwargs,
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
            expression_microbatch=4,
        )
        split = scorer(
            **kwargs,
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
            expression_microbatch=1,
        )
        for key in (
            "layer_validity_logits",
            "final_validity_logits",
            "final_score",
        ):
            torch.testing.assert_close(
                full[key], split[key], rtol=0.0, atol=1e-6, msg=key
            )

    def test_decoupled_validity_is_uniform_and_cannot_update_rank_decoder(self):
        torch.manual_seed(17)
        source = _FakeTransformerDecoder(num_layers=4, hidden_dim=4)
        scorer = FixedBoxFullTextScorer(
            source,
            num_layers=3,
            max_text_len=8,
            expression_microbatch=2,
            use_validity_head=True,
            decouple_validity_from_ranking=True,
            validity_pool_temperature=0.2,
        )
        with torch.no_grad():
            scorer.validity_head[-1].weight.fill_(0.1)
            scorer.validity_head[-1].bias.fill_(0.05)

        candidate_hs = torch.randn((2, 3, 4), requires_grad=True)
        candidate_boxes = torch.rand((2, 3, 4), requires_grad=True)
        outputs = scorer(
            candidate_hs=candidate_hs,
            candidate_boxes=candidate_boxes,
            expression_captions=[
                ["red object", "blue object"],
                ["small car", "large car"],
            ],
            expression_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )

        residual = (
            outputs["final_validity_logits"]
            - outputs["final_phrase_logits"].detach()
        )
        self.assertTrue(
            torch.allclose(residual, residual[:, :1, :].expand_as(residual))
        )
        self.assertTrue(
            torch.equal(
                outputs["final_validity_logits"].argsort(dim=1),
                outputs["final_phrase_logits"].argsort(dim=1),
            )
        )
        self.assertTrue(
            torch.allclose(
                residual[:, 0, :], outputs["final_validity_gate_logits"]
            )
        )

        outputs["final_score"].sum().backward()
        self.assertIsNone(candidate_hs.grad)
        self.assertIsNone(candidate_boxes.grad)
        for layer in scorer.decoder.layers:
            self.assertIsNone(layer.scale.grad)
        self.assertIsNotNone(scorer.validity_head[-1].weight.grad)
        self.assertGreater(
            float(scorer.validity_head[-1].weight.grad.abs().sum()), 0.0
        )

    def test_rank_optimizer_step_is_functionally_isolated_from_confidence(self):
        torch.manual_seed(29)
        scorer = FixedBoxFullTextScorer(
            _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
            num_layers=3,
            max_text_len=8,
            use_validity_head=True,
            decouple_validity_from_ranking=True,
            patch_rank_fusion=True,
            patch_rank_weight=0.5,
        )
        scorer.train()
        self.assertFalse(scorer.confidence_decoder.training)
        self.assertFalse(
            any(parameter.requires_grad for parameter in scorer.confidence_decoder.parameters())
        )
        kwargs = {
            "candidate_hs": torch.randn((1, 3, 4)),
            "candidate_boxes": torch.rand((1, 3, 4)),
            "candidate_patch_logits": torch.tensor([[0.7, -0.2, 0.1]]),
            "expression_captions": [["red object", "blue object"]],
            "expression_valid_mask": torch.ones((1, 2), dtype=torch.bool),
        }
        before = scorer(
            **kwargs,
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )
        confidence_before = before["final_validity_logits"].detach().clone()
        rank_before = before["final_phrase_logits"].detach().clone()

        optimizer = torch.optim.SGD(scorer.decoder.parameters(), lr=0.25)
        optimizer.zero_grad(set_to_none=True)
        before["final_phrase_logits"].square().sum().backward()
        optimizer.step()
        after = scorer(
            **kwargs,
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )

        self.assertFalse(torch.equal(rank_before, after["final_phrase_logits"]))
        self.assertTrue(
            torch.equal(confidence_before, after["final_validity_logits"])
        )

    def test_gate_only_broadcasts_scalar_and_does_not_add_frozen_base(self):
        torch.manual_seed(37)
        scorer = FixedBoxFullTextScorer(
            _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
            num_layers=3,
            max_text_len=8,
            use_validity_head=True,
            decouple_validity_from_ranking=True,
            patch_rank_fusion=True,
            patch_rank_weight=0.25,
            confidence_output_mode="gate_only",
        )
        scorer.eval()
        with torch.no_grad():
            scorer.validity_head[-1].weight.fill_(0.1)
            scorer.validity_head[-1].bias.fill_(0.05)
        kwargs = {
            "candidate_hs": torch.randn((2, 3, 4)),
            "candidate_boxes": torch.rand((2, 3, 4)),
            "expression_captions": [
                ["red object", "blue object"],
                ["small car", "large car"],
            ],
            "expression_valid_mask": torch.ones((2, 2), dtype=torch.bool),
        }
        base = scorer(
            **kwargs,
            candidate_patch_logits=torch.zeros((2, 3)),
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )
        shifted = scorer(
            **kwargs,
            candidate_patch_logits=torch.full((2, 3), 4.0),
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )

        expected_layers = shifted["layer_validity_gate_logits"].unsqueeze(2).expand_as(
            shifted["layer_validity_logits"]
        )
        self.assertTrue(
            torch.allclose(shifted["layer_validity_logits"], expected_layers)
        )
        expected_final = shifted["final_validity_gate_logits"][:, None, :].expand_as(
            shifted["final_validity_logits"]
        )
        self.assertTrue(
            torch.allclose(shifted["final_validity_logits"], expected_final)
        )
        self.assertTrue(
            torch.allclose(
                shifted["final_score"],
                shifted["final_score"][:, :1, :].expand_as(shifted["final_score"]),
            )
        )

        expected_base_shift = torch.ones_like(shifted["final_confidence_base_logits"])
        self.assertTrue(
            torch.allclose(
                shifted["final_confidence_base_logits"]
                - base["final_confidence_base_logits"],
                expected_base_shift,
            )
        )
        self.assertTrue(
            torch.allclose(
                shifted["final_phrase_logits"] - base["final_phrase_logits"],
                expected_base_shift,
            )
        )
        self.assertTrue(
            torch.allclose(
                shifted["final_validity_logits"],
                base["final_validity_logits"],
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_v19_base_plus_gate_and_global_max_identities(self):
        torch.manual_seed(43)
        scorer = FixedBoxFullTextScorer(
            _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
            num_layers=3,
            max_text_len=8,
            use_validity_head=True,
            decouple_validity_from_ranking=True,
            patch_rank_fusion=True,
            patch_rank_weight=0.25,
            confidence_output_mode="base_plus_gate",
            explicit_confidence_output_contract=True,
        )
        with torch.no_grad():
            scorer.validity_head[-1].weight.fill_(0.1)
            scorer.validity_head[-1].bias.fill_(0.05)
        outputs = scorer(
            candidate_hs=torch.randn((2, 3, 4)),
            candidate_boxes=torch.rand((2, 3, 4)),
            candidate_patch_logits=torch.tensor(
                [[0.7, -0.2, 0.1], [0.4, 0.3, -0.5]]
            ),
            expression_captions=[
                ["red object", "blue object"],
                ["small car", "large car"],
            ],
            expression_valid_mask=torch.ones((2, 2), dtype=torch.bool),
            context_provider=self._context_provider(
                [], hidden_dim=4, max_text_len=8
            ),
        )

        expected_layers = (
            outputs["layer_confidence_base_logits"]
            + outputs["layer_validity_gate_logits"].unsqueeze(2)
        )
        expected_final = (
            outputs["final_confidence_base_logits"]
            + outputs["final_validity_gate_logits"][:, None, :]
        )
        self.assertTrue(
            torch.equal(outputs["layer_validity_logits"], expected_layers)
        )
        self.assertTrue(
            torch.equal(outputs["final_validity_logits"], expected_final)
        )
        self.assertTrue(
            torch.equal(
                outputs["final_validity_logits"].max(dim=1).values,
                outputs["final_confidence_base_logits"].max(dim=1).values
                + outputs["final_validity_gate_logits"],
            )
        )
        self.assertTrue(
            torch.equal(
                outputs["final_score"].max(dim=1).values,
                outputs["final_validity_logits"].max(dim=1).values.sigmoid(),
            )
        )

    def test_decoupled_output_modes_are_bidirectionally_isolated(self):
        for output_mode, explicit in (
            ("gate_only", False),
            ("base_plus_gate", True),
        ):
            with self.subTest(output_mode=output_mode):
                torch.manual_seed(41)
                scorer = FixedBoxFullTextScorer(
                    _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
                    num_layers=3,
                    max_text_len=8,
                    use_validity_head=True,
                    decouple_validity_from_ranking=True,
                    confidence_output_mode=output_mode,
                    explicit_confidence_output_contract=explicit,
                )
                scorer.train()
                self.assertFalse(scorer.confidence_decoder.training)
                self.assertFalse(
                    any(
                        parameter.requires_grad
                        for parameter in scorer.confidence_decoder.parameters()
                    )
                )
                with torch.no_grad():
                    scorer.validity_head[-1].weight.fill_(0.1)
                    scorer.validity_head[-1].bias.fill_(0.05)
                kwargs = {
                    "candidate_hs": torch.randn((1, 3, 4)),
                    "candidate_boxes": torch.rand((1, 3, 4)),
                    "expression_captions": [["red object", "blue object"]],
                    "expression_valid_mask": torch.ones((1, 2), dtype=torch.bool),
                }
                context = lambda: self._context_provider(
                    [], hidden_dim=4, max_text_len=8
                )
                before = scorer(**kwargs, context_provider=context())
                rank_before = before["final_phrase_logits"].detach().clone()
                base_before = before["final_confidence_base_logits"].detach().clone()
                confidence_before = before["final_validity_logits"].detach().clone()

                rank_optimizer = torch.optim.SGD(
                    scorer.decoder.parameters(), lr=0.25
                )
                rank_optimizer.zero_grad(set_to_none=True)
                before["final_phrase_logits"].square().sum().backward()
                rank_optimizer.step()
                after_rank = scorer(**kwargs, context_provider=context())
                self.assertFalse(
                    torch.equal(rank_before, after_rank["final_phrase_logits"])
                )
                self.assertTrue(
                    torch.equal(
                        base_before, after_rank["final_confidence_base_logits"]
                    )
                )
                self.assertTrue(
                    torch.equal(
                        confidence_before, after_rank["final_validity_logits"]
                    )
                )

                rank_after = after_rank["final_phrase_logits"].detach().clone()
                base_after_rank = after_rank[
                    "final_confidence_base_logits"
                ].detach().clone()
                confidence_after_rank = after_rank[
                    "final_validity_logits"
                ].detach().clone()
                validity_optimizer = torch.optim.SGD(
                    scorer.validity_head.parameters(), lr=0.25
                )
                scorer.zero_grad(set_to_none=True)
                after_rank["final_validity_logits"].square().sum().backward()
                for parameter in scorer.decoder.parameters():
                    self.assertIsNone(parameter.grad)
                for parameter in scorer.confidence_decoder.parameters():
                    self.assertIsNone(parameter.grad)
                self.assertTrue(
                    all(
                        parameter.grad is not None
                        for parameter in scorer.validity_head.parameters()
                    )
                )
                validity_optimizer.step()
                after_validity = scorer(**kwargs, context_provider=context())
                self.assertTrue(
                    torch.equal(rank_after, after_validity["final_phrase_logits"])
                )
                self.assertTrue(
                    torch.equal(
                        base_after_rank,
                        after_validity["final_confidence_base_logits"],
                    )
                )
                self.assertFalse(
                    torch.equal(
                        confidence_after_rank,
                        after_validity["final_validity_logits"],
                    )
                )

    def test_decoupled_invalid_expression_slot_has_finite_confidence_gradients(self):
        for output_mode in ("base_plus_gate", "gate_only"):
            with self.subTest(output_mode=output_mode):
                scorer = FixedBoxFullTextScorer(
                    _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
                    num_layers=3,
                    max_text_len=8,
                    use_validity_head=True,
                    decouple_validity_from_ranking=True,
                    confidence_output_mode=output_mode,
                )
                score_mask = torch.zeros((1, 2, 8), dtype=torch.bool)
                score_mask[0, 0, 1] = True
                outputs = scorer(
                    candidate_hs=torch.randn((1, 3, 4)),
                    candidate_boxes=torch.rand((1, 3, 4)),
                    expression_captions=[["red object", ""]],
                    expression_valid_mask=torch.tensor([[True, False]]),
                    expression_score_token_mask=score_mask,
                    context_provider=self._context_provider(
                        [], hidden_dim=4, max_text_len=8
                    ),
                )
                self.assertEqual(
                    outputs["final_validity_gate_logits"][0, 1].item(), 0.0
                )
                self.assertTrue(
                    torch.equal(
                        outputs["final_score"][0, :, 1],
                        torch.zeros_like(outputs["final_score"][0, :, 1]),
                    )
                )
                outputs["final_score"].sum().backward()
                for parameter in scorer.validity_head.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all().item()))

    def test_patch_prior_is_added_to_rank_and_frozen_confidence_base(self):
        torch.manual_seed(31)
        scorer = FixedBoxFullTextScorer(
            _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
            num_layers=3,
            max_text_len=8,
            use_validity_head=True,
            decouple_validity_from_ranking=True,
            patch_rank_fusion=True,
            patch_rank_weight=0.25,
        )
        scorer.eval()
        kwargs = {
            "candidate_hs": torch.randn((1, 2, 4)),
            "candidate_boxes": torch.rand((1, 2, 4)),
            "expression_captions": [["red object", "blue object"]],
            "expression_valid_mask": torch.ones((1, 2), dtype=torch.bool),
        }
        zero = scorer(
            **kwargs,
            candidate_patch_logits=torch.zeros((1, 2)),
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )
        patch_logits = torch.tensor([[1.2, -0.8]])
        fused = scorer(
            **kwargs,
            candidate_patch_logits=patch_logits,
            context_provider=self._context_provider([], hidden_dim=4, max_text_len=8),
        )
        expected = (0.25 * patch_logits).unsqueeze(-1).expand_as(
            fused["final_phrase_logits"]
        )
        self.assertTrue(
            torch.allclose(
                fused["final_phrase_logits"] - zero["final_phrase_logits"],
                expected,
            )
        )
        self.assertTrue(
            torch.allclose(
                fused["final_confidence_base_logits"]
                - zero["final_confidence_base_logits"],
                expected,
            )
        )

    def test_checkpoint_rejects_score_contract_value_mismatch(self):
        class Wrapper(nn.Module):
            def __init__(self, patch_rank_weight):
                super().__init__()
                self.stage_b_fixed_text_scorer = FixedBoxFullTextScorer(
                    _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
                    num_layers=3,
                    max_text_len=8,
                    use_validity_head=True,
                    decouple_validity_from_ranking=True,
                    patch_rank_fusion=True,
                    patch_rank_weight=patch_rank_weight,
                    exclude_canonical_from_score=True,
                    candidate_topk=3,
                )

        source = Wrapper(0.5)
        destination = Wrapper(0.75)
        with self.assertRaisesRegex(ValueError, "contract_mismatches="):
            validate_stage_b_fixed_text_scorer_checkpoint(
                destination,
                source.state_dict(),
                checkpoint_label="different patch prior",
            )

    def test_explicit_output_contracts_are_persistent_and_legacy_v15_stays_v3(self):
        class Wrapper(nn.Module):
            def __init__(self, output_mode, *, explicit=False):
                super().__init__()
                self.stage_b_fixed_text_scorer = FixedBoxFullTextScorer(
                    _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
                    num_layers=3,
                    max_text_len=8,
                    use_validity_head=True,
                    decouple_validity_from_ranking=True,
                    patch_rank_fusion=True,
                    exclude_canonical_from_score=True,
                    candidate_topk=3,
                    confidence_output_mode=output_mode,
                    explicit_confidence_output_contract=explicit,
                )

        v15 = Wrapper("base_plus_gate")
        v15_contract = {
            key
            for key in v15.stage_b_fixed_text_scorer.state_dict()
            if key.startswith("_score_contract_")
        }
        self.assertEqual(
            v15_contract,
            {
                "_score_contract_version",
                "_score_contract_decoupled_confidence",
                "_score_contract_validity_pool_temperature",
                "_score_contract_patch_rank_fusion",
                "_score_contract_patch_rank_weight",
                "_score_contract_exclude_canonical",
                "_score_contract_candidate_topk",
            },
        )
        self.assertEqual(
            int(v15.stage_b_fixed_text_scorer._score_contract_version.item()), 3
        )

        v16 = Wrapper("gate_only")
        self.assertEqual(
            int(v16.stage_b_fixed_text_scorer._score_contract_version.item()), 4
        )
        self.assertEqual(
            int(
                v16.stage_b_fixed_text_scorer
                ._score_contract_confidence_output_mode.item()
            ),
            1,
        )

        v19 = Wrapper("base_plus_gate", explicit=True)
        self.assertEqual(
            int(v19.stage_b_fixed_text_scorer._score_contract_version.item()), 5
        )
        self.assertEqual(
            int(
                v19.stage_b_fixed_text_scorer
                ._score_contract_confidence_output_mode.item()
            ),
            0,
        )
        self.assertEqual(
            {
                key
                for key in v19.stage_b_fixed_text_scorer.state_dict()
                if key.startswith("_score_contract_")
            },
            {
                *v15_contract,
                "_score_contract_confidence_output_mode",
            },
        )
        with self.assertRaisesRegex(ValueError, "missing=.*confidence_output_mode"):
            validate_stage_b_fixed_text_scorer_checkpoint(
                v16,
                v15.state_dict(),
                checkpoint_label="v15 checkpoint into v16 scorer",
            )
        with self.assertRaisesRegex(ValueError, "unexpected=.*confidence_output_mode"):
            validate_stage_b_fixed_text_scorer_checkpoint(
                v15,
                v16.state_dict(),
                checkpoint_label="v16 checkpoint into v15 scorer",
            )
        with self.assertRaisesRegex(ValueError, "missing=.*confidence_output_mode"):
            validate_stage_b_fixed_text_scorer_checkpoint(
                v19,
                v15.state_dict(),
                checkpoint_label="v15 checkpoint into v19 scorer",
            )
        with self.assertRaisesRegex(ValueError, "unexpected=.*confidence_output_mode"):
            validate_stage_b_fixed_text_scorer_checkpoint(
                v15,
                v19.state_dict(),
                checkpoint_label="v19 checkpoint into v15 scorer",
            )
        for destination, source, label in (
            (v19, v16, "v16-v18 checkpoint into v19 scorer"),
            (v16, v19, "v19 checkpoint into v16-v18 scorer"),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "contract_mismatches="
            ):
                validate_stage_b_fixed_text_scorer_checkpoint(
                    destination,
                    source.state_dict(),
                    checkpoint_label=label,
                )

        corrupt = {
            key: value.detach().clone() for key, value in v16.state_dict().items()
        }
        corrupt[
            "stage_b_fixed_text_scorer._score_contract_confidence_output_mode"
        ].zero_()
        with self.assertRaisesRegex(ValueError, "contract_mismatches="):
            validate_stage_b_fixed_text_scorer_checkpoint(
                v16,
                corrupt,
                checkpoint_label="wrong v16 output mode",
            )

        corrupt_v19 = {
            key: value.detach().clone() for key, value in v19.state_dict().items()
        }
        corrupt_v19[
            "stage_b_fixed_text_scorer._score_contract_confidence_output_mode"
        ].fill_(1)
        with self.assertRaisesRegex(ValueError, "contract_mismatches="):
            validate_stage_b_fixed_text_scorer_checkpoint(
                v19,
                corrupt_v19,
                checkpoint_label="wrong v19 output mode",
            )

    def test_gate_only_requires_decoupled_validity_and_v16_config_enables_it(self):
        for kwargs in (
            {"use_validity_head": False, "decouple_validity_from_ranking": False},
            {"use_validity_head": True, "decouple_validity_from_ranking": False},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "gate_only.*requires decoupled confidence"
            ):
                FixedBoxFullTextScorer(
                    _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
                    num_layers=3,
                    max_text_len=8,
                    confidence_output_mode="gate_only",
                    **kwargs,
                )

        root = Path(__file__).resolve().parents[1]
        config = runpy.run_path(
            str(root / "config/ablations/cfg_stageb_v16_gate_only_confidence.py")
        )
        self.assertTrue(config["stage_b_v15_decoupled_confidence"])
        self.assertTrue(config["stage_b_v14_validity_head"])
        self.assertEqual(config["stage_b_v16_confidence_output_mode"], "gate_only")
        builder = (root / "models/GroundingDINO/groundingdino.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"stage_b_v16_confidence_output_mode"', builder)

    def test_v19_explicit_contract_requires_decoupled_base_plus_gate(self):
        for kwargs in (
            {
                "use_validity_head": True,
                "decouple_validity_from_ranking": True,
                "confidence_output_mode": "gate_only",
            },
            {
                "use_validity_head": True,
                "decouple_validity_from_ranking": False,
                "confidence_output_mode": "base_plus_gate",
            },
            {
                "use_validity_head": False,
                "decouple_validity_from_ranking": False,
                "confidence_output_mode": "base_plus_gate",
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "explicit_confidence_output_contract requires"
            ):
                FixedBoxFullTextScorer(
                    _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
                    num_layers=3,
                    max_text_len=8,
                    explicit_confidence_output_contract=True,
                    **kwargs,
                )

        root = Path(__file__).resolve().parents[1]
        config = runpy.run_path(
            str(
                root
                / "config/ablations/cfg_stageb_v19_full_text_base_plus_gate.py"
            )
        )
        self.assertTrue(config["stage_b_v15_decoupled_confidence"])
        self.assertTrue(config["stage_b_v14_validity_head"])
        self.assertFalse(config["stage_b_v15_exclude_canonical_from_score"])
        self.assertEqual(
            config["stage_b_v16_confidence_output_mode"], "base_plus_gate"
        )
        self.assertTrue(config["stage_b_v19_explicit_confidence_output_contract"])
        self.assertEqual(config["stage_b_v14_tail_queue_weight"], 1.0)
        self.assertEqual(config["stage_b_v15_tail_queue_pair_weight"], 1.0)
        self.assertEqual(
            config["stage_b_v15_tail_queue_positive_trust_weight"], 1.0
        )
        for legacy_config_name in (
            "cfg_stageb_v16_gate_only_confidence.py",
            "cfg_stageb_v17_full_text_gate_only.py",
            "cfg_stageb_v18_strong_fpr_tail.py",
        ):
            legacy_config = runpy.run_path(
                str(root / "config/ablations" / legacy_config_name)
            )
            self.assertEqual(
                legacy_config["stage_b_v16_confidence_output_mode"],
                "gate_only",
                legacy_config_name,
            )
            self.assertFalse(
                legacy_config.get(
                    "stage_b_v19_explicit_confidence_output_contract", False
                ),
                legacy_config_name,
            )
        builder = (root / "models/GroundingDINO/groundingdino.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"stage_b_v19_explicit_confidence_output_contract"', builder)

    def test_fixed_decoder_rejects_bbox_head(self):
        source = _FakeTransformerDecoder(num_layers=1, hidden_dim=4)
        kwargs = {
            "tgt": torch.zeros((1, 2, 4)),
            "reference_boxes": torch.tensor(
                [[[0.0, 1.0, 0.0001, 0.9999], [0.5, 0.5, 0.2, 0.2]]]
            ),
            "memory": torch.zeros((1, 3, 4)),
            "memory_key_padding_mask": torch.zeros((1, 3), dtype=torch.bool),
            "memory_pos": torch.zeros((1, 3, 4)),
            "level_start_index": torch.tensor([0]),
            "spatial_shapes": torch.tensor([[1, 3]]),
            "valid_ratios": torch.ones((1, 1, 2)),
            "memory_text": torch.zeros((1, 4, 4)),
            "text_attention_mask": torch.zeros((1, 4), dtype=torch.bool),
        }
        with self.assertRaisesRegex(RuntimeError, "bbox_embed=None"):
            source.forward_fixed_external(**kwargs)

        source.bbox_embed = None
        _outputs, references = source.forward_fixed_external(**kwargs)
        self.assertEqual(len(references), 1)
        self.assertTrue(torch.equal(references[0], kwargs["reference_boxes"]))

    def test_post_load_sync_uses_last_source_layers(self):
        source = _FakeTransformerDecoder(num_layers=4, hidden_dim=4)
        scorer = FixedBoxFullTextScorer(source, num_layers=3, max_text_len=8)
        replacement = _FakeTransformerDecoder(num_layers=5, hidden_dim=4)
        with torch.no_grad():
            for idx, layer in enumerate(replacement.layers):
                layer.scale.fill_(10.0 + idx)
        scorer.load_from_decoder(replacement)
        self.assertEqual(
            [layer.scale.detach().item() for layer in scorer.decoder.layers],
            [12.0, 13.0, 14.0],
        )
        self.assertIsNone(scorer.decoder.bbox_embed)

    def test_full_text_checkpoint_warm_start_is_exact_and_scorer_only(self):
        source = _FakeTransformerDecoder(num_layers=5, hidden_dim=4)
        with torch.no_grad():
            for idx, layer in enumerate(source.layers):
                layer.scale.fill_(20.0 + idx)
            source.ref_point_head.weight.fill_(7.0)
            source.ref_point_head.bias.fill_(8.0)
            source.norm.weight.fill_(9.0)
            source.norm.bias.fill_(10.0)
            for head in source.bbox_embed:
                head.weight.fill_(99.0)

        source_state = {
            "transformer.decoder." + key: value.detach().clone()
            for key, value in source.state_dict().items()
        }
        source_state["backbone.weight"] = torch.full((1,), 123.0)
        scorer = FixedBoxFullTextScorer(
            _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
            num_layers=3,
            max_text_len=8,
            use_validity_head=True,
            decouple_validity_from_ranking=True,
            confidence_output_mode="base_plus_gate",
            explicit_confidence_output_contract=True,
        )
        with torch.no_grad():
            scorer.validity_head[-1].bias.fill_(0.75)
        validity_before = {
            key: value.detach().clone()
            for key, value in scorer.validity_head.state_dict().items()
        }
        trainability_before = {
            name: parameter.requires_grad
            for name, parameter in scorer.named_parameters()
        }
        contract_before = {
            key: value.detach().clone()
            for key, value in scorer.state_dict().items()
            if key.startswith("_score_contract_")
        }

        audit = scorer.load_from_full_text_checkpoint_state(
            source_state,
            checkpoint_label="mature full-text checkpoint",
        )

        self.assertEqual(
            [layer.scale.detach().item() for layer in scorer.decoder.layers],
            [22.0, 23.0, 24.0],
        )
        self.assertEqual(
            [layer.scale.detach().item() for layer in scorer.confidence_decoder.layers],
            [22.0, 23.0, 24.0],
        )
        self.assertTrue(
            torch.equal(
                scorer.decoder.ref_point_head.weight,
                source.ref_point_head.weight,
            )
        )
        self.assertTrue(torch.equal(scorer.decoder.norm.bias, source.norm.bias))
        self.assertIsNone(scorer.decoder.bbox_embed)
        self.assertIsNone(scorer.decoder.class_embed)
        self.assertIsNone(scorer.confidence_decoder.bbox_embed)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in scorer.confidence_decoder.parameters()
            )
        )
        for key, value in scorer.validity_head.state_dict().items():
            self.assertTrue(torch.equal(value, validity_before[key]), key)
        self.assertEqual(
            {
                name: parameter.requires_grad
                for name, parameter in scorer.named_parameters()
            },
            trainability_before,
        )
        contract_after = {
            key: value
            for key, value in scorer.state_dict().items()
            if key.startswith("_score_contract_")
        }
        self.assertEqual(set(contract_after), set(contract_before))
        for key, value in contract_after.items():
            self.assertTrue(torch.equal(value, contract_before[key]), key)
        self.assertEqual(int(scorer._score_contract_version.item()), 5)
        self.assertEqual(
            int(scorer._score_contract_confidence_output_mode.item()), 0
        )
        self.assertEqual(audit["source_decoder_num_layers"], 5)
        self.assertEqual(audit["selected_source_layer_indices"], [2, 3, 4])
        self.assertEqual(audit["loaded_num_layers"], 3)
        self.assertGreater(audit["loaded_tensor_count"], 0)

    def test_full_text_checkpoint_warm_start_fails_before_mutation(self):
        source = _FakeTransformerDecoder(num_layers=5, hidden_dim=4)
        complete = {
            "transformer.decoder." + key: value.detach().clone()
            for key, value in source.state_dict().items()
        }
        corruptions = {}
        missing = dict(complete)
        missing.pop("transformer.decoder.norm.bias")
        corruptions["missing="] = missing
        wrong_shape = dict(complete)
        wrong_shape["transformer.decoder.layers.4.scale"] = torch.zeros(2)
        corruptions["shape_mismatches="] = wrong_shape
        unexpected = dict(complete)
        unexpected["transformer.decoder.layers.4.extra"] = torch.zeros(1)
        corruptions["unexpected="] = unexpected

        for expected_error, source_state in corruptions.items():
            with self.subTest(expected_error=expected_error):
                scorer = FixedBoxFullTextScorer(
                    _FakeTransformerDecoder(num_layers=4, hidden_dim=4),
                    num_layers=3,
                    max_text_len=8,
                    use_validity_head=True,
                    decouple_validity_from_ranking=True,
                )
                before = {
                    key: value.detach().clone()
                    for key, value in scorer.state_dict().items()
                }
                with self.assertRaisesRegex(ValueError, expected_error):
                    scorer.load_from_full_text_checkpoint_state(
                        source_state,
                        checkpoint_label="corrupt full-text checkpoint",
                    )
                after = scorer.state_dict()
                self.assertEqual(set(after), set(before))
                for key, value in after.items():
                    self.assertTrue(torch.equal(value, before[key]), key)


class NoncanonicalTokenMaskTest(unittest.TestCase):
    def test_removes_shared_canonical_tokens_and_falls_back_for_canonical_only(self):
        expression_ids = torch.tensor(
            [
                [
                    [101, 300, 200, 400, 102, 0],
                    [101, 200, 102, 0, 0, 0],
                ],
                [
                    [101, 500, 201, 102, 0, 0],
                    [101, 201, 102, 0, 0, 0],
                ],
            ]
        )
        expression_attention = expression_ids.ne(0)
        canonical_ids = torch.tensor(
            [[101, 200, 102, 0], [101, 201, 102, 0]]
        )
        canonical_attention = canonical_ids.ne(0)
        eligible = expression_attention & ~(
            expression_ids.eq(101) | expression_ids.eq(102)
        )
        result = build_stage_b_noncanonical_token_masks_from_ids(
            expression_ids,
            expression_attention,
            canonical_ids,
            canonical_attention,
            torch.tensor([[True, True], [True, False]]),
            eligible,
            max_text_len=8,
        )
        responsibility_separated = build_stage_b_noncanonical_token_masks_from_ids(
            expression_ids,
            expression_attention,
            canonical_ids,
            canonical_attention,
            torch.tensor([[True, True], [True, False]]),
            eligible,
            max_text_len=8,
            fallback_to_eligible=False,
        )

        self.assertEqual(
            result[0, 0].nonzero(as_tuple=False).flatten().tolist(), [1, 3]
        )
        self.assertEqual(
            result[0, 1].nonzero(as_tuple=False).flatten().tolist(), [1]
        )
        self.assertEqual(
            result[1, 0].nonzero(as_tuple=False).flatten().tolist(), [1]
        )
        self.assertFalse(bool(result[1, 1].any().item()))
        self.assertEqual(
            responsibility_separated[0, 0]
            .nonzero(as_tuple=False)
            .flatten()
            .tolist(),
            [1, 3],
        )
        self.assertFalse(bool(responsibility_separated[0, 1].any().item()))
        self.assertEqual(
            responsibility_separated[1, 0]
            .nonzero(as_tuple=False)
            .flatten()
            .tolist(),
            [1],
        )
        self.assertFalse(bool(responsibility_separated[1, 1].any().item()))


if __name__ == "__main__":
    unittest.main()
