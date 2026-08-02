import inspect
import types
import unittest

import torch
from torch import nn

import main as main_module
from engine import (
    _build_stage_b_v11_expression_slots,
    _clip_stage_b_v15_grad_norms,
    _set_stage_b_v11_training_mode,
)
from main import (
    _audit_stage_b_v11_optimizer_group_lrs,
    _audit_stage_b_v11_trainable_parameters,
    _isolate_stage_b_v15_validity_optimizer_group,
    _maybe_sync_stage_b_v11_scorer_from_decoder,
)
from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)
from models.GroundingDINO.groundingdino import GroundingDINO
from models.GroundingDINO.stage_b_fixed_text_scorer import (
    validate_stage_b_fixed_text_scorer_checkpoint,
)


def _target(cap_list, is_tn, pair_stride):
    return {
        "cap_list": cap_list,
        "is_tn": torch.tensor(is_tn, dtype=torch.bool),
        "verifier_pair_stride": torch.tensor([pair_stride]),
        "verifier_num_patch_slots": torch.tensor([1]),
    }


class _DummyV11Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_a = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.9))
        self.stage_b_fixed_text_scorer = nn.Sequential(
            nn.Linear(4, 4), nn.Dropout(p=0.5)
        )


class _SyncScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))
        self.calls = []

    def load_from_decoder(self, decoder):
        self.calls.append(decoder)


class _SyncModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.stage_b_fixed_text_scorer = _SyncScorer()
        self.transformer = types.SimpleNamespace(decoder=object())


class _DummyV15Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank_weight = nn.Parameter(torch.zeros(()))
        self.validity_head = nn.Linear(1, 1, bias=True)


class _DummyV15Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_b_fixed_text_scorer = _DummyV15Scorer()


class StageBV11IntegrationTest(unittest.TestCase):
    def test_trainability_is_finalized_before_ddp_and_optimizer_groups(self):
        source = inspect.getsource(main_module.main)
        ddp_index = source.index("torch.nn.parallel.DistributedDataParallel(")
        optimizer_group_index = source.index(
            "param_dicts = _stage_b_gdino_adapter_optimizer_groups("
        )

        trainability_operations = (
            'if args.freeze_keywords is not None:',
            'unfreeze_n = int(getattr(args, "unfreeze_decoder_last_n_layers"',
            'only_train_keywords = getattr(args, "only_train_keywords", None)',
            '_freeze_and_audit_stage_b_gdino_adapter(',
            '_audit_stage_b_v11_trainable_parameters(',
            'model_without_ddp.stage_b_verifier.freeze_bert()',
        )
        for operation in trainability_operations:
            with self.subTest(operation=operation):
                self.assertLess(source.index(operation), ddp_index)
        self.assertLess(ddp_index, optimizer_group_index)

    def test_v15_validity_optimizer_group_is_isolated_with_full_coverage(self):
        model = _DummyV15Model()
        scorer = model.stage_b_fixed_text_scorer
        source_groups = [
            {
                "params": list(model.parameters()),
                "lr": 2e-5,
                "weight_decay": 0.03,
            }
        ]

        groups = _isolate_stage_b_v15_validity_optimizer_group(
            source_groups,
            model,
            validity_lr=5e-4,
        )

        validity_groups = [
            group
            for group in groups
            if group.get("stage_b_v15_validity_group", False)
        ]
        self.assertEqual(len(validity_groups), 1)
        validity_group = validity_groups[0]
        self.assertEqual(validity_group["lr"], 5e-4)
        self.assertEqual(validity_group["weight_decay"], 0.03)
        self.assertEqual(
            {id(p) for p in validity_group["params"]},
            {id(p) for p in scorer.validity_head.parameters()},
        )

        occurrences = {}
        for group in groups:
            for parameter in group["params"]:
                occurrences[id(parameter)] = occurrences.get(id(parameter), 0) + 1
        self.assertEqual(
            occurrences,
            {id(parameter): 1 for parameter in model.parameters()},
        )
        self.assertEqual(
            _audit_stage_b_v11_optimizer_group_lrs(
                groups,
                base_lr=2e-5,
                validity_lr=5e-4,
            ),
            5e-4,
        )
        bad_groups = [dict(group) for group in groups]
        bad_groups[0]["lr"] = 1e-3
        with self.assertRaisesRegex(RuntimeError, "exceeds the base rate"):
            _audit_stage_b_v11_optimizer_group_lrs(
                bad_groups,
                base_lr=2e-5,
                validity_lr=5e-4,
            )

    def test_v15_validity_optimizer_split_rejects_bad_coverage(self):
        model = _DummyV15Model()
        params = list(model.parameters())
        with self.assertRaisesRegex(RuntimeError, "duplicate="):
            _isolate_stage_b_v15_validity_optimizer_group(
                [{"params": params + [params[0]], "lr": 2e-5}],
                model,
                validity_lr=5e-4,
            )
        with self.assertRaisesRegex(RuntimeError, "missing="):
            _isolate_stage_b_v15_validity_optimizer_group(
                [{"params": params[:-1], "lr": 2e-5}],
                model,
                validity_lr=5e-4,
            )

    def test_v15_rank_and_confidence_gradients_are_clipped_independently(self):
        model = _DummyV15Model()
        scorer = model.stage_b_fixed_text_scorer
        scorer.rank_weight.grad = torch.tensor(2.0)
        scorer.validity_head.weight.grad = torch.tensor([[100.0]])

        stats = _clip_stage_b_v15_grad_norms(model, max_norm=1.0)

        self.assertAlmostEqual(stats["grad_norm_rank_preclip"], 2.0, places=5)
        self.assertAlmostEqual(
            stats["grad_norm_confidence_preclip"], 100.0, places=5
        )
        self.assertAlmostEqual(stats["grad_norm_rank_postclip"], 1.0, places=5)
        self.assertAlmostEqual(
            stats["grad_norm_confidence_postclip"], 1.0, places=5
        )
        # A shared norm would shrink this to about 0.02 because of the much
        # larger confidence gradient. Independent clipping keeps its full step.
        self.assertAlmostEqual(float(scorer.rank_weight.grad), 1.0, places=5)

    def test_expression_slots_keep_pair_text_independent_and_clean_slot_invalid(self):
        captions, valid = _build_stage_b_v11_expression_slots(
            [
                _target(["red car", "blue car"], [False, True], 2),
                _target(["small dog"], [False], 1),
            ],
            torch.device("cpu"),
        )
        self.assertEqual(
            captions,
            [["red car .", "blue car ."], ["small dog .", "object ."]],
        )
        self.assertTrue(
            torch.equal(
                valid,
                torch.tensor([[True, True], [True, False]], dtype=torch.bool),
            )
        )
        self.assertNotIn("red car . blue car", captions[0])

    def test_expression_slots_reject_bad_pair_order_and_multiple_patch_slots(self):
        bad_order = _target(["red car", "blue car"], [True, False], 2)
        with self.assertRaisesRegex(ValueError, "positive expression, local TN"):
            _build_stage_b_v11_expression_slots(
                [bad_order], torch.device("cpu")
            )

        multiple_patch = _target(["red car"], [False], 1)
        multiple_patch["verifier_num_patch_slots"] = torch.tensor([2])
        with self.assertRaisesRegex(ValueError, "one localization patch"):
            _build_stage_b_v11_expression_slots(
                [multiple_patch], torch.device("cpu")
            )

    def test_exclamation_expression_gets_recognized_period_boundary(self):
        captions, valid = _build_stage_b_v11_expression_slots(
            [_target(["stop!"], [False], 1)], torch.device("cpu")
        )
        self.assertEqual(captions, [["stop! .", "object ."]])
        self.assertEqual(valid.tolist(), [[True, False]])

    def test_train_mode_keeps_stage_a_eval_and_only_scorer_train(self):
        model = _DummyV11Model()
        model.train()
        _set_stage_b_v11_training_mode(model)

        self.assertFalse(model.training)
        self.assertFalse(model.stage_a.training)
        self.assertTrue(model.stage_b_fixed_text_scorer.training)
        first = model.stage_a(torch.ones((2, 4)))
        second = model.stage_a(torch.ones((2, 4)))
        self.assertTrue(torch.equal(first, second))

    def test_candidate_selection_is_repeatable_and_does_not_change_boxes(self):
        query_hs = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
        pred_boxes = torch.linspace(0.0, 1.0, 2 * 5 * 4).view(2, 5, 4)
        patch_score = torch.tensor(
            [[0.1, 0.9, 0.3, 0.7, 0.5], [0.8, 0.2, 0.4, 0.6, 0.0]]
        )
        boxes_before = pred_boxes.clone()
        first = GroundingDINO.select_stage_b_v11_candidates(
            query_hs, pred_boxes, patch_score, topk=3
        )
        second = GroundingDINO.select_stage_b_v11_candidates(
            query_hs, pred_boxes, patch_score, topk=3
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.equal(pred_boxes, boxes_before))
        expected_boxes = torch.gather(
            pred_boxes,
            1,
            first[0].unsqueeze(-1).expand(-1, -1, 4),
        )
        self.assertTrue(torch.equal(first[2], expected_boxes))
        self.assertFalse(first[1].requires_grad)
        self.assertFalse(first[2].requires_grad)

    def test_scatter_masks_noncandidates_and_invalid_clean_slot(self):
        logits = torch.tensor([[[1.0, -1.0], [2.0, -2.0]]])
        scores = logits.sigmoid()
        idx = torch.tensor([[3, 1]])
        valid = torch.tensor([[True, False]])
        dense_logits, dense_score, dense_mask = (
            GroundingDINO.scatter_stage_b_v11_candidates(
                logits, scores, idx, num_queries=5, expression_valid_mask=valid
            )
        )
        self.assertEqual(tuple(dense_logits.shape), (1, 5, 2))
        self.assertEqual(dense_score[0, :, 1].tolist(), [0.0] * 5)
        self.assertFalse(bool(dense_mask[0, :, 1].any()))
        self.assertTrue(torch.equal(dense_mask[0, :, 0], torch.tensor([0, 1, 0, 1, 0], dtype=torch.bool)))
        self.assertEqual(dense_score[0, 0, 0].item(), 0.0)
        self.assertEqual(dense_logits[0, 0, 0].item(), torch.finfo(logits.dtype).min)

    def test_v14_validity_drives_dense_and_compatibility_outputs(self):
        source = inspect.getsource(GroundingDINO.forward)
        self.assertIn('scorer_out["final_validity_logits"]', source)
        self.assertIn('"stage_b_v14_final_validity_logits"', source)
        self.assertIn(
            'out["stage_b_v7_final_logits"] = dense_logits',
            source,
        )

    def test_checkpoint_sync_only_initializes_missing_v11_scorer_state(self):
        model = _SyncModel()
        _maybe_sync_stage_b_v11_scorer_from_decoder(model, {"base.weight": 1}, None)
        self.assertEqual(model.stage_b_fixed_text_scorer.calls, [model.transformer.decoder])

        model.stage_b_fixed_text_scorer.calls.clear()
        _maybe_sync_stage_b_v11_scorer_from_decoder(
            model,
            {"stage_b_fixed_text_scorer.weight": torch.ones(3)},
            None,
        )
        self.assertEqual(model.stage_b_fixed_text_scorer.calls, [])

    def test_resume_checkpoint_requires_exact_scorer_state(self):
        model = _SyncModel()
        state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        validate_stage_b_fixed_text_scorer_checkpoint(
            model, state, checkpoint_label="complete"
        )

        missing = dict(state)
        missing.pop("stage_b_fixed_text_scorer.weight")
        with self.assertRaisesRegex(ValueError, "missing="):
            validate_stage_b_fixed_text_scorer_checkpoint(
                model, missing, checkpoint_label="truncated"
            )

        wrong_shape = dict(state)
        wrong_shape["stage_b_fixed_text_scorer.weight"] = torch.zeros(4)
        with self.assertRaisesRegex(ValueError, "shape_mismatches="):
            validate_stage_b_fixed_text_scorer_checkpoint(
                model, wrong_shape, checkpoint_label="wrong shape"
            )

        extra_layer = dict(state)
        extra_layer["stage_b_fixed_text_scorer.decoder.layers.9.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "unexpected="):
            validate_stage_b_fixed_text_scorer_checkpoint(
                model, extra_layer, checkpoint_label="wrong layer count"
            )

        self.assertIn(
            "validate_stage_b_fixed_text_scorer_checkpoint",
            inspect.getsource(main_module.main),
        )
        self.assertIn(
            "_maybe_sync_stage_b_v11_scorer_from_decoder",
            inspect.getsource(main_module.main),
        )

    def test_trainable_audit_allows_only_scorer(self):
        model = _SyncModel()
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
        count = _audit_stage_b_v11_trainable_parameters(
            model, minimum=3, maximum=3
        )
        self.assertEqual(count, 3)

        model.base.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "unexpected trainable"):
            _audit_stage_b_v11_trainable_parameters(
                model, minimum=3, maximum=100
            )

    def test_optimizer_group_lr_audit_rejects_rate_above_base(self):
        maximum = _audit_stage_b_v11_optimizer_group_lrs(
            [{"params": []}, {"params": [], "lr": 2e-6}],
            base_lr=2e-5,
        )
        self.assertEqual(maximum, 2e-5)
        with self.assertRaisesRegex(RuntimeError, "exceeds the base rate"):
            _audit_stage_b_v11_optimizer_group_lrs(
                [{"params": [], "lr": 0.1}], base_lr=2e-5
            )

    def test_mixed_clean_and_paired_loss_forward_is_finite(self):
        _, expression_valid = _build_stage_b_v11_expression_slots(
            [
                _target(["red car", "blue car"], [False, True], 2),
                _target(["small dog"], [False], 1),
            ],
            torch.device("cpu"),
        )
        criterion = StageBFixedTextCriterion(
            listwise_weight=0.2,
            local_tn_rank_weight=1.0,
            local_anchor_weight=0.5,
            positive_anchor_logit=0.5,
            negative_anchor_logit=-0.5,
            global_tn_negative_weight=0.0,
            global_tn_tail_weight=0.0,
        )
        positive_logits = torch.tensor(
            [[0.2, -0.1, -0.5], [0.3, -0.2, -0.4]], requires_grad=True
        )
        local_tn_logits = torch.tensor(
            [[-0.2, 0.1, -0.3], [-4.0, -4.0, -4.0]], requires_grad=True
        )
        candidate_ious = torch.tensor(
            [[0.8, 0.1, 0.2], [0.7, 0.1, 0.2]]
        )
        loss_dict = criterion(
            candidate_logits=positive_logits,
            candidate_ious=candidate_ious,
            local_tn_logits=local_tn_logits,
            local_tn_mask=expression_valid[:, 1:2],
        )
        total = sum(
            loss_dict[key] * criterion.weight_dict[key]
            for key in criterion.weight_dict
        )
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(positive_logits.grad).all())
        self.assertTrue(torch.isfinite(local_tn_logits.grad).all())
        self.assertNotEqual(float(positive_logits.grad[1, 0]), 0.0)
        self.assertEqual(float(local_tn_logits.grad[1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
