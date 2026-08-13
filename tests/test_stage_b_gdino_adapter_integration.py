import types
import unittest
import json
from pathlib import Path

import torch
from torch import nn

from engine import (
    _build_stage_b_gdino_adapter_pair_captions,
    _build_stage_b_gdino_adapter_rank_captions,
    _clip_stage_b_gdino_adapter_grad_norms,
    _forward_stage_b_gdino_adapter_pair,
    _set_stage_b_gdino_adapter_training_mode,
)
from main import (
    _freeze_and_audit_stage_b_gdino_adapter,
    _stage_b_gdino_adapter_optimizer_groups,
)
from models.GroundingDINO.stage_b_gdino_score_adapter import (
    StageBGDINOScoreAdapter,
    StageBGDINOScoreAdapterCriterion,
    validate_stage_b_gdino_score_adapter_checkpoint,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_stageb_tn_val as tn_eval
from tools import eval_text_groundingdino_refcoco_tn as text_eval
from tools.stageb_eval_records import (
    EvalManifest,
    extract_adapter_tn_pair_captions,
)
from util.misc import NestedTensor
from util.slconfig import SLConfig


def _target(scope="benchmark_dataft_alltn"):
    return {
        "cap_list": ["red car", "blue car"],
        "is_tn": torch.tensor([False, True], dtype=torch.bool),
        "verifier_pair_stride": torch.tensor([2]),
        "verifier_num_patch_slots": torch.tensor([1]),
        "global_tn_verified": torch.tensor(
            [scope == "image_global_topk_verified"], dtype=torch.bool
        ),
        "benchmark_dataft_alltn": torch.tensor(
            [scope == "benchmark_dataft_alltn"], dtype=torch.bool
        ),
        "proposalset_proxy_verified": torch.tensor([False], dtype=torch.bool),
        "tn_scope": scope,
        "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=torch.float32),
    }


class _DummyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.9))
        self.stage_b_gdino_score_adapter = StageBGDINOScoreAdapter(
            hidden_dim=4, adapter_dim=5, gate_hidden_dim=6
        )
        self.calls = []

    def forward(self, samples, *, captions):
        self.calls.append((int(samples.tensors.shape[0]), list(captions)))
        batch_size = len(captions)
        query_hs = torch.arange(
            batch_size * 3 * 4,
            dtype=samples.tensors.dtype,
            device=samples.tensors.device,
        ).reshape(batch_size, 3, 4)
        base_score = torch.linspace(
            0.1,
            0.9,
            batch_size * 3,
            dtype=samples.tensors.dtype,
            device=samples.tensors.device,
        ).reshape(batch_size, 3)
        adapted = self.stage_b_gdino_score_adapter(query_hs, base_score)
        return {
            "stage_b_gdino_base_score": adapted["base_score"],
            "stage_b_gdino_rank_residual": adapted["rank_residual"],
            "stage_b_gdino_rank_score": adapted["rank_score"],
            "stage_b_gdino_confidence_score": adapted["confidence_score"],
            "pred_boxes": torch.zeros(batch_size, 3, 4, device=samples.tensors.device),
        }


class StageBGDINOAdapterIntegrationTest(unittest.TestCase):
    def test_adapter_configs_do_not_redeclare_argparse_only_keys(self):
        for path in (
            "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py",
            "config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py",
            "config/ablations/cfg_stageb_gdino_score_adapter_fixed_top1_verified.py",
        ):
            cfg = SLConfig.fromfile(path)
            self.assertNotIn("find_unused_params", cfg._cfg_dict, path)

    def test_adapter_tn_caption_extraction_is_separate_and_fail_closed(self):
        benchmark = {
            "caption": "red car . blue car .",
            "cap_list": ["red car", "blue car"],
            "is_tn": torch.tensor([False, True], dtype=torch.bool),
        }
        positive, negative, valid = extract_adapter_tn_pair_captions(
            [benchmark]
        )
        self.assertEqual(positive, ["red car ."])
        self.assertEqual(negative, ["blue car ."])
        self.assertEqual(valid.tolist(), [True])

        strict = {
            "cap_list": ["blue car"],
            "is_tn": torch.tensor([True], dtype=torch.bool),
            "rank_positive_captions": ["red car ."],
            "has_rank_positive": torch.tensor([True], dtype=torch.bool),
        }
        positive, negative, _valid = extract_adapter_tn_pair_captions([strict])
        self.assertEqual(positive, ["red car ."])
        self.assertEqual(negative, ["blue car ."])

        malformed = dict(benchmark)
        malformed["is_tn"] = torch.tensor([True, True], dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "exactly one TN"):
            extract_adapter_tn_pair_captions([malformed])

    def test_rank_only_caption_audit_requires_one_positive_nonnegative_row(self):
        target = {
            "cap_list": ["red car"],
            "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]]),
            "is_negative": torch.tensor([0], dtype=torch.int64),
        }
        self.assertEqual(
            _build_stage_b_gdino_adapter_rank_captions([target]),
            ["red car ."],
        )
        target["is_negative"] = torch.tensor([1], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "is_negative=0"):
            _build_stage_b_gdino_adapter_rank_captions([target])

    def test_u0_rank_caption_audit_requires_exact_positive_patch_episode(self):
        target = {
            "cap_list": ["red car"],
            "boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]]),
            "is_negative_episode": torch.tensor([0], dtype=torch.int64),
            "is_lvis_neg_category_episode": torch.tensor([0], dtype=torch.int64),
        }
        self.assertEqual(
            _build_stage_b_gdino_adapter_rank_captions(
                [target], u0_patch_episode=True
            ),
            ["red car ."],
        )
        for key in (
            "is_negative_episode",
            "is_lvis_neg_category_episode",
        ):
            malformed = dict(target)
            malformed[key] = torch.tensor([1], dtype=torch.int64)
            with self.assertRaisesRegex(ValueError, rf"{key}=0"):
                _build_stage_b_gdino_adapter_rank_captions(
                    [malformed], u0_patch_episode=True
                )

            missing = dict(target)
            missing.pop(key)
            with self.assertRaisesRegex(ValueError, rf"{key}=0"):
                _build_stage_b_gdino_adapter_rank_captions(
                    [missing], u0_patch_episode=True
                )

    def test_pair_scope_is_fail_closed_and_supports_only_two_protocols(self):
        benchmark = _target("benchmark_dataft_alltn")
        positive, negative, scope = _build_stage_b_gdino_adapter_pair_captions(
            [benchmark], "benchmark_dataft_alltn"
        )
        self.assertEqual(positive, ["red car ."])
        self.assertEqual(negative, ["blue car ."])
        self.assertEqual(scope.tolist(), [2])

        global_target = _target("image_global_topk_verified")
        _, _, scope = _build_stage_b_gdino_adapter_pair_captions(
            [global_target], "image_global_topk_verified"
        )
        self.assertEqual(scope.tolist(), [1])

        benchmark["proposalset_proxy_verified"] = "false"
        with self.assertRaisesRegex(RuntimeError, "proposal-proxy"):
            _build_stage_b_gdino_adapter_pair_captions(
                [benchmark], "benchmark_dataft_alltn"
            )
        proposal = _target("benchmark_dataft_alltn")
        proposal["proposalset_proxy_verified"] = torch.tensor([True])
        with self.assertRaisesRegex(RuntimeError, "proposal-proxy"):
            _build_stage_b_gdino_adapter_pair_captions(
                [proposal], "benchmark_dataft_alltn"
            )

    def test_pair_forward_is_one_2b_model_call(self):
        model = _DummyAdapterModel()
        samples = NestedTensor(
            torch.ones(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.bool)
        )
        targets = [_target(), _target()]
        outputs = _forward_stage_b_gdino_adapter_pair(
            model,
            samples,
            targets,
            ["red car .", "small dog ."],
            ["blue car .", "large dog ."],
            torch.tensor([2, 2]),
        )

        self.assertEqual(
            model.calls,
            [(4, ["red car .", "small dog .", "blue car .", "large dog ."])],
        )
        self.assertEqual(tuple(outputs["stage_b_gdino_rank_score"].shape), (2, 3))
        self.assertEqual(
            tuple(
                outputs["stage_b_gdino_tn_outputs"][
                    "stage_b_gdino_confidence_score"
                ].shape
            ),
            (2, 3),
        )

    def test_tn_evaluator_adapter_forward_is_one_ordinary_nonpatch_2b_call(self):
        model = _DummyAdapterModel()
        samples = NestedTensor(
            torch.ones(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.bool)
        )
        raw_targets = []
        for positive, negative in (("red car", "blue car"), ("small dog", "large dog")):
            target = _target()
            target["cap_list"] = [positive, negative]
            target["is_tn"] = torch.tensor([False, True], dtype=torch.bool)
            target["caption"] = f"{positive} . {negative} ."
            target["rank_positive_captions"] = [positive + " ."]
            target["has_rank_positive"] = torch.tensor([True])
            raw_targets.append(target)

        negative, positive, _targets, valid = tn_eval._forward_pair(
            model,
            (samples, raw_targets),
            torch.device("cpu"),
            amp=False,
        )

        self.assertEqual(
            model.calls,
            [(4, ["red car .", "small dog .", "blue car .", "large dog ."])],
        )
        self.assertEqual(valid.tolist(), [True, True])
        self.assertEqual(tuple(positive["stage_b_gdino_confidence_score"].shape), (2, 3))
        self.assertEqual(tuple(negative["stage_b_gdino_confidence_score"].shape), (2, 3))

    def test_ref_evaluator_adapter_forward_is_ordinary_nonpatch(self):
        model = _DummyAdapterModel()
        samples = NestedTensor(
            torch.ones(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.bool)
        )
        raw_targets = []
        for caption in ("red car .", "small dog ."):
            target = _target()
            target["caption"] = caption
            raw_targets.append(target)
        outputs, targets = ref_eval._forward(
            model,
            (samples, raw_targets),
            torch.device("cpu"),
            amp=False,
            cfg=types.SimpleNamespace(stage_b_gdino_score_adapter=True),
        )

        self.assertEqual(model.calls, [(2, ["red car .", "small dog ."])])
        self.assertEqual(tuple(outputs["stage_b_gdino_rank_score"].shape), (2, 3))
        self.assertEqual(len(targets), 2)

    def test_ref_evaluator_adapter_datasetinfos_have_no_patch_dependency(self):
        data_root = Path("/tmp/pivot-data")
        anno = Path("/tmp/ref.jsonl")
        for make_datasetinfo in (
            ref_eval._make_datasetinfo,
            text_eval._make_datasetinfo,
        ):
            adapter_info = make_datasetinfo(
                data_root,
                "refcoco_val",
                anno,
                adapter_no_support=True,
            )
            self.assertTrue(adapter_info["stage_b_gdino_adapter_no_support"])
            self.assertTrue(adapter_info["stage_b_gdino_adapter_ref_eval"])
            self.assertEqual(adapter_info["neg_episode_prob"], 0.0)
            self.assertFalse(adapter_info["tn_balance_sampling"])
            self.assertNotIn("support_patch_tsv", adapter_info)
            self.assertNotIn("support_patch_image_root", adapter_info)

            legacy_info = make_datasetinfo(data_root, "refcoco_val", anno)
            self.assertIn("support_patch_tsv", legacy_info)
            self.assertNotIn("stage_b_gdino_adapter_no_support", legacy_info)

    def test_freeze_optimizer_mode_and_clipping_are_branch_isolated(self):
        model = _DummyAdapterModel()
        count = _freeze_and_audit_stage_b_gdino_adapter(model)
        self.assertEqual(
            count,
            sum(p.numel() for p in model.stage_b_gdino_score_adapter.parameters()),
        )
        groups = _stage_b_gdino_adapter_optimizer_groups(
            model, rank_lr=1e-4, gate_lr=3e-4
        )
        self.assertEqual([group["stage_b_gdino_branch"] for group in groups], ["rank", "confidence"])
        self.assertEqual([group["lr"] for group in groups], [1e-4, 3e-4])

        model.train()
        _set_stage_b_gdino_adapter_training_mode(model)
        self.assertFalse(model.training)
        self.assertFalse(model.base.training)
        self.assertTrue(model.stage_b_gdino_score_adapter.training)

        adapter = model.stage_b_gdino_score_adapter
        adapter.rank_output.weight.grad = torch.full_like(adapter.rank_output.weight, 2.0)
        adapter.confidence_gate[-1].weight.grad = torch.full_like(
            adapter.confidence_gate[-1].weight, 100.0
        )
        stats = _clip_stage_b_gdino_adapter_grad_norms(model, 1.0)
        self.assertGreater(stats["grad_norm_gdino_rank_preclip"], 1.0)
        self.assertGreater(stats["grad_norm_gdino_confidence_preclip"], 1.0)
        self.assertAlmostEqual(stats["grad_norm_gdino_rank_postclip"], 1.0, places=5)
        self.assertAlmostEqual(
            stats["grad_norm_gdino_confidence_postclip"], 1.0, places=5
        )

        rank_model = _DummyAdapterModel()
        _freeze_and_audit_stage_b_gdino_adapter(rank_model, train_mode="rank_only")
        rank_groups = _stage_b_gdino_adapter_optimizer_groups(
            rank_model,
            rank_lr=3e-5,
            gate_lr=3e-4,
            train_mode="rank_only",
        )
        self.assertEqual([group["stage_b_gdino_branch"] for group in rank_groups], ["rank"])
        self.assertTrue(all(p.requires_grad for p in rank_model.stage_b_gdino_score_adapter.rank_parameters()))
        self.assertTrue(all(not p.requires_grad for p in rank_model.stage_b_gdino_score_adapter.gate_parameters()))

        confidence_model = _DummyAdapterModel()
        _freeze_and_audit_stage_b_gdino_adapter(
            confidence_model, train_mode="confidence_only"
        )
        confidence_groups = _stage_b_gdino_adapter_optimizer_groups(
            confidence_model,
            rank_lr=3e-5,
            gate_lr=3e-4,
            train_mode="confidence_only",
        )
        self.assertEqual(
            [group["stage_b_gdino_branch"] for group in confidence_groups],
            ["confidence"],
        )

    def test_criterion_scope_queue_and_checkpoint_roundtrip(self):
        criterion = StageBGDINOScoreAdapterCriterion(
            tn_scope="benchmark_dataft_alltn",
            train_mode="confidence_only",
            queue_size=8,
            queue_min_count=1,
            paired_margin_weight=0.25,
        )
        rank = torch.tensor(
            [[0.1, 0.0, -0.2], [0.2, -0.1, -0.3]], requires_grad=True
        )
        positive_confidence = torch.tensor(
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.2]], requires_grad=True
        )
        negative_confidence = torch.tensor(
            [[0.5, 0.1, 0.0], [0.4, 0.2, 0.1]], requires_grad=True
        )
        boxes = torch.tensor(
            [
                [[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]],
                [[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]],
            ]
        )
        targets = [_target(), _target()]
        outputs = {
            "stage_b_gdino_rank_score": rank,
            "stage_b_gdino_confidence_score": positive_confidence,
            "pred_boxes": boxes,
            "stage_b_gdino_tn_scope_code": torch.tensor([2, 2]),
            "stage_b_gdino_tn_outputs": {
                "stage_b_gdino_confidence_score": negative_confidence
            },
        }
        losses = criterion(outputs, targets)
        total = sum(losses[key] * criterion.weight_dict[key] for key in criterion.weight_dict)
        total.backward()
        self.assertIsNone(rank.grad)
        self.assertGreater(float(positive_confidence.grad.abs().sum()), 0.0)
        self.assertGreater(float(negative_confidence.grad.abs().sum()), 0.0)
        criterion.commit_tail_queue(True)
        self.assertEqual(int(criterion.fpr_queue_count), 2)

        restored = StageBGDINOScoreAdapterCriterion(
            tn_scope="benchmark_dataft_alltn",
            train_mode="confidence_only",
            queue_size=8,
            queue_min_count=1,
            paired_margin_weight=0.25,
        )
        restored.load_state_dict(criterion.state_dict(), strict=True)
        self.assertTrue(
            torch.equal(restored.fpr_positive_queue, criterion.fpr_positive_queue)
        )
        wrong_scope = StageBGDINOScoreAdapterCriterion(
            tn_scope="image_global_topk_verified",
            train_mode="confidence_only",
            queue_size=8,
            queue_min_count=1,
        )
        with self.assertRaisesRegex(RuntimeError, "scope mismatch"):
            wrong_scope.load_state_dict(criterion.state_dict(), strict=True)

        wrong_phase = StageBGDINOScoreAdapterCriterion(train_mode="rank_only", queue_size=0, queue_min_count=0)
        with self.assertRaisesRegex(RuntimeError, "train-mode mismatch"):
            wrong_phase.load_state_dict(criterion.state_dict(), strict=True)

    def test_p3_criterion_logs_trust_and_rejects_objective_drift(self):
        criterion = StageBGDINOScoreAdapterCriterion(
            tn_scope="benchmark_dataft_alltn",
            train_mode="confidence_only",
            confidence_objective="detached_recent_q05_trust",
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            queue_size=4,
            queue_min_count=1,
            paired_margin_weight=0.0,
        )
        positive_confidence = torch.tensor(
            [[0.7, 0.2], [0.6, 0.3]], requires_grad=True
        )
        negative_confidence = torch.tensor(
            [[0.5, 0.1], [0.4, 0.2]], requires_grad=True
        )
        positive_gate = torch.tensor([-0.01, -0.03], requires_grad=True)
        negative_gate = torch.tensor([0.1, -0.1], requires_grad=True)
        outputs = {
            "stage_b_gdino_confidence_score": positive_confidence,
            "stage_b_gdino_confidence_gate": positive_gate,
            "stage_b_gdino_tn_scope_code": torch.tensor([2, 2]),
            "stage_b_gdino_tn_outputs": {
                "stage_b_gdino_confidence_score": negative_confidence,
                "stage_b_gdino_confidence_gate": negative_gate,
            },
        }

        losses = criterion(outputs, [_target(), _target()])
        losses["loss_stage_b_gdino_confidence"].backward()

        self.assertIn("stage_b_gdino_confidence_negative_loss", losses)
        self.assertIn("stage_b_gdino_confidence_trust_loss", losses)
        self.assertIn("stage_b_gdino_current_positive_threshold", losses)
        self.assertIn("stage_b_gdino_threshold_drift", losses)
        self.assertIn("stage_b_gdino_current_q05_fpr", losses)
        self.assertIn("stage_b_gdino_positive_gate_q05", losses)
        self.assertAlmostEqual(
            float(losses["stage_b_gdino_positive_trust_violation_rate"]),
            0.5,
            places=6,
        )
        self.assertLess(float(positive_gate.grad[0]), 0.0)
        self.assertAlmostEqual(
            float(positive_gate.grad[1] - positive_gate.grad[0]),
            -0.5,
            places=6,
        )
        self.assertIsNone(positive_confidence.grad)
        self.assertTrue(bool((negative_confidence.grad[:, 0] > 0).all().item()))
        self.assertEqual(float(negative_confidence.grad[:, 1].abs().sum()), 0.0)

        state = criterion.state_dict()
        self.assertEqual(int(state["criterion_confidence_objective_code"]), 2)
        self.assertAlmostEqual(
            float(state["criterion_positive_trust_margin"]), 0.02, places=6
        )
        self.assertEqual(int(state["criterion_queue_size"]), 4)
        p2 = StageBGDINOScoreAdapterCriterion(
            tn_scope="benchmark_dataft_alltn",
            train_mode="confidence_only",
            confidence_objective="queue_q05_st",
            queue_size=4,
            queue_min_count=1,
        )
        with self.assertRaisesRegex(RuntimeError, "confidence-objective mismatch"):
            p2.load_state_dict(state, strict=True)
        missing_mode = dict(state)
        missing_mode.pop("criterion_confidence_objective_code")
        with self.assertRaisesRegex(RuntimeError, "missing.*confidence-objective"):
            criterion.load_state_dict(missing_mode, strict=True)

    def test_total_trust_criterion_owns_deployed_positive_score(self):
        criterion = StageBGDINOScoreAdapterCriterion(
            tn_scope="benchmark_dataft_alltn",
            train_mode="confidence_only",
            confidence_objective="detached_recent_q05_total_trust",
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            queue_size=4,
            queue_min_count=1,
            paired_margin_weight=0.0,
        )
        criterion._pending_queue_payload = torch.tensor(
            [[0.8, 0.0], [0.7, 0.0]]
        )
        criterion.commit_tail_queue(True)
        positive_base = torch.tensor(
            [[0.2, 0.1], [0.3, 0.0]], requires_grad=True
        )
        negative_base = torch.tensor(
            [[0.6, 0.1], [0.5, 0.2]], requires_grad=True
        )
        positive_gate = torch.tensor([-0.01, -0.02], requires_grad=True)
        negative_gate = torch.tensor([0.1, 0.05], requires_grad=True)
        outputs = {
            "stage_b_gdino_confidence_score": (
                positive_base.detach() + positive_gate[:, None]
            ),
            "stage_b_gdino_confidence_gate": positive_gate,
            "stage_b_gdino_tn_scope_code": torch.tensor([2, 2]),
            "stage_b_gdino_tn_outputs": {
                "stage_b_gdino_confidence_score": (
                    negative_base.detach() + negative_gate[:, None]
                ),
                "stage_b_gdino_confidence_gate": negative_gate,
            },
        }
        losses = criterion(outputs, [_target(), _target()])
        losses["loss_stage_b_gdino_confidence"].backward()

        self.assertEqual(int(criterion.criterion_confidence_objective_code), 3)
        self.assertIn(
            "stage_b_gdino_confidence_positive_score_trust_loss", losses
        )
        self.assertGreater(
            float(losses["stage_b_gdino_confidence_positive_score_trust_loss"]),
            0.0,
        )
        self.assertGreater(float(positive_gate.grad.abs().sum()), 0.0)
        self.assertIsNone(positive_base.grad)
        self.assertEqual(
            float(losses["stage_b_gdino_positive_score_trust_violation_rate"]),
            1.0,
        )
        restored = StageBGDINOScoreAdapterCriterion(
            tn_scope="benchmark_dataft_alltn",
            train_mode="confidence_only",
            confidence_objective="detached_recent_q05_total_trust",
            queue_size=4,
            queue_min_count=1,
        )
        restored.load_state_dict(criterion.state_dict(), strict=True)

    def test_p3_queue_retains_only_recent_values(self):
        criterion = StageBGDINOScoreAdapterCriterion(
            tn_scope="benchmark_dataft_alltn",
            train_mode="confidence_only",
            confidence_objective="detached_recent_q05_trust",
            queue_size=4,
            queue_min_count=1,
        )
        criterion._pending_queue_payload = torch.tensor(
            [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3], [0.4, -0.4]]
        )
        criterion.commit_tail_queue(True)
        criterion._pending_queue_payload = torch.tensor(
            [[0.5, -0.5], [0.6, -0.6]]
        )
        criterion.commit_tail_queue(True)

        self.assertEqual(int(criterion.fpr_queue_count), 4)
        self.assertEqual(int(criterion.fpr_queue_ptr), 2)
        self.assertTrue(
            torch.allclose(
                criterion._queue_values(criterion.fpr_positive_queue).sort().values,
                torch.tensor([0.3, 0.4, 0.5, 0.6]),
            )
        )
        self.assertTrue(
            torch.allclose(
                criterion._queue_values(criterion.fpr_negative_queue).sort().values,
                torch.tensor([-0.6, -0.5, -0.4, -0.3]),
            )
        )

    def test_rank_only_criterion_needs_no_tn_or_scope(self):
        criterion = StageBGDINOScoreAdapterCriterion(
            train_mode="rank_only",
            queue_size=0,
            queue_min_count=0,
        )
        base = torch.tensor([[0.2, 0.8, 0.1]])
        residual = torch.zeros_like(base, requires_grad=True)
        outputs = {
            "stage_b_gdino_base_score": base,
            "stage_b_gdino_rank_residual": residual,
            "stage_b_gdino_rank_score": base + residual,
            "pred_boxes": torch.tensor(
                [[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]]
            ),
        }
        losses = criterion(outputs, [{"boxes": torch.tensor([[0.2, 0.2, 0.2, 0.2]])}])
        self.assertIn("loss_stage_b_gdino_rank", losses)
        self.assertNotIn("loss_stage_b_gdino_confidence", losses)
        self.assertEqual(int(criterion.criterion_confidence_objective_code), 0)
        self.assertEqual(float(criterion.criterion_positive_trust_margin), 0.0)
        self.assertEqual(float(criterion.criterion_positive_trust_weight), 0.0)
        losses["loss_stage_b_gdino_rank"].backward()
        self.assertGreater(float(residual.grad.abs().sum()), 0.0)

    def test_checkpoint_validator_requires_complete_adapter(self):
        model = _DummyAdapterModel()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        validate_stage_b_gdino_score_adapter_checkpoint(
            model, state, checkpoint_label="complete"
        )
        missing = dict(state)
        missing.pop("stage_b_gdino_score_adapter.rank_output.weight")
        with self.assertRaisesRegex(ValueError, "missing="):
            validate_stage_b_gdino_score_adapter_checkpoint(
                model, missing, checkpoint_label="truncated"
            )

    def test_ref_uses_rank_and_tn_uses_confidence(self):
        cfg = types.SimpleNamespace(stage_b_v11_fixed_text=False, stage_b_v7=False)
        rank = torch.tensor([[0.1, 0.9, 0.2]])
        confidence = torch.tensor([[0.6, 1.4, 0.7]])
        outputs = {
            "stage_b_gdino_rank_score": rank,
            "stage_b_gdino_confidence_score": confidence,
        }
        self.assertTrue(
            torch.equal(ref_eval._slot_scores(outputs, cfg, 1.0), rank.unsqueeze(-1))
        )
        self.assertTrue(
            torch.equal(tn_eval._slot_scores(outputs, cfg, 1.0), confidence.unsqueeze(-1))
        )
        adapter_cfg = types.SimpleNamespace(
            stage_b_gdino_score_adapter=True,
            stage_b_v11_fixed_text=False,
            stage_b_v7=False,
        )
        with self.assertRaisesRegex(KeyError, "rank_score"):
            ref_eval._slot_scores({}, adapter_cfg, 1.0)
        with self.assertRaisesRegex(KeyError, "confidence_score"):
            tn_eval._slot_scores({}, adapter_cfg, 1.0)

    def test_text_evaluator_audits_base_and_selects_requested_branch(self):
        logits = torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=torch.float32
        )
        mask = torch.tensor([[False, True, True]])
        base = logits.sigmoid().masked_fill(~mask[:, None], 0.0).sum(-1) / 2.0
        rank = base + torch.tensor([[0.0, 0.1]])
        confidence = base + 0.2
        outputs = {
            "pred_logits": logits,
            "pred_boxes": torch.zeros(1, 2, 4),
            "stage_b_gdino_base_score": base,
            "stage_b_gdino_rank_score": rank,
            "stage_b_gdino_confidence_score": confidence,
            "stage_b_gdino_expression_token_mask": mask,
        }
        targets = [{"phrase_to_token_mask": mask}]
        observed, valid = text_eval._phrase_scores(
            outputs,
            targets,
            "phrase_to_token_mask",
            adapter_score_key="stage_b_gdino_rank_score",
        )
        self.assertTrue(torch.equal(observed, rank))
        self.assertEqual(valid.tolist(), [True])
        outputs["stage_b_gdino_base_score"] = base + 1e-3
        with self.assertRaisesRegex(RuntimeError, "base score drifted"):
            text_eval._phrase_scores(
                outputs,
                targets,
                "phrase_to_token_mask",
                adapter_score_key="stage_b_gdino_confidence_score",
            )
        outputs.pop("stage_b_gdino_confidence_score")
        with self.assertRaisesRegex(KeyError, "confidence_score"):
            text_eval._phrase_scores(
                outputs,
                targets,
                "phrase_to_token_mask",
                adapter_score_key="stage_b_gdino_confidence_score",
            )

    def test_strict_v2_manifests_validate_independently_of_train_scope(self):
        cfg = types.SimpleNamespace(
            stage_b_gdino_score_adapter=True,
            stage_b_gdino_tn_scope="benchmark_dataft_alltn",
        )
        root = Path(
            "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
        )
        for filename in (
            "eval_manifest.jsonl",
            "semantic_stageb_union_image_disjoint_manifest.jsonl",
        ):
            rows = [json.loads(line) for line in (root / filename).read_text().splitlines()]
            self.assertEqual(
                tn_eval._validate_adapter_tn_eval_manifest(cfg, rows),
                "image_global_topk_verified",
            )
        rows[0]["coverage_pass"] = False
        with self.assertRaisesRegex(ValueError, "strict-v2"):
            tn_eval._validate_adapter_tn_eval_manifest(cfg, rows)

    def test_ref_and_tn_manifest_alignment_fails_closed_on_order_drift(self):
        row = {
            "image_id": 11,
            "ann_id": 12,
            "ref_id": 13,
            "sent_id": 14,
            "sample_id": "ref:val:11:12:13:14",
        }
        manifest = EvalManifest(
            path=Path("dummy.jsonl"),
            task="ref",
            manifest_key="ref:val",
            split="val",
            sha256="0" * 64,
            rows=[row],
        )
        target = {
            key: torch.tensor([value], dtype=torch.int64)
            for key, value in row.items()
            if key != "sample_id"
        }
        target["sample_id"] = row["sample_id"]
        tn_eval._validate_eval_manifest_batch_alignment([target], manifest, 0)

        target["ref_id"] = torch.tensor([999], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "order drift"):
            tn_eval._validate_eval_manifest_batch_alignment([target], manifest, 0)

    def test_ref_accumulator_reports_frozen_box_recall_headroom(self):
        row = {
            "image_id": 11,
            "ann_id": 12,
            "ref_id": 13,
            "sent_id": 14,
            "sample_id": "ref:val:11:12:13:14",
        }
        manifest = EvalManifest(
            path=Path("dummy.jsonl"),
            task="ref",
            manifest_key="ref:val",
            split="val",
            sha256="0" * 64,
            rows=[row],
        )
        accumulator = text_eval.RefCocoTextAccumulator(
            [1], manifest=manifest, run_id="headroom"
        )
        outputs = {
            # Query 0 wins the text ranking but misses; query 1 is an exact box.
            "pred_logits": torch.tensor([[[5.0, -5.0], [0.0, -5.0], [-1.0, -5.0]]]),
            "pred_boxes": torch.tensor(
                [[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]]
            ),
        }
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "phrase_to_token_mask": torch.tensor([[True, False]]),
        }
        accumulator.update(outputs, [target])
        result = accumulator.result()

        self.assertEqual(result["acc50"], 0.0)
        self.assertEqual(result["recall50@all_queries"], 1.0)
        self.assertEqual(result["recall25@all_queries"], 1.0)
        self.assertAlmostEqual(
            result["mean_best_iou@all_queries"], 1.0, places=4
        )


if __name__ == "__main__":
    unittest.main()
