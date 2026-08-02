import types
import unittest

import torch

from models.GroundingDINO.stage_b_gdino_score_adapter import (
    aggregate_gdino_full_expression_score,
)
from models.GroundingDINO.stage_b_native_patch_category import (
    apply_native_patch_category_gate,
)
from tools import eval_refcoco_stageb as ref_eval
from tools import eval_stageb_tn_val as tn_eval
from tools import eval_text_groundingdino_refcoco_tn as joint_eval
from util.misc import NestedTensor


def _cfg(**overrides):
    values = {
        "stage_b_native_patch_category": True,
        "stage_b_native_patch_contract_version": 1,
        "stage_b_native_patch_confidence_trained": False,
        "stage_b_data_driven_score": False,
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
        "stage_b": False,
        "enable_patch_branch": True,
        "patch_gate_with_text": False,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _target(*, paired=False):
    target = {
        "caption": "small blue car .",
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "patch": torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
        "phrase_to_token_mask": torch.tensor(
            [[True, True, False, False]], dtype=torch.bool
        ),
    }
    if paired:
        target.update(
            cap_list=["red car", "small blue car"],
            is_tn=torch.tensor([False, True], dtype=torch.bool),
        )
    return target


def _batch(*, paired=False):
    samples = NestedTensor(
        torch.ones((1, 3, 4, 4), dtype=torch.float32),
        torch.zeros((1, 4, 4), dtype=torch.bool),
    )
    return samples, [_target(paired=paired)]


class _RecordingNativeRefModel:
    stage_b_native_patch_category = True

    def __init__(self, *, omit_key=None):
        self.calls = []
        self.omit_key = omit_key

    def __call__(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        batch_size = int(samples.tensors.shape[0])
        text = torch.full((batch_size, 900, 4), -6.0)
        text[:, 0, :2] = 4.0
        text[:, 1, :2] = 3.0
        patch = torch.zeros((batch_size, 900, 1))
        patch[:, 1, 0] = 10.0
        outputs = {
            "pred_logits_text": text,
            "pred_logits_patch": patch,
            "pred_boxes": torch.zeros((batch_size, 900, 4)),
            "phrase_to_token_mask": kwargs["phrase_to_token_mask"],
        }
        if self.omit_key is not None:
            outputs.pop(self.omit_key)
        return outputs


class _RecordingNativeConfidenceModel:
    stage_b_native_patch_category = True
    stage_b_native_patch_confidence_trained = True

    def __init__(self):
        self.calls = []

    def __call__(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        batch_size = int(samples.tensors.shape[0])
        confidence = torch.arange(
            batch_size * 3, dtype=torch.float32
        ).reshape(batch_size, 3)
        return {
            tn_eval._NATIVE_PATCH_CONFIDENCE_SCORE_KEY: confidence,
            "pred_boxes": torch.zeros((batch_size, 3, 4)),
        }


class StageBNativePatchCategoryEvalRoutingTest(unittest.TestCase):
    def test_ref_accepts_v1_through_v9_exact_contracts(self):
        self.assertTrue(ref_eval._validate_native_patch_category_config(_cfg()))
        d2_cfg = _cfg(
            stage_b_native_patch_contract_version=2,
            stage_b_native_patch_objective="d2_gate_aligned",
            stage_b_native_patch_gate_max_gap=3.0,
            stage_b_native_patch_score_clip=5.0,
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d2_cfg))

        d3_cfg = _cfg(
            stage_b_native_patch_contract_version=3,
            stage_b_native_patch_objective="d3_critical_winner",
            stage_b_native_patch_gate_max_gap=3.0,
            stage_b_native_patch_score_clip=5.0,
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d3_cfg))

        d4_cfg = _cfg(
            stage_b_native_patch_contract_version=4,
            stage_b_native_patch_objective=(
                "d4_positive_protected_critical_winner"
            ),
            stage_b_native_patch_gate_max_gap=3.0,
            stage_b_native_patch_score_clip=5.0,
            stage_b_native_patch_d4_critical_weight=2.0,
            stage_b_native_patch_d4_critical_keep_weight=1.0,
            stage_b_native_patch_d4_positive_keep_weight=32.0,
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d4_cfg))

        d5_cfg = _cfg(
            stage_b_native_patch_contract_version=5,
            stage_b_native_patch_objective="d5_active_tail_positive_barrier",
            stage_b_native_patch_gate_max_gap=3.0,
            stage_b_native_patch_score_clip=5.0,
            stage_b_native_patch_d5_keep_gap=2.75,
            stage_b_native_patch_d5_separation_gap=3.25,
            stage_b_native_patch_d5_temperature=0.25,
            stage_b_native_patch_d5_critical_weight=2.0,
            stage_b_native_patch_d5_critical_keep_weight=1.0,
            stage_b_native_patch_d5_active_gap=2.0,
            stage_b_native_patch_d5_target_gap=2.5,
            stage_b_native_patch_d5_positive_barrier_weight=2.0,
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d5_cfg))

        d6_cfg = _cfg(
            stage_b_native_patch_contract_version=6,
            stage_b_native_patch_objective="d6_direct_deployment_gap",
            stage_b_native_patch_gate_max_gap=3.0,
            stage_b_native_patch_score_clip=5.0,
            stage_b_native_patch_positive_iou_threshold=0.5,
            stage_b_native_patch_negative_iou_threshold=0.3,
            stage_b_native_patch_d6_weight=1.0,
            stage_b_native_patch_d6_keep_gap=2.75,
            stage_b_native_patch_d6_drop_gap=3.25,
            stage_b_native_patch_d6_drop_active_gap=3.75,
            stage_b_native_patch_d6_temperature=0.25,
            stage_b_native_patch_d6_drop_weight=2.0,
            stage_b_native_patch_d6_critical_keep_weight=1.0,
            stage_b_native_patch_d6_positive_active_gap=2.0,
            stage_b_native_patch_d6_positive_target_gap=2.5,
            stage_b_native_patch_d6_positive_barrier_weight=2.0,
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d6_cfg))

        d7_cfg = _cfg(
            stage_b_native_patch_contract_version=7,
            stage_b_native_patch_objective="d7_all_state_positive_anchor",
            stage_b_native_patch_gate_max_gap=3.0,
            stage_b_native_patch_score_clip=5.0,
            stage_b_native_patch_positive_iou_threshold=0.5,
            stage_b_native_patch_negative_iou_threshold=0.3,
            stage_b_native_patch_d7_weight=1.0,
            stage_b_native_patch_d7_keep_gap=2.75,
            stage_b_native_patch_d7_drop_gap=3.25,
            stage_b_native_patch_d7_drop_active_gap=3.75,
            stage_b_native_patch_d7_temperature=0.25,
            stage_b_native_patch_d7_drop_weight=2.0,
            stage_b_native_patch_d7_critical_keep_weight=1.0,
            stage_b_native_patch_d7_positive_active_gap=2.0,
            stage_b_native_patch_d7_positive_target_gap=2.5,
            stage_b_native_patch_d7_positive_barrier_weight=2.0,
            stage_b_native_patch_d7_anchor_active_gap=2.0,
            stage_b_native_patch_d7_anchor_target_gap=2.5,
            stage_b_native_patch_d7_anchor_weight=2.0,
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d7_cfg))

        d8_cfg = _cfg(
            stage_b_native_patch_contract_version=8,
            stage_b_native_patch_objective="d8_state_class_macro_anchor",
            stage_b_native_patch_gate_max_gap=3.0,
            stage_b_native_patch_score_clip=5.0,
            stage_b_native_patch_positive_iou_threshold=0.5,
            stage_b_native_patch_negative_iou_threshold=0.3,
            stage_b_native_patch_d8_weight=1.0,
            stage_b_native_patch_d8_keep_gap=2.75,
            stage_b_native_patch_d8_drop_gap=3.25,
            stage_b_native_patch_d8_drop_active_gap=3.75,
            stage_b_native_patch_d8_temperature=0.25,
            stage_b_native_patch_d8_drop_weight=2.0,
            stage_b_native_patch_d8_critical_keep_weight=1.0,
            stage_b_native_patch_d8_positive_active_gap=2.0,
            stage_b_native_patch_d8_positive_target_gap=2.5,
            stage_b_native_patch_d8_positive_barrier_weight=2.0,
            stage_b_native_patch_d8_anchor_active_gap=2.0,
            stage_b_native_patch_d8_anchor_target_gap=2.5,
            stage_b_native_patch_d8_anchor_negative_weight=1.0,
            stage_b_native_patch_d8_anchor_neutral_weight=2.0,
            stage_b_native_patch_d8_anchor_positive_weight=4.0,
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d8_cfg))
        d9_cfg = _cfg(
            **{
                **vars(d8_cfg),
                "stage_b_native_patch_contract_version": 9,
                "stage_b_native_patch_objective": (
                    "d9_loss_gradient_localized"
                ),
                "stage_b_native_patch_d9_detach_row_stats": True,
            }
        )
        self.assertTrue(ref_eval._validate_native_patch_category_config(d9_cfg))

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d2_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d9_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d6_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d7_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d8_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d3_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d4_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

        outputs, _targets = ref_eval._forward(
            _RecordingNativeRefModel(),
            _batch(),
            torch.device("cpu"),
            amp=False,
            cfg=d5_cfg,
        )
        self.assertIn(ref_eval._NATIVE_PATCH_RANK_SCORE_KEY, outputs)

    def test_ref_rejects_unknown_or_drifted_v2_contract(self):
        exact = {
            "stage_b_native_patch_contract_version": 2,
            "stage_b_native_patch_objective": "d2_gate_aligned",
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
        }
        cases = (
            (
                {"stage_b_native_patch_contract_version": 10},
                "contract version",
            ),
            (
                {"stage_b_native_patch_objective": "d1_raw_margin"},
                "d2_gate_aligned",
            ),
            ({"stage_b_native_patch_gate_max_gap": 2.99}, "max gap 3"),
            ({"stage_b_native_patch_score_clip": 4.99}, "clip 5"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                values = dict(exact)
                values.update(override)
                with self.assertRaisesRegex(ValueError, message):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

    def test_ref_rejects_missing_or_drifted_v3_contract(self):
        exact = {
            "stage_b_native_patch_contract_version": 3,
            "stage_b_native_patch_objective": "d3_critical_winner",
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
        }
        cases = (
            (
                {
                    key: value
                    for key, value in exact.items()
                    if key != "stage_b_native_patch_objective"
                },
                "d3_critical_winner",
            ),
            (
                {
                    key: value
                    for key, value in exact.items()
                    if key != "stage_b_native_patch_gate_max_gap"
                },
                "max gap 3",
            ),
            (
                {
                    key: value
                    for key, value in exact.items()
                    if key != "stage_b_native_patch_score_clip"
                },
                "clip 5",
            ),
            (
                {**exact, "stage_b_native_patch_objective": "d2_gate_aligned"},
                "d3_critical_winner",
            ),
            (
                {**exact, "stage_b_native_patch_gate_max_gap": 2.99},
                "max gap 3",
            ),
            (
                {**exact, "stage_b_native_patch_score_clip": 4.99},
                "clip 5",
            ),
        )
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

    def test_ref_rejects_missing_or_drifted_v4_contract(self):
        exact = {
            "stage_b_native_patch_contract_version": 4,
            "stage_b_native_patch_objective": (
                "d4_positive_protected_critical_winner"
            ),
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d4_critical_weight": 2.0,
            "stage_b_native_patch_d4_critical_keep_weight": 1.0,
            "stage_b_native_patch_d4_positive_keep_weight": 32.0,
        }
        cases = (
            ({**exact, "stage_b_native_patch_objective": "d3_critical_winner"},
             "d4_positive_protected_critical_winner"),
            ({**exact, "stage_b_native_patch_gate_max_gap": 2.99},
             "gate_max_gap=3"),
            ({**exact, "stage_b_native_patch_score_clip": 4.99},
             "score_clip=5"),
            ({**exact, "stage_b_native_patch_d4_critical_weight": 1.99},
             "critical_weight=2"),
            ({**exact, "stage_b_native_patch_d4_critical_keep_weight": 0.99},
             "critical_keep_weight=1"),
            ({**exact, "stage_b_native_patch_d4_positive_keep_weight": 31.0},
             "positive_keep_weight=32"),
        )
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

    def test_ref_rejects_missing_or_drifted_v5_contract(self):
        exact = {
            "stage_b_native_patch_contract_version": 5,
            "stage_b_native_patch_objective": "d5_active_tail_positive_barrier",
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d5_keep_gap": 2.75,
            "stage_b_native_patch_d5_separation_gap": 3.25,
            "stage_b_native_patch_d5_temperature": 0.25,
            "stage_b_native_patch_d5_critical_weight": 2.0,
            "stage_b_native_patch_d5_critical_keep_weight": 1.0,
            "stage_b_native_patch_d5_active_gap": 2.0,
            "stage_b_native_patch_d5_target_gap": 2.5,
            "stage_b_native_patch_d5_positive_barrier_weight": 2.0,
        }
        cases = (
            ({**exact, "stage_b_native_patch_objective": "d3_critical_winner"},
             "d5_active_tail_positive_barrier"),
            ({**exact, "stage_b_native_patch_gate_max_gap": 2.99},
             "gate_max_gap=3"),
            ({**exact, "stage_b_native_patch_score_clip": 4.99},
             "score_clip=5"),
            ({**exact, "stage_b_native_patch_d5_keep_gap": 2.74},
             "keep_gap=2.75"),
            ({**exact, "stage_b_native_patch_d5_separation_gap": 3.24},
             "separation_gap=3.25"),
            ({**exact, "stage_b_native_patch_d5_temperature": 0.2},
             "temperature=0.25"),
            ({**exact, "stage_b_native_patch_d5_critical_weight": 1.0},
             "critical_weight=2"),
            ({**exact, "stage_b_native_patch_d5_critical_keep_weight": 0.0},
             "critical_keep_weight=1"),
            ({**exact, "stage_b_native_patch_d5_active_gap": 2.1},
             "active_gap=2"),
            ({**exact, "stage_b_native_patch_d5_target_gap": 2.6},
             "target_gap=2.5"),
            ({**exact, "stage_b_native_patch_d5_positive_barrier_weight": 3.0},
             "positive_barrier_weight=2"),
        )
        for values, message in cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

    def test_ref_rejects_missing_or_drifted_v6_contract(self):
        exact = {
            "stage_b_native_patch_contract_version": 6,
            "stage_b_native_patch_objective": "d6_direct_deployment_gap",
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d6_weight": 1.0,
            "stage_b_native_patch_d6_keep_gap": 2.75,
            "stage_b_native_patch_d6_drop_gap": 3.25,
            "stage_b_native_patch_d6_drop_active_gap": 3.75,
            "stage_b_native_patch_d6_temperature": 0.25,
            "stage_b_native_patch_d6_drop_weight": 2.0,
            "stage_b_native_patch_d6_critical_keep_weight": 1.0,
            "stage_b_native_patch_d6_positive_active_gap": 2.0,
            "stage_b_native_patch_d6_positive_target_gap": 2.5,
            "stage_b_native_patch_d6_positive_barrier_weight": 2.0,
        }
        for missing in tuple(exact)[1:]:
            with self.subTest(missing=missing):
                values = dict(exact)
                del values[missing]
                with self.assertRaisesRegex(ValueError, "contract v6 requires"):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

        drifted = {
            "stage_b_native_patch_objective": "d5_active_tail_positive_barrier",
            **{
                name: expected + 0.01
                for name, expected in exact.items()
                if name
                not in {
                    "stage_b_native_patch_contract_version",
                    "stage_b_native_patch_objective",
                }
            },
        }
        for name, value in drifted.items():
            with self.subTest(drifted=name):
                values = {**exact, name: value}
                with self.assertRaisesRegex(ValueError, "contract v6 requires"):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

    def test_ref_rejects_missing_or_drifted_v7_contract(self):
        exact = {
            "stage_b_native_patch_contract_version": 7,
            "stage_b_native_patch_objective": "d7_all_state_positive_anchor",
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d7_weight": 1.0,
            "stage_b_native_patch_d7_keep_gap": 2.75,
            "stage_b_native_patch_d7_drop_gap": 3.25,
            "stage_b_native_patch_d7_drop_active_gap": 3.75,
            "stage_b_native_patch_d7_temperature": 0.25,
            "stage_b_native_patch_d7_drop_weight": 2.0,
            "stage_b_native_patch_d7_critical_keep_weight": 1.0,
            "stage_b_native_patch_d7_positive_active_gap": 2.0,
            "stage_b_native_patch_d7_positive_target_gap": 2.5,
            "stage_b_native_patch_d7_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d7_anchor_active_gap": 2.0,
            "stage_b_native_patch_d7_anchor_target_gap": 2.5,
            "stage_b_native_patch_d7_anchor_weight": 2.0,
        }
        for missing in tuple(exact)[1:]:
            with self.subTest(missing=missing):
                values = dict(exact)
                del values[missing]
                with self.assertRaisesRegex(ValueError, "contract v7 requires"):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

        drifted = {
            "stage_b_native_patch_objective": "d6_direct_deployment_gap",
            **{
                name: expected + 0.01
                for name, expected in exact.items()
                if name
                not in {
                    "stage_b_native_patch_contract_version",
                    "stage_b_native_patch_objective",
                }
            },
        }
        for name, value in drifted.items():
            with self.subTest(drifted=name):
                values = {**exact, name: value}
                with self.assertRaisesRegex(ValueError, "contract v7 requires"):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

    def test_ref_rejects_invalid_v8_contract(self):
        exact = {
            "stage_b_native_patch_contract_version": 8,
            "stage_b_native_patch_objective": "d8_state_class_macro_anchor",
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d8_weight": 1.0,
            "stage_b_native_patch_d8_keep_gap": 2.75,
            "stage_b_native_patch_d8_drop_gap": 3.25,
            "stage_b_native_patch_d8_drop_active_gap": 3.75,
            "stage_b_native_patch_d8_temperature": 0.25,
            "stage_b_native_patch_d8_drop_weight": 2.0,
            "stage_b_native_patch_d8_critical_keep_weight": 1.0,
            "stage_b_native_patch_d8_positive_active_gap": 2.0,
            "stage_b_native_patch_d8_positive_target_gap": 2.5,
            "stage_b_native_patch_d8_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d8_anchor_active_gap": 2.0,
            "stage_b_native_patch_d8_anchor_target_gap": 2.5,
            "stage_b_native_patch_d8_anchor_negative_weight": 1.0,
            "stage_b_native_patch_d8_anchor_neutral_weight": 2.0,
            "stage_b_native_patch_d8_anchor_positive_weight": 4.0,
        }
        numeric_names = tuple(exact)[2:]

        for missing in tuple(exact)[1:]:
            with self.subTest(missing=missing):
                values = dict(exact)
                del values[missing]
                with self.assertRaisesRegex(ValueError, "contract v8 requires"):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

        drifted = {"stage_b_native_patch_objective": "d7_all_state_positive_anchor"}
        drifted.update(
            {name: exact[name] + 0.01 for name in numeric_names}
        )
        for name, value in drifted.items():
            with self.subTest(drifted=name):
                values = {**exact, name: value}
                with self.assertRaisesRegex(ValueError, "contract v8 requires"):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

        for name in numeric_names:
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(nonfinite=name, value=nonfinite):
                    values = {**exact, name: nonfinite}
                    with self.assertRaisesRegex(ValueError, "contract v8 requires"):
                        ref_eval._validate_native_patch_category_config(
                            _cfg(**values)
                        )
            with self.subTest(boolean=name):
                values = {**exact, name: True}
                with self.assertRaisesRegex(ValueError, "contract v8 requires"):
                    ref_eval._validate_native_patch_category_config(_cfg(**values))

    def test_ref_uses_full_expression_non_patch_only_and_fixed_gap3(self):
        model = _RecordingNativeRefModel()
        outputs, targets = ref_eval._forward(
            model, _batch(), torch.device("cpu"), amp=False, cfg=_cfg()
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(len(model.calls), 1)
        _samples, call = model.calls[0]
        self.assertEqual(call["captions"], ["small blue car ."])
        self.assertIs(call["patch_only"], False)
        self.assertTrue(call["patch_only_compute_text_logits"])
        self.assertTrue(call["disable_patch_dn"])
        self.assertEqual(tuple(call["patches"].shape), (1, 3, 2, 2))
        self.assertTrue(
            torch.equal(
                call["phrase_to_token_mask"],
                _target()["phrase_to_token_mask"].unsqueeze(0),
            )
        )

        expression_mask = call["phrase_to_token_mask"].any(dim=1)
        expected_base = aggregate_gdino_full_expression_score(
            outputs["pred_logits_text"], expression_mask
        )
        expected_rank, expected_eligible, _expected_patch = (
            apply_native_patch_category_gate(
                expected_base,
                outputs["pred_logits_patch"],
                torch.ones_like(expected_base, dtype=torch.bool),
                max_gap=3.0,
                clip=5.0,
            )
        )
        self.assertTrue(
            torch.equal(outputs[ref_eval._NATIVE_PATCH_BASE_SCORE_KEY], expected_base)
        )
        self.assertTrue(
            torch.equal(outputs[ref_eval._NATIVE_PATCH_RANK_SCORE_KEY], expected_rank)
        )
        self.assertTrue(
            torch.equal(
                outputs[ref_eval._NATIVE_PATCH_ELIGIBLE_MASK_KEY],
                expected_eligible,
            )
        )
        self.assertEqual(int(expected_base.argmax(dim=1).item()), 0)
        self.assertEqual(int(expected_rank.argmax(dim=1).item()), 1)
        self.assertEqual(int(expected_eligible.sum().item()), 1)
        observed = ref_eval._slot_scores(outputs, _cfg(), beta=999.0)
        self.assertTrue(torch.equal(observed[..., 0], expected_rank))

    def test_ref_required_source_and_rank_keys_fail_closed(self):
        with self.assertRaisesRegex(KeyError, "pred_logits_patch"):
            ref_eval._forward(
                _RecordingNativeRefModel(omit_key="pred_logits_patch"),
                _batch(),
                torch.device("cpu"),
                amp=False,
                cfg=_cfg(),
            )
        with self.assertRaisesRegex(
            KeyError, ref_eval._NATIVE_PATCH_RANK_SCORE_KEY
        ):
            ref_eval._slot_scores({}, _cfg(), beta=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            ref_eval._slot_scores(
                {
                    ref_eval._NATIVE_PATCH_RANK_SCORE_KEY: torch.tensor(
                        [[float("nan")]]
                    ),
                    "pred_boxes": torch.zeros((1, 1, 4)),
                },
                _cfg(),
                beta=0.0,
            )

    def test_joint_ref_reuses_authoritative_native_rank(self):
        model = _RecordingNativeRefModel()
        outputs, _targets, query_scores = joint_eval._forward_ref_batch(
            _cfg(), model, _batch(), torch.device("cpu"), amp=False
        )
        self.assertTrue(
            torch.equal(
                query_scores, outputs[ref_eval._NATIVE_PATCH_RANK_SCORE_KEY]
            )
        )
        self.assertIsNone(joint_eval._adapter_ref_score_key(_cfg()))

    def test_joint_request_requires_one_checkpoint_and_ref_only(self):
        valid = types.SimpleNamespace(
            ckpts=["d1.pth"],
            skip_tn=True,
            category_gate_max_gaps=None,
            category_gate_include_base_expert=False,
        )
        self.assertTrue(
            joint_eval._validate_native_patch_category_joint_request(
                valid, _cfg()
            )
        )

        multiple = types.SimpleNamespace(**vars(valid))
        multiple.ckpts = ["d1.pth", "other.pth"]
        with self.assertRaisesRegex(ValueError, "exactly one checkpoint"):
            joint_eval._validate_native_patch_category_joint_request(
                multiple, _cfg()
            )

        with_tn = types.SimpleNamespace(**vars(valid))
        with_tn.skip_tn = False
        with self.assertRaisesRegex(RuntimeError, "pass --skip_tn"):
            joint_eval._validate_native_patch_category_joint_request(
                with_tn, _cfg()
            )

    def test_tn_requires_independent_trained_confidence_key(self):
        outputs = {
            tn_eval._NATIVE_PATCH_CONFIDENCE_SCORE_KEY: torch.tensor(
                [[0.2, 0.8]]
            ),
            ref_eval._NATIVE_PATCH_RANK_SCORE_KEY: torch.tensor(
                [[99.0, -99.0]]
            ),
            "pred_boxes": torch.zeros((1, 2, 4)),
        }
        with self.assertRaisesRegex(RuntimeError, "confidence head"):
            tn_eval._slot_scores(outputs, _cfg(), beta=0.0)
        trained = _cfg(stage_b_native_patch_confidence_trained=True)
        without_confidence = dict(outputs)
        without_confidence.pop(tn_eval._NATIVE_PATCH_CONFIDENCE_SCORE_KEY)
        with self.assertRaisesRegex(
            KeyError, tn_eval._NATIVE_PATCH_CONFIDENCE_SCORE_KEY
        ):
            tn_eval._slot_scores(without_confidence, trained, beta=0.0)
        observed = tn_eval._slot_scores(outputs, trained, beta=0.0)
        self.assertTrue(
            torch.equal(observed, torch.tensor([[[0.2], [0.8]]]))
        )
        self.assertFalse(
            torch.equal(observed[..., 0], outputs[ref_eval._NATIVE_PATCH_RANK_SCORE_KEY])
        )

    def test_future_trained_confidence_forward_is_paired_and_non_patch_only(self):
        model = _RecordingNativeConfidenceModel()
        negative, positive, _targets, valid = tn_eval._forward_pair(
            model, _batch(paired=True), torch.device("cpu"), amp=False
        )
        self.assertEqual(valid.tolist(), [True])
        self.assertEqual(len(model.calls), 1)
        samples, call = model.calls[0]
        self.assertEqual(int(samples.tensors.shape[0]), 2)
        self.assertIs(call["patch_only"], False)
        self.assertEqual(tuple(call["patches"].shape), (2, 3, 2, 2))
        self.assertTrue(torch.equal(call["patches"][0], call["patches"][1]))
        self.assertEqual(
            tuple(positive[tn_eval._NATIVE_PATCH_CONFIDENCE_SCORE_KEY].shape),
            (1, 3),
        )
        self.assertEqual(
            tuple(negative[tn_eval._NATIVE_PATCH_CONFIDENCE_SCORE_KEY].shape),
            (1, 3),
        )


if __name__ == "__main__":
    unittest.main()
