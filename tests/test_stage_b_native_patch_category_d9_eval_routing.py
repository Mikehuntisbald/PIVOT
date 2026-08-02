from types import SimpleNamespace
import unittest

from tools import eval_refcoco_stageb as ref_eval


def _cfg(**overrides):
    values = {
        "stage_b_native_patch_category": True,
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
    return SimpleNamespace(**values)


def _exact_d9():
    return {
        "stage_b_native_patch_contract_version": 9,
        "stage_b_native_patch_objective": "d9_loss_gradient_localized",
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
        "stage_b_native_patch_d9_detach_row_stats": True,
    }


class StageBNativePatchCategoryD9EvalRoutingTest(unittest.TestCase):
    def test_exact_d9_is_accepted(self):
        self.assertTrue(
            ref_eval._validate_native_patch_category_config(
                _cfg(**_exact_d9())
            )
        )

    def test_missing_or_drifted_d9_value_is_rejected(self):
        exact = _exact_d9()
        for missing in tuple(exact)[1:]:
            with self.subTest(missing=missing):
                values = dict(exact)
                del values[missing]
                with self.assertRaisesRegex(ValueError, "contract v9 requires"):
                    ref_eval._validate_native_patch_category_config(
                        _cfg(**values)
                    )

        numeric_names = tuple(exact)[2:-1]
        for name in numeric_names:
            for value in (
                exact[name] + 0.01,
                float("nan"),
                float("inf"),
                -float("inf"),
                True,
            ):
                with self.subTest(name=name, value=value):
                    values = {**exact, name: value}
                    with self.assertRaisesRegex(
                        ValueError, "contract v9 requires"
                    ):
                        ref_eval._validate_native_patch_category_config(
                            _cfg(**values)
                        )

        for name, value in (
            ("stage_b_native_patch_objective", "d8_state_class_macro_anchor"),
            ("stage_b_native_patch_d9_detach_row_stats", False),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "contract v9 requires"):
                    ref_eval._validate_native_patch_category_config(
                        _cfg(**{**exact, name: value})
                    )

    def test_contract_v10_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "contract version"):
            ref_eval._validate_native_patch_category_config(
                _cfg(stage_b_native_patch_contract_version=10)
            )


if __name__ == "__main__":
    unittest.main()
