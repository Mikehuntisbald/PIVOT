import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main as training_main
from main import (
    _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE,
    _bind_stage_b_native_patch_runtime_inputs,
    _validate_stage_b_native_patch_d2_resume_checkpoint,
)
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_native_patch_category_d9_u100.py"
)
SMOKE_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_native_patch_category_d9_smoke.py"
)
DATASET_CONFIG = (
    REPO_ROOT
    / "config/datasets_stageb_native_patch_category_d2_train_20260724.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_args() -> SimpleNamespace:
    values = SLConfig.fromfile(str(FORMAL_CONFIG))._cfg_dict.to_dict()
    values.update(
        {
            "eval": False,
            "resume": "",
            "config_file": str(FORMAL_CONFIG),
            "datasets": str(DATASET_CONFIG),
            "pretrain_model_path": values[
                "stage_b_native_patch_initializer_path"
            ],
            "output_dir": values[
                "stage_b_native_patch_formal_output_dir"
            ],
            "seed": 42,
            "max_train_iters": 100,
            "iter_checkpoint_interval": 50,
            "num_workers": 8,
            "prefetch_factor": 1,
            "pin_memory": None,
            "persistent_workers": False,
            "gradient_accumulation_steps": 2,
            "amp": True,
            "world_size": 1,
            "distributed": False,
        }
    )
    return SimpleNamespace(**values)


class StageBNativePatchCategoryD9FormalTest(unittest.TestCase):
    def test_formal_and_smoke_configs_freeze_the_single_variable(self):
        cfg = SLConfig.fromfile(str(FORMAL_CONFIG))._cfg_dict.to_dict()
        expected = {
            "stage_b_native_patch_contract_version": 9,
            "stage_b_native_patch_objective": "d9_loss_gradient_localized",
            "stage_b_native_patch_d9_detach_row_stats": True,
            "stage_b_native_patch_execution_scope": (
                _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE
            ),
            "stage_b_native_patch_initializer_sha256": (
                "ac8b29a8d8a5e5bb8877a7c21769ff08c0b5ca805c522f80549dfc99f55c5dc5"
            ),
            "stage_b_native_patch_d9_base_initializer_sha256": (
                "addec47338c2e36a3121d999370349d2351535f6ff7334729424aeb1bcd880b4"
            ),
            "stage_b_native_patch_dataset_config_sha256": (
                "f8d1eda36b663bfdba43e986ccef060dd461e0ae3400ae7813c8c7ba6d32a398"
            ),
            "stage_b_native_patch_dataset_receipt_sha256": (
                "96d11562d1a0f064bdf5de676b48409967b35b31441777b7dac662b792d7eb94"
            ),
            "stage_b_native_patch_dataset_receipt_canonical_sha256": (
                "b791bbde32f891ac5e4c30b5e511c4cd06433ba1980dc4442f05184100d9dca1"
            ),
            "stage_b_native_patch_lr": 5e-5,
            "lr": 5e-5,
            "stage_b_native_patch_d8_weight": 1.0,
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
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
            "stage_b_data_driven_sampler_seed": 43,
            "stage_b_data_driven_loader_seed": 1043,
            "stage_b_native_patch_expected_max_train_iters": 100,
            "stage_b_native_patch_expected_gradient_accumulation_steps": 2,
            "stage_b_native_patch_expected_num_workers": 8,
            "stage_b_native_patch_expected_seed": 42,
            "batch_size": 36,
            "amp_init_scale": 8.0,
            "save_checkpoint_interval": 100,
        }
        for key, value in expected.items():
            self.assertEqual(cfg[key], value, key)
        self.assertEqual(
            Path(cfg["stage_b_native_patch_initializer_path"]),
            REPO_ROOT
            / "outputs/paper_cvpr_v1/native_patch_category_d1_seed42_b36a2_lr3e4_u500_v1/checkpoint_iter.pth",
        )
        self.assertEqual(
            Path(cfg["stage_b_native_patch_formal_output_dir"]),
            REPO_ROOT
            / "outputs/paper_cvpr_v1/native_patch_category_d9_seed42_s43_b36a2_lr5e5_amp8_u100_v1",
        )
        self.assertEqual(
            _sha256(DATASET_CONFIG),
            cfg["stage_b_native_patch_dataset_config_sha256"],
        )

        smoke = SLConfig.fromfile(str(SMOKE_CONFIG))._cfg_dict.to_dict()
        self.assertEqual(smoke["stage_b_native_patch_execution_scope"], "")
        for key in expected:
            if key != "stage_b_native_patch_execution_scope":
                self.assertEqual(smoke[key], cfg[key], key)

    def test_actual_formal_inputs_bind_to_d9_runtime_receipt(self):
        args = _formal_args()
        with patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ):
            _bind_stage_b_native_patch_runtime_inputs(args)
        binding = args.stage_b_native_patch_runtime_binding
        self.assertEqual(
            binding["schema"],
            "pivot.stageb.native_patch_category_d9_runtime/v1",
        )
        self.assertEqual(binding["scope"], _STAGE_B_NATIVE_PATCH_D9_U100_SCOPE)
        self.assertEqual(
            binding["initializer"]["sha256"],
            args.stage_b_native_patch_initializer_sha256,
        )
        self.assertEqual(
            binding["d9_base_initializer"]["sha256"],
            args.stage_b_native_patch_d9_base_initializer_sha256,
        )
        self.assertEqual(binding["runtime"]["max_train_iters"], 100)

    def test_runtime_fails_closed_on_the_only_d9_variable(self):
        args = _formal_args()
        args.stage_b_native_patch_d9_detach_row_stats = False
        with patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ):
            with self.assertRaisesRegex(
                RuntimeError, "loss-gradient-localized objective drifted"
            ):
                _bind_stage_b_native_patch_runtime_inputs(args)

    def test_resume_and_source_routes_include_d9_contract(self):
        resume_source = inspect.getsource(
            _validate_stage_b_native_patch_d2_resume_checkpoint
        )
        self.assertIn("_STAGE_B_NATIVE_PATCH_D9_U100_SCOPE", resume_source)
        self.assertIn(
            '"stage_b_native_patch_d9_detach_row_stats"', resume_source
        )
        self.assertIn(
            '"stage_b_native_patch_d9_base_initializer_path"', resume_source
        )
        main_source = inspect.getsource(training_main)
        self.assertIn('"d9_loss_gradient_localized"', main_source)
        self.assertIn(
            "pivot.stageb.native_patch_category_d9_source_audit/v1",
            main_source,
        )
        self.assertIn(
            "args.stage_b_native_patch_d9_source_audit", main_source
        )


if __name__ == "__main__":
    unittest.main()
