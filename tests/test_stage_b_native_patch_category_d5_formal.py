import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

import main as training_main
from main import (
    _STAGE_B_NATIVE_PATCH_D1_U500_SCOPE,
    _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE,
    _bind_stage_b_native_patch_runtime_inputs,
    _validate_stage_b_native_patch_d2_resume_checkpoint,
)
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_native_patch_category_d5_u100.py"
)
SMOKE_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_native_patch_category_d5_smoke.py"
)
DATASET_CONFIG = (
    REPO_ROOT
    / "config/datasets_stageb_native_patch_category_d2_train_20260724.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d5_runtime_args(root: Path) -> SimpleNamespace:
    config = root / "config.py"
    dataset = root / "datasets.json"
    initializer = root / "d1_source.pth"
    base_initializer = root / "b58_source.pth"
    output = root / "output"
    config.write_text("value = 1\n", encoding="ascii")
    dataset.write_text('{"train": [], "val": []}\n', encoding="ascii")
    initializer.write_bytes(b"exact-d1-u500-source")
    base_initializer.write_bytes(b"exact-b58-base")
    output.mkdir()
    return SimpleNamespace(
        stage_b_native_patch_category=True,
        stage_b_native_patch_execution_scope=_STAGE_B_NATIVE_PATCH_D5_U100_SCOPE,
        stage_b_native_patch_contract_version=5,
        stage_b_native_patch_objective="d5_active_tail_positive_barrier",
        eval=False,
        resume="",
        config_file=str(config),
        datasets=str(dataset),
        pretrain_model_path=str(initializer),
        output_dir=str(output),
        stage_b_native_patch_formal_config_path=str(config),
        stage_b_native_patch_dataset_config_path=str(dataset),
        stage_b_native_patch_initializer_path=str(initializer),
        stage_b_native_patch_initializer_sha256=_sha256(initializer),
        stage_b_native_patch_d5_base_initializer_path=str(base_initializer),
        stage_b_native_patch_d5_base_initializer_sha256=_sha256(base_initializer),
        stage_b_native_patch_dataset_config_sha256=_sha256(dataset),
        stage_b_native_patch_formal_output_dir=str(output),
        stage_b_native_patch_positive_iou_threshold=0.5,
        stage_b_native_patch_negative_iou_threshold=0.3,
        stage_b_native_patch_d5_weight=1.0,
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
        lr=5e-5,
        stage_b_native_patch_lr=5e-5,
        amp_init_scale=8.0,
        seed=42,
        batch_size=36,
        epochs=250,
        max_train_iters=100,
        iter_checkpoint_interval=50,
        num_workers=8,
        prefetch_factor=1,
        pin_memory=None,
        persistent_workers=False,
        gradient_accumulation_steps=2,
        amp=True,
        world_size=1,
        distributed=False,
        stage_b_native_patch_expected_max_train_iters=100,
        stage_b_native_patch_expected_gradient_accumulation_steps=2,
        stage_b_native_patch_expected_num_workers=8,
        stage_b_native_patch_expected_seed=42,
        stage_b_data_driven_sampling_contract="deterministic_epoch_ledger_v1",
        stage_b_data_driven_sampler_seed=43,
        stage_b_data_driven_loader_seed=1043,
        stage_b_data_driven_required_allocator_env="PYTORCH_CUDA_ALLOC_CONF",
        stage_b_data_driven_required_allocator_conf="expandable_segments:True",
    )


class StageBNativePatchCategoryD5FormalTest(unittest.TestCase):
    def test_formal_and_smoke_configs_are_exact(self) -> None:
        cfg = SLConfig.fromfile(str(FORMAL_CONFIG))._cfg_dict.to_dict()
        expected = {
            "stage_b_native_patch_contract_version": 5,
            "stage_b_native_patch_objective": "d5_active_tail_positive_barrier",
            "stage_b_native_patch_execution_scope": (
                _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE
            ),
            "stage_b_native_patch_initializer_sha256": (
                "ac8b29a8d8a5e5bb8877a7c21769ff08c0b5ca805c522f80549dfc99f55c5dc5"
            ),
            "stage_b_native_patch_d5_base_initializer_sha256": (
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
            "stage_b_native_patch_d5_weight": 1.0,
            "stage_b_native_patch_d5_keep_gap": 2.75,
            "stage_b_native_patch_d5_separation_gap": 3.25,
            "stage_b_native_patch_d5_temperature": 0.25,
            "stage_b_native_patch_d5_critical_weight": 2.0,
            "stage_b_native_patch_d5_critical_keep_weight": 1.0,
            "stage_b_native_patch_d5_active_gap": 2.0,
            "stage_b_native_patch_d5_target_gap": 2.5,
            "stage_b_native_patch_d5_positive_barrier_weight": 2.0,
            "stage_b_data_driven_sampler_seed": 43,
            "stage_b_data_driven_loader_seed": 1043,
            "stage_b_native_patch_expected_max_train_iters": 100,
            "stage_b_native_patch_expected_gradient_accumulation_steps": 2,
            "stage_b_native_patch_expected_num_workers": 8,
            "stage_b_native_patch_expected_seed": 42,
            "batch_size": 36,
            "amp_init_scale": 8.0,
        }
        for key, value in expected.items():
            self.assertEqual(cfg[key], value, key)
        self.assertNotIn(
            "stage_b_native_patch_d5_positive_keep_weight",
            cfg,
        )
        self.assertEqual(
            Path(cfg["stage_b_native_patch_initializer_path"]),
            REPO_ROOT
            / "outputs/paper_cvpr_v1/native_patch_category_d1_seed42_b36a2_lr3e4_u500_v1/checkpoint_iter.pth",
        )
        self.assertEqual(
            Path(cfg["stage_b_native_patch_formal_output_dir"]),
            REPO_ROOT
            / "outputs/paper_cvpr_v1/native_patch_category_d5_seed42_s43_b36a2_lr5e5_amp8_u100_v1",
        )
        self.assertEqual(
            _sha256(DATASET_CONFIG),
            cfg["stage_b_native_patch_dataset_config_sha256"],
        )

        smoke = SLConfig.fromfile(str(SMOKE_CONFIG))._cfg_dict.to_dict()
        self.assertEqual(smoke["stage_b_native_patch_execution_scope"], "")
        for key in (
            "stage_b_native_patch_contract_version",
            "stage_b_native_patch_objective",
            "stage_b_native_patch_initializer_sha256",
            "stage_b_native_patch_dataset_config_sha256",
            "stage_b_native_patch_d5_positive_barrier_weight",
            "stage_b_data_driven_sampler_seed",
            "stage_b_data_driven_loader_seed",
            "amp_init_scale",
        ):
            self.assertEqual(smoke[key], cfg[key], key)

    def test_formal_config_binds_actual_locked_inputs(self) -> None:
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
        args = SimpleNamespace(**values)
        with patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        ):
            _bind_stage_b_native_patch_runtime_inputs(args)
        binding = args.stage_b_native_patch_runtime_binding
        self.assertEqual(binding["initializer"]["sha256"], values[
            "stage_b_native_patch_initializer_sha256"
        ])
        self.assertEqual(
            binding["d5_base_initializer"]["sha256"],
            values["stage_b_native_patch_d5_base_initializer_sha256"],
        )
        self.assertEqual(binding["runtime"]["max_train_iters"], 100)
        self.assertEqual(binding["runtime"]["iter_checkpoint_interval"], 50)

    def test_d5_runtime_binding_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _d5_runtime_args(Path(directory))
            allocator = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            with patch.dict(os.environ, allocator):
                _bind_stage_b_native_patch_runtime_inputs(args)
            binding = args.stage_b_native_patch_runtime_binding
            self.assertEqual(
                binding["schema"],
                "pivot.stageb.native_patch_category_d5_runtime/v1",
            )
            self.assertIn("d5_base_initializer", binding)
            self.assertNotIn("d4_base_initializer", binding)

            for key, bad_value, message in (
                ("stage_b_data_driven_sampler_seed", 42, "formal contract drifted"),
                ("max_train_iters", 101, "runtime drifted"),
                ("iter_checkpoint_interval", 25, "runtime drifted"),
                (
                    "stage_b_native_patch_d5_positive_barrier_weight",
                    1.0,
                    "objective drifted",
                ),
                ("amp_init_scale", 16.0, "objective drifted"),
            ):
                original = getattr(args, key)
                setattr(args, key, bad_value)
                with patch.dict(os.environ, allocator):
                    with self.assertRaisesRegex(RuntimeError, message):
                        _bind_stage_b_native_patch_runtime_inputs(args)
                setattr(args, key, original)

            exact_resume = Path(args.output_dir) / "checkpoint_iter.pth"
            exact_resume.write_bytes(b"resume")
            args.resume = str(exact_resume)
            with patch.dict(os.environ, allocator):
                _bind_stage_b_native_patch_runtime_inputs(args)
            outside = Path(directory) / "outside.pth"
            outside.write_bytes(b"outside")
            args.resume = str(outside)
            with patch.dict(os.environ, allocator):
                with self.assertRaisesRegex(RuntimeError, "exact formal checkpoint"):
                    _bind_stage_b_native_patch_runtime_inputs(args)

    def test_d5_strict_resume_uses_u100_contract_and_d1_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            initializer = Path(directory) / "d1_source.pth"
            trainable_names = (
                "patch_encoder.input_proj.0.weight",
                "patch_encoder.input_proj.0.bias",
                "patch_encoder.input_proj.1.weight",
                "patch_encoder.input_proj.1.bias",
                "patch_encoder.norm.weight",
                "patch_encoder.norm.bias",
                "query_proj_for_patch.weight",
                "query_proj_for_patch.bias",
            )
            source = {"frozen.weight": torch.tensor([7.0])}
            source.update(
                {
                    name: torch.tensor([float(index)])
                    for index, name in enumerate(trainable_names)
                }
            )
            torch.save({"model": source}, initializer)
            args = SimpleNamespace(
                stage_b_native_patch_execution_scope=(
                    _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE
                ),
                stage_b_native_patch_initializer_path=str(initializer),
            )
            resumed = {name: value.clone() for name, value in source.items()}
            for name in trainable_names:
                resumed[name] = resumed[name] + 1.0
            checkpoint = {
                "model": resumed,
                "criterion": {},
                "optimizer": {},
                "lr_scheduler": {},
                "scaler": {},
                "epoch": 0,
                "iteration": 100,
                "optimizer_updates": 50,
                "epoch_finished": False,
                "rng_state": {},
                "epoch_rng_state": {},
                "args": {
                    "stage_b_native_patch_execution_scope": (
                        _STAGE_B_NATIVE_PATCH_D5_U100_SCOPE
                    ),
                    "stage_b_native_patch_initializer_path": str(initializer),
                },
                "checkpoint_reason": "interval",
                "stage_b_data_driven_sampling_state": {},
            }
            _validate_stage_b_native_patch_d2_resume_checkpoint(args, checkpoint)

            checkpoint["iteration"] = 200
            checkpoint["optimizer_updates"] = 100
            with self.assertRaisesRegex(RuntimeError, "exact unfinished"):
                _validate_stage_b_native_patch_d2_resume_checkpoint(args, checkpoint)
            checkpoint["iteration"] = 100
            checkpoint["optimizer_updates"] = 50
            checkpoint["model"] = dict(resumed)
            checkpoint["model"]["frozen.weight"] = torch.tensor([8.0])
            with self.assertRaisesRegex(RuntimeError, "eight-tensor surface"):
                _validate_stage_b_native_patch_d2_resume_checkpoint(args, checkpoint)

    def test_pretrain_route_audits_d1_as_d5_source(self) -> None:
        source = inspect.getsource(training_main.main)
        self.assertIn('"d5_active_tail_positive_barrier"', source)
        self.assertIn("stage_b_native_patch_d5_base_initializer_path", source)
        self.assertIn("audit_d2_source_transition", source)
        self.assertIn("stage_b_native_patch_d5_source_audit", source)
        self.assertIn("pivot.stageb.native_patch_category_d5_source_audit/v1", source)
        self.assertIn("source_scope", source)
        self.assertIn("_STAGE_B_NATIVE_PATCH_D1_U500_SCOPE", source)


if __name__ == "__main__":
    unittest.main()
