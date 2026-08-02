import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from engine import (
    _set_stage_b_native_patch_category_training_mode,
    train_one_epoch,
)
from main import (
    _STAGE_B_NATIVE_PATCH_D1_U500_SCOPE,
    _STAGE_B_NATIVE_PATCH_D2_U500_SCOPE,
    _atomic_torch_save_on_master,
    _bind_stage_b_native_patch_runtime_inputs,
    _freeze_and_audit_stage_b_native_patch_category,
    _stage_b_native_patch_category_optimizer_groups,
    _validate_stage_b_native_patch_d2_resume_checkpoint,
)
from models.GroundingDINO.groundingdino import GroundingDINO, build_groundingdino


class _PatchEncoder(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.norm = nn.LayerNorm(4)


class _NativePatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.bert = nn.Linear(4, 4)
        self.transformer = nn.Linear(4, 4)
        self.patch_encoder = _PatchEncoder(self.backbone)
        self.query_proj_for_patch = nn.Linear(4, 4)
        self.patch_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.stage_b_gdino_score_adapter = None
        self.stage_b_u0_patch_rank_adapter = None
        self.stage_b_data_driven_score_heads = None


class NativePatchCategoryIntegrationTest(unittest.TestCase):
    def test_iteration_checkpoint_publish_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint_iter.pth"
            output.write_bytes(b"previous-complete-checkpoint")
            _atomic_torch_save_on_master({"value": torch.tensor([3])}, output)
            saved = torch.load(output, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(saved["value"], torch.tensor([3])))
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

            previous = output.read_bytes()
            with patch("main.torch.save", side_effect=OSError("disk write failed")):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    _atomic_torch_save_on_master({"value": torch.tensor([4])}, output)
            self.assertEqual(output.read_bytes(), previous)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_formal_runtime_binding_fails_closed_on_cli_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            dataset = root / "datasets.json"
            initializer = root / "initializer.pth"
            config.write_text("value = 1\n", encoding="utf-8")
            dataset.write_text('{"train": [], "val": []}\n', encoding="utf-8")
            initializer.write_bytes(b"initializer")
            dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
            args = SimpleNamespace(
                stage_b_native_patch_category=True,
                stage_b_native_patch_execution_scope=(
                    _STAGE_B_NATIVE_PATCH_D1_U500_SCOPE
                ),
                eval=False,
                resume="",
                config_file=str(config),
                datasets=str(dataset),
                pretrain_model_path=str(initializer),
                output_dir=str(root / "output"),
                stage_b_native_patch_formal_config_path=str(config),
                stage_b_native_patch_dataset_config_path=str(dataset),
                stage_b_native_patch_initializer_path=str(initializer),
                stage_b_native_patch_dataset_config_sha256=dataset_sha,
                stage_b_native_patch_formal_output_dir=str(root / "output"),
                seed=42,
                batch_size=36,
                epochs=250,
                max_train_iters=500,
                iter_checkpoint_interval=500,
                num_workers=8,
                prefetch_factor=1,
                pin_memory=None,
                persistent_workers=False,
                gradient_accumulation_steps=2,
                amp=True,
                world_size=1,
                distributed=False,
                stage_b_data_driven_required_allocator_env=(
                    "PYTORCH_CUDA_ALLOC_CONF"
                ),
                stage_b_data_driven_required_allocator_conf=(
                    "expandable_segments:True"
                ),
                stage_b_native_patch_expected_max_train_iters=500,
                stage_b_native_patch_expected_gradient_accumulation_steps=2,
                stage_b_native_patch_expected_num_workers=8,
                stage_b_native_patch_expected_seed=42,
                stage_b_data_driven_sampling_contract=(
                    "deterministic_epoch_ledger_v1"
                ),
                stage_b_data_driven_sampler_seed=42,
                stage_b_data_driven_loader_seed=1042,
            )

            with patch.dict(
                os.environ,
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            ):
                _bind_stage_b_native_patch_runtime_inputs(args)
            self.assertEqual(
                args.stage_b_native_patch_runtime_binding["scope"],
                _STAGE_B_NATIVE_PATCH_D1_U500_SCOPE,
            )
            args.max_train_iters = 499
            with patch.dict(
                os.environ,
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            ):
                with self.assertRaisesRegex(RuntimeError, "runtime drifted"):
                    _bind_stage_b_native_patch_runtime_inputs(args)

    def test_d2_formal_resume_accepts_only_the_exact_output_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            dataset = root / "datasets.json"
            initializer = root / "d1_source.pth"
            base_initializer = root / "b58_source.pth"
            output = root / "output"
            output.mkdir()
            resume = output / "checkpoint_iter.pth"
            external_resume = root / "external.pth"
            config.write_text("value = 1\n", encoding="utf-8")
            dataset.write_text('{"train": [], "val": []}\n', encoding="utf-8")
            initializer.write_bytes(b"d1-source")
            base_initializer.write_bytes(b"b58-source")
            resume.write_bytes(b"resume")
            external_resume.write_bytes(b"external")
            args = SimpleNamespace(
                stage_b_native_patch_category=True,
                stage_b_native_patch_execution_scope=(
                    _STAGE_B_NATIVE_PATCH_D2_U500_SCOPE
                ),
                stage_b_native_patch_contract_version=2,
                stage_b_native_patch_objective="d2_gate_aligned",
                eval=False,
                resume=str(resume),
                config_file=str(config),
                datasets=str(dataset),
                pretrain_model_path=str(initializer),
                output_dir=str(output),
                stage_b_native_patch_formal_config_path=str(config),
                stage_b_native_patch_dataset_config_path=str(dataset),
                stage_b_native_patch_initializer_path=str(initializer),
                stage_b_native_patch_initializer_sha256=hashlib.sha256(
                    initializer.read_bytes()
                ).hexdigest(),
                stage_b_native_patch_d2_base_initializer_path=str(base_initializer),
                stage_b_native_patch_d2_base_initializer_sha256=hashlib.sha256(
                    base_initializer.read_bytes()
                ).hexdigest(),
                stage_b_native_patch_dataset_config_sha256=hashlib.sha256(
                    dataset.read_bytes()
                ).hexdigest(),
                stage_b_native_patch_formal_output_dir=str(output),
                lr=1e-4,
                stage_b_native_patch_lr=1e-4,
                stage_b_native_patch_positive_iou_threshold=0.5,
                stage_b_native_patch_negative_iou_threshold=0.3,
                stage_b_native_patch_d2_weight=1.0,
                stage_b_native_patch_gate_max_gap=3.0,
                stage_b_native_patch_score_clip=5.0,
                stage_b_native_patch_d2_keep_gap=2.75,
                stage_b_native_patch_d2_drop_gap=3.25,
                stage_b_native_patch_d2_temperature=0.25,
                stage_b_native_patch_d2_native_hard_negatives=16,
                stage_b_native_patch_d2_patch_hard_negatives=4,
                stage_b_native_patch_d2_keep_weight=2.0,
                stage_b_native_patch_d2_drop_weight=1.0,
                stage_b_native_patch_d2_coverage_weight=0.25,
                seed=42,
                batch_size=36,
                epochs=250,
                max_train_iters=500,
                iter_checkpoint_interval=100,
                num_workers=8,
                prefetch_factor=1,
                pin_memory=None,
                persistent_workers=False,
                gradient_accumulation_steps=2,
                amp=True,
                world_size=1,
                distributed=False,
                stage_b_data_driven_required_allocator_env=(
                    "PYTORCH_CUDA_ALLOC_CONF"
                ),
                stage_b_data_driven_required_allocator_conf=(
                    "expandable_segments:True"
                ),
                stage_b_native_patch_expected_max_train_iters=500,
                stage_b_native_patch_expected_gradient_accumulation_steps=2,
                stage_b_native_patch_expected_num_workers=8,
                stage_b_native_patch_expected_seed=42,
                stage_b_data_driven_sampling_contract=(
                    "deterministic_epoch_ledger_v1"
                ),
                stage_b_data_driven_sampler_seed=42,
                stage_b_data_driven_loader_seed=1042,
            )
            with patch.dict(
                os.environ,
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            ):
                _bind_stage_b_native_patch_runtime_inputs(args)
            self.assertEqual(
                args.stage_b_native_patch_runtime_binding["runtime"][
                    "iter_checkpoint_interval"
                ],
                100,
            )

            args.resume = str(external_resume)
            with patch.dict(
                os.environ,
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            ):
                with self.assertRaisesRegex(RuntimeError, "exact formal checkpoint"):
                    _bind_stage_b_native_patch_runtime_inputs(args)

    def test_d2_resume_checkpoint_requires_exact_eight_tensor_lineage(self) -> None:
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
                {name: torch.tensor([float(index)]) for index, name in enumerate(trainable_names)}
            )
            torch.save({"model": source}, initializer)
            args = SimpleNamespace(
                stage_b_native_patch_initializer_path=str(initializer)
            )
            saved_args = {"stage_b_native_patch_initializer_path": str(initializer)}
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
                "iteration": 200,
                "optimizer_updates": 100,
                "epoch_finished": False,
                "rng_state": {},
                "epoch_rng_state": {},
                "args": saved_args,
                "checkpoint_reason": "interval",
                "stage_b_data_driven_sampling_state": {},
            }
            _validate_stage_b_native_patch_d2_resume_checkpoint(args, checkpoint)

            checkpoint["model"] = dict(resumed)
            checkpoint["model"]["frozen.weight"] = torch.tensor([8.0])
            with self.assertRaisesRegex(RuntimeError, "eight-tensor surface"):
                _validate_stage_b_native_patch_d2_resume_checkpoint(args, checkpoint)

    def test_freeze_surface_is_exactly_eight_tensors(self) -> None:
        model = _NativePatchModel()
        trainable_count = _freeze_and_audit_stage_b_native_patch_category(model)

        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(len(trainable), 8)
        self.assertEqual(trainable_count, sum(p.numel() for p in trainable.values()))
        self.assertTrue(
            all(
                name.startswith(
                    (
                        "patch_encoder.input_proj.",
                        "patch_encoder.norm.",
                        "query_proj_for_patch.",
                    )
                )
                for name in trainable
            )
        )
        self.assertFalse(model.patch_logit_scale.requires_grad)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))

    def test_optimizer_owns_only_the_audited_surface(self) -> None:
        model = _NativePatchModel()
        _freeze_and_audit_stage_b_native_patch_category(model)
        groups = _stage_b_native_patch_category_optimizer_groups(model, lr=3e-4)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["stage_b_native_patch_branch"], "patch_category")
        self.assertEqual(groups[0]["lr"], 3e-4)
        self.assertEqual(len(groups[0]["params"]), 8)
        self.assertEqual(
            {id(parameter) for parameter in groups[0]["params"]},
            {
                id(parameter)
                for parameter in model.parameters()
                if parameter.requires_grad
            },
        )

    def test_training_mode_keeps_frozen_features_deterministic(self) -> None:
        model = _NativePatchModel()
        _freeze_and_audit_stage_b_native_patch_category(model)
        model.train()

        _set_stage_b_native_patch_category_training_mode(model)

        self.assertFalse(model.training)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.bert.training)
        self.assertFalse(model.transformer.training)
        self.assertTrue(model.patch_encoder.input_proj.training)
        self.assertTrue(model.patch_encoder.norm.training)
        self.assertTrue(model.query_proj_for_patch.training)

    def test_build_and_engine_have_fail_closed_native_routes(self) -> None:
        build_source = inspect.getsource(build_groundingdino)
        forward_source = inspect.getsource(GroundingDINO.forward)
        train_source = inspect.getsource(train_one_epoch)

        self.assertIn("stage_b_native_patch_category", build_source)
        self.assertIn(
            "stage_b_native_patch_category requires patch_gate_with_text=False",
            build_source,
        )
        self.assertIn(
            "stage_b_native_patch_category requires enable_patch_branch=True",
            build_source,
        )
        self.assertIn(
            "native patch-category routing must not rewrite full-text captions",
            train_source,
        )
        self.assertIn(
            "and not self.stage_b_native_patch_category",
            forward_source,
        )
        self.assertIn("StageBNativePatchCategoryCriterion", build_source)


if __name__ == "__main__":
    unittest.main()
