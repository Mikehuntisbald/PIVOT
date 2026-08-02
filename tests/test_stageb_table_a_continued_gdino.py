import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools import build_stageb_table_a_continued_gdino as builder
from tools import run_stageb_table_a_controls as runner
from util.slconfig import SLConfig


class ContinuedGdinoControlTest(unittest.TestCase):
    def test_tn_conversion_preserves_exact_negative_and_trace_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tn.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "image_id": 9,
                        "file_name": "image.jpg",
                        "sample_id": "tn-9",
                        "sent": "a red bowl",
                        "try_tn": "a blue bowl",
                        "category_name": "bowl",
                        "replace_from": ["red"],
                        "replace_to": ["blue"],
                        "replace_category": ["color"],
                        "tn_scope": "proposal_covered_verified",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "out.jsonl"
            stats = builder._convert_tn(
                {
                    "anno": str(source),
                    "sam3_tn_image_root": str(root),
                },
                output,
            )
            self.assertEqual(stats["rows"], 1)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["grounding"]["caption"], "a blue bowl .")
            self.assertEqual(row["grounding"]["regions"], [])
            self.assertTrue(row["grounding"]["is_negative"])
            self.assertEqual(
                row["grounding"]["tn_records"][0]["positive_phrase"],
                "a red bowl",
            )
            self.assertEqual(
                row["table_a_tn_scope"], "proposal_covered_verified"
            )

    def test_g0c_config_is_pure_gdino_and_disables_flip(self):
        cfg = SLConfig.fromfile(
            "config/ablations/cfg_stageb_table_a_g0c_continued_gdino.py"
        )
        self.assertEqual(cfg.paper_table_a_id, "G0c")
        self.assertFalse(cfg.patch_only)
        self.assertFalse(cfg.stage_b)
        self.assertFalse(cfg.enable_patch_branch)
        self.assertEqual(cfg.data_aug_hflip_prob, 0.0)
        self.assertTrue(cfg.skip_eval)
        self.assertEqual(cfg.amp_init_scale, 512.0)
        self.assertEqual(cfg.amp_max_consecutive_skips, 8)

    def test_g0c_batch_contract_is_effective_global_batch_40(self):
        contract = runner.resolve_batch_contract(
            micro_batch_size=10,
            gradient_accumulation_steps=4,
            effective_global_batch=40,
        )
        self.assertEqual(
            contract,
            {
                "world_size": 1,
                "distributed_environment_scrubbed": True,
                "micro_batch_size_per_rank": 10,
                "gradient_accumulation_steps": 4,
                "effective_global_batch": 40,
            },
        )

    def test_g0c_batch_contract_rejects_nominal_only_match(self):
        with self.assertRaisesRegex(ValueError, "effective global batch mismatch"):
            runner.resolve_batch_contract(
                micro_batch_size=10,
                gradient_accumulation_steps=2,
                effective_global_batch=40,
            )

    def test_g0c_plan_passes_physical_batch_and_accumulation_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"placeholder")
            args = types.SimpleNamespace(
                purpose="probe",
                checkpoint=str(checkpoint),
                batch_size=10,
                gradient_accumulation_steps=4,
                effective_batch_size=40,
                updates=50,
                output_dir=str(root / "output"),
                python=sys.executable,
                seed=17,
                num_workers=2,
                cuda_visible_devices="0",
            )
            with mock.patch.object(
                runner,
                "_validate",
                return_value={"checkpoint_sha256": "a" * 64},
            ):
                plan = runner.build_plan(args)
        self.assertEqual(plan["schema"], runner.PLAN_SCHEMA)
        self.assertEqual(plan["purpose"], "probe")
        self.assertEqual(plan["plan_sha256"], runner._plan_sha256(plan))
        self.assertEqual(plan["matched_contract"]["effective_global_batch"], 40)
        self.assertEqual(plan["matched_contract"]["optimizer_updates"], 50)
        command = plan["command"]
        self.assertEqual(command[command.index("--gradient_accumulation_steps") + 1], "4")
        self.assertIn("batch_size=10", command)
        self.assertEqual(command[command.index("--world_size") + 1], "1")
        self.assertEqual(
            plan["inputs"]["python_runtime"]["path"],
            str(Path(sys.executable).resolve()),
        )

    def test_g0c_plan_rejects_python_that_differs_from_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "other-python"
            other.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            other.chmod(0o755)
            with self.assertRaisesRegex(
                ValueError, "must run under the selected Python"
            ):
                runner._require_current_python_runtime(other)
        self.assertEqual(
            runner._require_current_python_runtime(Path(sys.executable)),
            Path(sys.executable).resolve(),
        )

    def test_single_process_environment_scrubs_ddp_and_slurm_state(self):
        cleaned = runner._single_process_environment(
            {
                "WORLD_SIZE": "8",
                "LOCAL_RANK": "3",
                "RANK": "3",
                "SLURM_PROCID": "3",
                "SLURM_NTASKS": "8",
                "KEEP_ME": "yes",
            }
        )
        self.assertEqual(cleaned, {"KEEP_ME": "yes"})

    def test_formal_plan_requires_exact_seed_batch_updates_root_and_soak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"placeholder")
            args = types.SimpleNamespace(
                purpose="formal",
                checkpoint=str(checkpoint),
                batch_size=10,
                gradient_accumulation_steps=4,
                effective_batch_size=40,
                updates=1000,
                output_dir=str(runner.formal_output_root(17)),
                python=sys.executable,
                seed=17,
                num_workers=2,
                cuda_visible_devices="0",
                soak_seal=str(root / "seal.json"),
            )
            with (
                mock.patch.object(
                    runner,
                    "_validate",
                    return_value={"checkpoint_sha256": "a" * 64},
                ),
                mock.patch.object(
                    runner,
                    "_validate_soak_seal",
                    return_value={
                        "path": str(root / "seal.json"),
                        "sha256": "b" * 64,
                        "payload": {},
                        "plan": {},
                    },
                ),
                mock.patch.object(
                    runner,
                    "_validate_formal_soak_compatibility",
                    return_value={
                        "schema": runner.SOAK_COMPATIBILITY_SCHEMA,
                        "status": "passed",
                        "semantic_sha256": "c" * 64,
                        "allowed_differences": [],
                    },
                ),
            ):
                plan = runner.build_plan(args)
                self.assertEqual(plan["purpose"], "formal")
                self.assertEqual(plan["matched_contract"]["optimizer_updates"], 1000)
                self.assertEqual(plan["soak_seal"]["sha256"], "b" * 64)
                args.updates = 50
                with self.assertRaisesRegex(ValueError, "exactly 1000"):
                    runner.build_plan(args)
                args.updates = 1000
                args.seed = 99
                with self.assertRaisesRegex(ValueError, "seed is not predeclared"):
                    runner.build_plan(args)

    def test_plan_identity_rejects_claim_mutation(self):
        plan = {"schema": runner.PLAN_SCHEMA, "purpose": "probe", "value": 1}
        plan["plan_sha256"] = runner._plan_sha256(plan)
        runner._validate_plan_identity(plan)
        plan["value"] = 2
        with self.assertRaisesRegex(ValueError, "canonical SHA-256"):
            runner._validate_plan_identity(plan)

    def test_legacy_soak_cannot_masquerade_as_new_sealed_soak(self):
        with tempfile.TemporaryDirectory() as tmp:
            seal = Path(tmp) / "legacy.json"
            seal.write_text(
                json.dumps(
                    {
                        "schema": "stageb-table-a-g0c-soak-v0",
                        "status": "PASS",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "v1 sealed-soak schema"):
                runner._validate_soak_seal(seal)

    def test_g0c_source_dependency_tree_is_recursive(self):
        tree = runner._source_dependency_tree()
        relative = {record["relative_path"] for record in tree["records"]}
        for required in (
            "util/box_ops.py",
            "util/logger.py",
            "util/path_compat.py",
            "util/slio.py",
            "util/utils.py",
        ):
            self.assertIn(required, relative)
        self.assertTrue(
            any(
                name.startswith(
                    "models/GroundingDINO/ops/MultiScaleDeformableAttention"
                )
                and name.endswith(".so")
                for name in relative
            )
        )
        for required in (
            "models/GroundingDINO/ops/src/vision.cpp",
            "models/GroundingDINO/ops/src/cuda/ms_deform_attn_cuda.cu",
        ):
            self.assertIn(required, relative)
        self.assertGreater(tree["native_file_count"], 0)
        actual_extension = tree["actual_native_extension"]
        self.assertTrue(actual_extension["path"].endswith(".so"))
        self.assertEqual(
            actual_extension["path"], str(runner._actual_native_extension_path())
        )
        self.assertEqual(tree["file_count"], len(tree["records"]))

    def test_formal_soak_rejects_source_or_input_drift(self):
        command = [
            "/python",
            "/main.py",
            "--output_dir",
            "/soak",
            "--seed",
            "17",
            "--max_train_iters",
            "50",
            "--iter_checkpoint_interval",
            "50",
            "--note",
            "soak",
            "--amp",
        ]
        soak = {
            "purpose": "soak",
            "row_id": "G0c",
            "inputs": {"config": {"path": "/config", "sha256": "a" * 64}},
            "source_dependency_tree": {"sha256": "b" * 64},
            "matched_contract": {
                "seed": 17,
                "optimizer_updates": 50,
                "planned_micro_batches_without_amp_skips": 200,
                "expected_checkpoint_iteration": 200,
                "effective_global_batch": 40,
            },
            "command": command,
            "runtime_evidence_required": True,
            "cuda_visible_devices": "0",
        }
        formal = copy.deepcopy(soak)
        formal["purpose"] = "formal"
        formal["matched_contract"].update(
            seed=42,
            optimizer_updates=1000,
            planned_micro_batches_without_amp_skips=4000,
            expected_checkpoint_iteration=4000,
        )
        for flag, value in (
            ("--output_dir", "/formal"),
            ("--seed", "42"),
            ("--max_train_iters", "1000"),
            ("--iter_checkpoint_interval", "1000"),
            ("--note", "formal"),
        ):
            formal["command"][formal["command"].index(flag) + 1] = value
        runner._validate_formal_soak_compatibility(formal, soak)
        for mutation in ("source", "input"):
            changed = copy.deepcopy(formal)
            if mutation == "source":
                changed["source_dependency_tree"]["sha256"] = "c" * 64
            else:
                changed["inputs"]["config"]["sha256"] = "d" * 64
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError, "not semantically identical"
            ):
                runner._validate_formal_soak_compatibility(changed, soak)

    def test_g0c_plan_rejects_cross_epoch_update_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"placeholder")
            args = types.SimpleNamespace(
                purpose="probe",
                checkpoint=str(checkpoint),
                batch_size=10,
                gradient_accumulation_steps=4,
                effective_batch_size=40,
                updates=8388,
                output_dir=str(root / "output"),
                python=sys.executable,
                seed=17,
                num_workers=2,
                cuda_visible_devices="0",
            )
            with mock.patch.object(
                runner,
                "_validate",
                return_value={"checkpoint_sha256": "a" * 64},
            ):
                with self.assertRaisesRegex(ValueError, "terminate inside epoch 0"):
                    runner.build_plan(args)

    def test_g0c_postflight_distinguishes_micro_batches_from_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_checkpoint = root / "source.pth"
            source_checkpoint.write_bytes(b"source")
            output = root / "output"
            output.mkdir()
            args = types.SimpleNamespace(
                purpose="probe",
                checkpoint=str(source_checkpoint),
                batch_size=10,
                gradient_accumulation_steps=4,
                effective_batch_size=40,
                updates=2,
                output_dir=str(output),
                python=sys.executable,
                seed=17,
                num_workers=2,
                cuda_visible_devices="0",
            )
            with mock.patch.object(
                runner,
                "_validate",
                return_value={
                    "checkpoint_sha256": runner._sha256(source_checkpoint)
                },
            ):
                plan = runner.build_plan(args)
            model_state = {
                f"transformer.layer_{index}": torch.tensor(float(index))
                for index in range(97)
            }
            model_state.update(
                {
                    "bert.encoder.weight": torch.ones(1),
                    "bbox_embed.0.weight": torch.ones(1),
                    "backbone.0.weight": torch.ones(1),
                }
            )
            torch.save(
                {
                    "model": model_state,
                    "criterion": {},
                    "optimizer": {
                        "state": {0: {"step": torch.tensor(2.0)}},
                        "param_groups": [{"params": [0]}],
                    },
                    "lr_scheduler": {
                        "base_lrs": [1e-4],
                        "last_epoch": 0,
                        "_step_count": 1,
                    },
                    "scaler": {
                        "scale": 512.0,
                        "growth_factor": 2.0,
                        "backoff_factor": 0.5,
                        "growth_interval": 2000,
                        "_growth_tracker": 2,
                    },
                    "checkpoint_reason": "max_train_iters",
                    "epoch": 0,
                    "epoch_finished": False,
                    "iteration": 8,
                    "optimizer_updates": 2,
                    "args": {
                        "batch_size": 10,
                        "gradient_accumulation_steps": 4,
                        "max_train_iters": 2,
                        "iter_checkpoint_interval": 2,
                        "seed": 17,
                        "amp": True,
                        "amp_init_scale": 512.0,
                        "amp_max_consecutive_skips": 8,
                        "paper_table_a_id": "G0c",
                        "patch_only": False,
                        "stage_b": False,
                        "enable_patch_branch": False,
                        "data_aug_hflip_prob": 0.0,
                        "skip_eval": True,
                        "world_size": 1,
                        "rank": 0,
                        "local_rank": 0,
                        "distributed": False,
                        "config_file": str(runner.CONFIG),
                        "datasets": str(runner.DATASET),
                        "output_dir": str(output),
                        "pretrain_model_path": str(source_checkpoint),
                        "resume": "",
                    },
                },
                output / "checkpoint_iter.pth",
            )
            result = runner.verify_checkpoint(plan)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["schema"], runner.POSTFLIGHT_SCHEMA)
            self.assertEqual(result["purpose"], "probe")
            self.assertEqual(result["seed"], 17)
            self.assertEqual(result["optimizer_updates"], 2)
            self.assertEqual(result["consumed_micro_batches"], 8)
            self.assertEqual(result["amp_skips_inferred"], 0)

    def test_g0c_postflight_rejects_empty_training_state(self):
        with self.assertRaisesRegex(ValueError, "model state"):
            runner._validate_training_state(
                {
                    "model": {},
                    "criterion": {},
                    "optimizer": {},
                    "lr_scheduler": {},
                    "scaler": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
