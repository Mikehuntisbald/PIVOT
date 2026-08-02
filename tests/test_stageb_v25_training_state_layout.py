import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import main as main_module


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(3, 4)
        self.second = torch.nn.Linear(4, 2)
        self.register_buffer("layout_buffer", torch.zeros(2))


class _TinyCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("queue", torch.zeros(3))
        self.register_buffer("queue_count", torch.zeros((), dtype=torch.long))


class StageBV25TrainingStateLayoutTest(unittest.TestCase):
    expected_parameter_count = 4
    expected_group_count = 2

    def _objects(self):
        model = _TinyModel()
        criterion = _TinyCriterion()
        named = list(model.named_parameters())
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameter for _, parameter in named[:2]],
                    "lr": 2e-5,
                },
                {
                    "params": [parameter for _, parameter in named[2:]],
                    "lr": 5e-4,
                    "stage_b_v15_validity_group": True,
                },
            ],
            lr=2e-5,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=4,
        )
        scaler = main_module._make_grad_scaler(enabled=False)
        return model, criterion, optimizer, scheduler, scaler

    def _layout(self, objects):
        return main_module._build_stage_b_v25_training_state_layout(
            *objects,
            expected_trainable_parameter_count=self.expected_parameter_count,
            expected_optimizer_group_count=self.expected_group_count,
        )

    def _fixture(self, root: Path):
        source = self._objects()
        layout = self._layout(source)
        main_module._write_stage_b_v25_training_state_layout(root, layout)

        model, criterion, optimizer, scheduler, scaler = source
        sum(parameter.sum() for parameter in model.parameters()).backward()
        optimizer.step()
        checkpoint = {
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": 0,
            "iteration": 1,
            "optimizer_updates": 1,
            "epoch_finished": False,
            "rng_state": main_module._capture_rng_state(),
            "epoch_rng_state": main_module._capture_rng_state(),
            "args": {
                "gradient_accumulation_steps": 1,
                "batch_size": 2,
                "amp": False,
                main_module.STAGE_B_V25_TRAINING_STATE_LAYOUT_ARG: layout[
                    "semantic_sha256"
                ],
            },
        }
        checkpoint_path = root / "checkpoint_iter.pth"
        torch.save(checkpoint, checkpoint_path)

        current = self._objects()
        current_layout = self._layout(current)
        self.assertEqual(layout, current_layout)
        return checkpoint, checkpoint_path, current, current_layout

    def _validate(self, checkpoint, checkpoint_path, current, current_layout):
        return main_module._validate_stage_b_v25_resume_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_path,
            current_layout=current_layout,
            optimizer=current[2],
        )

    def test_layout_is_deterministic_and_records_the_ordered_contract(self):
        left = self._layout(self._objects())
        right = self._layout(self._objects())
        self.assertEqual(left, right)
        self.assertEqual(
            left["schema"],
            main_module.STAGE_B_V25_TRAINING_STATE_LAYOUT_SCHEMA,
        )
        self.assertEqual(len(left["trainable_parameters"]), 4)
        self.assertEqual(left["optimizer"]["group_count"], 2)
        self.assertEqual(
            [entry["name"] for entry in left["trainable_parameters"]],
            [
                "first.weight",
                "first.bias",
                "second.weight",
                "second.bias",
            ],
        )
        self.assertEqual(
            left["semantic_sha256"],
            main_module._stage_b_v25_semantic_sha256(left),
        )
        self.assertIn("ordered_state_schema", left["model"])
        self.assertIn("ordered_state_schema", left["criterion"])
        self.assertIn("state_schema", left["lr_scheduler"])
        self.assertIn("state_schema", left["scaler"])
        self.assertIn("state_schema", left["rng_state"])

    def test_production_defaults_seal_94_parameters_and_four_groups(self):
        self.assertEqual(main_module.STAGE_B_V25_TRAINABLE_PARAMETER_COUNT, 94)
        self.assertEqual(main_module.STAGE_B_V25_OPTIMIZER_GROUP_COUNT, 4)
        with self.assertRaisesRegex(RuntimeError, "expected 94, got 4"):
            main_module._build_stage_b_v25_training_state_layout(
                *self._objects()
            )

    def test_valid_complete_checkpoint_matches_sidecar_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            digest = self._validate(*fixture)
            self.assertEqual(digest, fixture[3]["semantic_sha256"])

    def test_missing_or_extra_model_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, current, layout = self._fixture(Path(directory))
            first_key = next(iter(checkpoint["model"]))
            missing = copy.deepcopy(checkpoint)
            missing["model"].pop(first_key)
            with self.assertRaisesRegex(RuntimeError, "model ordered schema drifted"):
                self._validate(missing, path, current, layout)

            extra = copy.deepcopy(checkpoint)
            extra["model"]["unexpected.weight"] = torch.zeros(1)
            with self.assertRaisesRegex(RuntimeError, "model ordered schema drifted"):
                self._validate(extra, path, current, layout)

    def test_missing_or_extra_criterion_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, current, layout = self._fixture(Path(directory))
            first_key = next(iter(checkpoint["criterion"]))
            missing = copy.deepcopy(checkpoint)
            missing["criterion"].pop(first_key)
            with self.assertRaisesRegex(
                RuntimeError, "criterion ordered schema drifted"
            ):
                self._validate(missing, path, current, layout)

            extra = copy.deepcopy(checkpoint)
            extra["criterion"]["unexpected"] = torch.zeros(1)
            with self.assertRaisesRegex(
                RuntimeError, "criterion ordered schema drifted"
            ):
                self._validate(extra, path, current, layout)

    def test_optimizer_group_names_order_and_static_options_are_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, current, layout = self._fixture(Path(directory))
            name_drift = copy.deepcopy(checkpoint)
            names_key = main_module._STAGE_B_V25_OPTIMIZER_PARAMETER_NAMES_KEY
            name_drift["optimizer"]["param_groups"][0][names_key] = list(
                reversed(
                    name_drift["optimizer"]["param_groups"][0][names_key]
                )
            )
            with self.assertRaisesRegex(RuntimeError, "name layout drifted"):
                self._validate(name_drift, path, current, layout)

            order_drift = copy.deepcopy(checkpoint)
            order_drift["optimizer"]["param_groups"][0]["params"] = list(
                reversed(order_drift["optimizer"]["param_groups"][0]["params"])
            )
            with self.assertRaisesRegex(RuntimeError, "parameter order drifted"):
                self._validate(order_drift, path, current, layout)

            option_drift = copy.deepcopy(checkpoint)
            option_drift["optimizer"]["param_groups"][1]["weight_decay"] = 0.5
            with self.assertRaisesRegex(RuntimeError, "static options drifted"):
                self._validate(option_drift, path, current, layout)

    def test_incomplete_optimizer_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, current, layout = self._fixture(Path(directory))
            drifted = copy.deepcopy(checkpoint)
            state_key = next(iter(drifted["optimizer"]["state"]))
            drifted["optimizer"]["state"].pop(state_key)
            with self.assertRaisesRegex(RuntimeError, "does not cover every"):
                self._validate(drifted, path, current, layout)

    def test_missing_scheduler_scaler_or_rng_component_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, current, layout = self._fixture(Path(directory))
            for key in (
                "lr_scheduler",
                "scaler",
                "rng_state",
                "epoch_rng_state",
            ):
                with self.subTest(key=key):
                    drifted = copy.deepcopy(checkpoint)
                    drifted.pop(key)
                    with self.assertRaisesRegex(RuntimeError, "incomplete"):
                        self._validate(drifted, path, current, layout)

    def test_scheduler_scaler_and_rng_schema_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, current, layout = self._fixture(Path(directory))
            cases = []
            scheduler = copy.deepcopy(checkpoint)
            scheduler["lr_scheduler"]["unexpected"] = 1
            cases.append(("scheduler", scheduler))
            scaler = copy.deepcopy(checkpoint)
            scaler["scaler"]["unexpected"] = torch.zeros(1)
            cases.append(("scaler", scaler))
            rng = copy.deepcopy(checkpoint)
            rng["rng_state"].pop("torch")
            cases.append(("rng_state", rng))
            for label, drifted in cases:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, label):
                        self._validate(drifted, path, current, layout)

    def test_checkpoint_digest_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path, current, layout = self._fixture(Path(directory))
            checkpoint["args"][
                main_module.STAGE_B_V25_TRAINING_STATE_LAYOUT_ARG
            ] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "digest drifted"):
                self._validate(checkpoint, path, current, layout)

    def test_sidecar_digest_and_semantic_layout_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, path, current, layout = self._fixture(root)
            sidecar_path = (
                root / main_module.STAGE_B_V25_TRAINING_STATE_LAYOUT_FILENAME
            )
            payload = json.loads(sidecar_path.read_text(encoding="ascii"))
            payload["semantic_sha256"] = "f" * 64
            sidecar_path.write_bytes(
                main_module._stage_b_v25_layout_file_bytes(payload)
            )
            with self.assertRaisesRegex(RuntimeError, "semantic digest"):
                self._validate(checkpoint, path, current, layout)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, path, current, layout = self._fixture(root)
            sidecar_path = (
                root / main_module.STAGE_B_V25_TRAINING_STATE_LAYOUT_FILENAME
            )
            payload = json.loads(sidecar_path.read_text(encoding="ascii"))
            payload["trainable_parameters"][0]["numel"] += 1
            payload["semantic_sha256"] = (
                main_module._stage_b_v25_semantic_sha256(payload)
            )
            sidecar_path.write_bytes(
                main_module._stage_b_v25_layout_file_bytes(payload)
            )
            with self.assertRaisesRegex(RuntimeError, "differs from its sidecar"):
                self._validate(checkpoint, path, current, layout)

    def test_sidecar_is_canonical_and_never_overwritten_with_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(self._objects())
            path = main_module._write_stage_b_v25_training_state_layout(
                root, layout
            )
            self.assertEqual(
                path.read_bytes(),
                main_module._stage_b_v25_layout_file_bytes(layout),
            )
            drifted = copy.deepcopy(layout)
            drifted["trainable_parameters"][0]["numel"] += 1
            drifted["semantic_sha256"] = (
                main_module._stage_b_v25_semantic_sha256(drifted)
            )
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                main_module._write_stage_b_v25_training_state_layout(
                    root, drifted
                )


if __name__ == "__main__":
    unittest.main()
