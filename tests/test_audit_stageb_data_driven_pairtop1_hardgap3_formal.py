import copy
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from tools.audit_stageb_data_driven_pairtop1_hardgap3_formal import (
    EXPECTED_SAMPLING_STATE,
    EXPECTED_SCHEDULER,
    PATCH_PARAMETER_NAMES,
    RANK_FIXED_BUFFER,
    RANK_PREFIX,
    PairTop1HardGap3FormalAuditError,
    _audit_code_manifest,
    _audit_criterion_state,
    _audit_model_against_initializer,
    _audit_optimizer_state,
    _audit_scaler_state,
    _audit_terminal_checkpoint,
    _audit_training_log,
    _load_strict_json,
    _publish_receipt,
    canonical_json_sha256,
    stable_file_record,
)


class PairTop1HardGap3FormalAuditTest(unittest.TestCase):
    def _model_and_roles(self):
        initializer = OrderedDict()
        initializer["backbone.weight"] = torch.tensor([1.0, 2.0])
        initializer["patch_encoder.backbone.0.weight"] = torch.tensor([3.0])
        shapes = {
            "patch_encoder.input_proj.0.weight": (2, 2),
            "patch_encoder.input_proj.0.bias": (2,),
            "patch_encoder.input_proj.1.weight": (2, 2),
            "patch_encoder.input_proj.1.bias": (2,),
            "patch_encoder.norm.weight": (2,),
            "patch_encoder.norm.bias": (2,),
            "query_proj_for_patch.weight": (2, 2),
            "query_proj_for_patch.bias": (2,),
            "patch_logit_scale": (),
        }
        for index, name in enumerate(PATCH_PARAMETER_NAMES):
            initializer[name] = torch.full(shapes[name], float(index + 1))
        rank_names = []
        for index in range(39):
            name = f"{RANK_PREFIX}parameter_{index:02d}"
            initializer[name] = torch.full((2,), float(index))
            rank_names.append(name)
        initializer[RANK_FIXED_BUFFER] = torch.arange(16, dtype=torch.float32)
        confidence = (
            "stage_b_data_driven_score_heads.confidence_branch.logit_scale"
        )
        initializer[confidence] = torch.tensor(1.0)
        contract = "stage_b_data_driven_score_heads._contract_version"
        initializer[contract] = torch.tensor(4, dtype=torch.int64)
        trained = OrderedDict(
            (name, value.clone()) for name, value in initializer.items()
        )
        for name in rank_names:
            trained[name].add_(0.25)
        for name in PATCH_PARAMETER_NAMES:
            trained[name].add_(0.5)
        roles = {
            "b58_base": ["backbone.weight"],
            "shared_backbone_alias": ["patch_encoder.backbone.0.weight"],
            "random_patch_projection": list(PATCH_PARAMETER_NAMES),
            "random_relational_rank": rank_names + [RANK_FIXED_BUFFER],
            "random_absolute_confidence": [confidence],
            "score_contract_buffer": [contract],
        }
        return initializer, trained, roles

    def _optimizer(self, model):
        rank_names = [
            name
            for name in model
            if name.startswith(RANK_PREFIX) and name != RANK_FIXED_BUFFER
        ]
        rank_ids = list(range(39))
        patch_ids = list(range(39, 48))
        state = {}
        for name, parameter_id in zip(rank_names, rank_ids):
            state[parameter_id] = {
                "step": torch.tensor(5020.0),
                "exp_avg": torch.full_like(model[name], 0.5),
                "exp_avg_sq": torch.ones_like(model[name]),
            }
        for name, parameter_id in zip(PATCH_PARAMETER_NAMES, patch_ids):
            state[parameter_id] = {
                "step": torch.tensor(5020.0),
                "exp_avg": torch.full_like(model[name], 0.25),
                "exp_avg_sq": torch.ones_like(model[name]),
            }
        common = {
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": True,
        }
        return {
            "param_groups": [
                {
                    "params": rank_ids,
                    "stage_b_data_driven_branch": "rank",
                    "lr": 3e-5,
                    "initial_lr": 3e-5,
                    **common,
                },
                {
                    "params": patch_ids,
                    "stage_b_data_driven_branch": "patch",
                    "lr": 3e-4,
                    "initial_lr": 3e-4,
                    **common,
                },
            ],
            "state": state,
        }

    def test_only_rank_and_patch_may_change_from_initializer(self):
        initializer, trained, roles = self._model_and_roles()
        audit = _audit_model_against_initializer(
            initializer,
            trained,
            roles,
            expected_model_tensor_count=len(trained),
        )
        self.assertEqual(audit["rank_trainable_tensors_changed"], 39)
        self.assertEqual(audit["patch_trainable_tensors_changed"], 9)
        self.assertEqual(
            audit["frozen_tensors_bitwise_equal_to_initializer"], 5
        )

    def test_frozen_drift_and_unchanged_trainable_fail_closed(self):
        initializer, trained, roles = self._model_and_roles()
        trained["backbone.weight"][0] = 99.0
        with self.assertRaisesRegex(
            PairTop1HardGap3FormalAuditError, "frozen b58/confidence"
        ):
            _audit_model_against_initializer(
                initializer,
                trained,
                roles,
                expected_model_tensor_count=len(trained),
            )

        initializer, trained, roles = self._model_and_roles()
        name = PATCH_PARAMETER_NAMES[0]
        trained[name] = initializer[name].clone()
        with self.assertRaisesRegex(
            PairTop1HardGap3FormalAuditError, "not every rank\+patch"
        ):
            _audit_model_against_initializer(
                initializer,
                trained,
                roles,
                expected_model_tensor_count=len(trained),
            )

    def test_optimizer_requires_all_48_states_at_step_5020(self):
        _, trained, _ = self._model_and_roles()
        optimizer = self._optimizer(trained)
        audit = _audit_optimizer_state(optimizer, trained)
        self.assertEqual(audit["optimizer_parameter_states"], 48)
        self.assertEqual(audit["optimizer_state_tensors"], 144)
        self.assertEqual(audit["optimizer_state_step"], 5020)

        optimizer["state"][0]["step"] = torch.tensor(5019.0)
        with self.assertRaisesRegex(
            PairTop1HardGap3FormalAuditError, "must equal 5020"
        ):
            _audit_optimizer_state(optimizer, trained)

    def test_nonfinite_optimizer_and_scaler_drift_fail_closed(self):
        _, trained, _ = self._model_and_roles()
        optimizer = self._optimizer(trained)
        optimizer["state"][0]["exp_avg"][0] = float("nan")
        with self.assertRaises(RuntimeError):
            _audit_optimizer_state(optimizer, trained)

        _audit_scaler_state(
            {
                "scale": 32768.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 1020,
            }
        )
        with self.assertRaisesRegex(
            PairTop1HardGap3FormalAuditError, "zero-skip U5020"
        ):
            _audit_scaler_state(
                {
                    "scale": 16384.0,
                    "growth_factor": 2.0,
                    "backoff_factor": 0.5,
                    "growth_interval": 2000,
                    "_growth_tracker": 1020,
                }
            )

    def test_terminal_epoch_boundary_is_exact(self):
        payload = {
            "model": {},
            "criterion": {},
            "optimizer": {},
            "lr_scheduler": copy.deepcopy(EXPECTED_SCHEDULER),
            "scaler": {},
            "epoch": 0,
            "iteration": 0,
            "optimizer_updates": 5020,
            "epoch_finished": True,
            "rng_state": {"python": (), "numpy": (), "torch": (), "cuda": ()},
            "epoch_rng_state": {
                "python": (),
                "numpy": (),
                "torch": (),
                "cuda": (),
            },
            "args": {},
            "stage_b_data_driven_sampling_state": copy.deepcopy(
                EXPECTED_SAMPLING_STATE
            ),
            "checkpoint_reason": "max_train_iters",
        }
        _audit_terminal_checkpoint(payload)
        payload["iteration"] = 5020
        with self.assertRaisesRegex(
            PairTop1HardGap3FormalAuditError, "terminal checkpoint"
        ):
            _audit_terminal_checkpoint(payload)

    def test_criterion_contract_is_exact_and_finite(self):
        criterion = {
            "fpr_positive_queue": torch.zeros(4096, dtype=torch.float32),
            "fpr_positive_queue_count": torch.tensor(0, dtype=torch.int64),
            "fpr_positive_queue_cursor": torch.tensor(0, dtype=torch.int64),
            "criterion_contract_version": torch.tensor(4, dtype=torch.int64),
            "rank_supervision_contract_id": torch.tensor(4, dtype=torch.int64),
        }
        self.assertEqual(_audit_criterion_state(criterion), 5)
        criterion["rank_supervision_contract_id"] = torch.tensor(
            3, dtype=torch.int64
        )
        with self.assertRaisesRegex(
            PairTop1HardGap3FormalAuditError, "must equal 4"
        ):
            _audit_criterion_state(criterion)

    def test_code_manifest_must_match_saved_digest_and_current_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.py"
            second = Path(directory) / "second.py"
            first.write_text("first\n", encoding="ascii")
            second.write_text("second\n", encoding="ascii")
            records = [
                stable_file_record(first, label="first"),
                stable_file_record(second, label="second"),
            ]
            args = {
                "stage_b_data_driven_training_provenance": {
                    "schema": "pivot.stageb.data_driven_training_provenance/v1",
                    "code_files": records,
                }
            }
            digest = canonical_json_sha256(records)
            audit = _audit_code_manifest(
                args, expected_manifest_sha256=digest, expected_file_count=2
            )
            self.assertTrue(audit["current_files_match_saved_manifest"])
            first.write_text("changed\n", encoding="ascii")
            with self.assertRaisesRegex(
                PairTop1HardGap3FormalAuditError, "changed after launch"
            ):
                _audit_code_manifest(
                    args,
                    expected_manifest_sha256=digest,
                    expected_file_count=2,
                )

    def test_log_requires_zero_skip_anchors_and_one_fresh_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "info.txt"
            command = (
                "INFO | Command: main.py "
                "-c config/ablations/"
                "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2_formal.py "
                "--datasets config/datasets_stageb_data_driven_dd1_official_assignment_three_ref.json "
                "--output_dir outputs/paper_cvpr_v1/"
                "data_driven_dd1_pairtop1_hardgap3_fair_v2_seed42_b64_u5020_v1 "
                "--pretrain_model_path outputs/paper_cvpr_v1/data_driven_initializers/"
                "fair_v2_seed42/checkpoint_dd_a1_relational_v2_init.pth "
                "--seed 42 --max_train_iters 5020 --iter_checkpoint_interval 500 "
                "--num_workers 4 --prefetch_factor 1 --no_persistent_workers "
                "--gradient_accumulation_steps 1 --amp --save_log "
                "--options batch_size=64 epochs=1"
            )
            metric = (
                "INFO | Epoch: [0]  [   0/5020]  "
                "amp_step_skipped: 0.0000 (0.0000)  "
                "optimizer_step: 1.0000 (1.0000)"
            )
            terminal = (
                "INFO | Reached max_train_iters=5020 optimizer updates; saved "
                "epoch-boundary checkpoint after advancing the epoch scheduler."
            )
            path.write_text("\n".join((command, metric, terminal)), encoding="ascii")
            audit = _audit_training_log(path, expected_iterations=[0])
            self.assertEqual(audit["amp_log_anchors"], 1)
            path.write_text(
                "\n".join(
                    (
                        command,
                        metric.replace(
                            "0.0000 (0.0000)", "1.0000 (1.0000)"
                        ),
                        terminal,
                    )
                ),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                PairTop1HardGap3FormalAuditError, "AMP skip"
            ):
                _audit_training_log(path, expected_iterations=[0])

    def test_strict_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"status":"passed","status":"failed"}\n')
            with self.assertRaisesRegex(
                PairTop1HardGap3FormalAuditError, "duplicate key"
            ):
                _load_strict_json(path, label="duplicate evidence")

    def test_receipt_publication_is_fresh_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            _publish_receipt(output, {"status": "passed"})
            self.assertEqual(json.loads(output.read_text()), {"status": "passed"})
            with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                _publish_receipt(output, {"status": "replaced"})
            self.assertEqual(json.loads(output.read_text()), {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
