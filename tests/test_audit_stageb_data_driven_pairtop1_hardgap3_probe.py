import copy
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from tools.audit_stageb_data_driven_pairtop1_hardgap3_probe import (
    EXPECTED_ASSIGNMENT_RECEIPT_SHA256,
    EXPECTED_DATASET_CONFIG_SHA256,
    EXPECTED_INITIALIZER_PAIR_SHA256,
    EXPECTED_INITIALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_RANK_SUPERVISION,
    EXPECTED_ROWS,
    EXPECTED_SCOPE,
    EXPECTED_VALID_ROWS,
    HARDGAP3_ROLE,
    HARDGAP3_VARIANT,
    PAIRTOP1_ROLE,
    PAIRTOP1_VARIANT,
    PATCH_PARAMETER_NAMES,
    RANK_FIXED_BUFFER,
    RANK_PREFIX,
    PairTop1HardGap3ProbeAuditError,
    _audit_criterion_state,
    _atomic_publish_fresh_json,
    _audit_finite_tree,
    _audit_model_pair,
    _audit_optimizer_pair,
    _audit_scaler_state,
    _validate_scalar_training_contract,
)


class PairTop1HardGap3ProbeAuditTest(unittest.TestCase):
    def test_receipt_publication_is_fresh_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            _atomic_publish_fresh_json(output, {"status": "passed"})
            self.assertEqual(json.loads(output.read_text()), {"status": "passed"})
            with self.assertRaisesRegex(FileExistsError, "must be fresh"):
                _atomic_publish_fresh_json(output, {"status": "replaced"})
            self.assertEqual(json.loads(output.read_text()), {"status": "passed"})

    def _model_pair(self):
        pair = OrderedDict()
        pair["backbone.weight"] = torch.tensor([1.0, 2.0])
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
            pair[name] = torch.full(shapes[name], float(index + 1))
        for index in range(39):
            pair[f"{RANK_PREFIX}parameter_{index:02d}"] = torch.full(
                (2,), float(index)
            )
        pair[RANK_FIXED_BUFFER] = torch.arange(16, dtype=torch.float32)
        hard = OrderedDict((name, value.clone()) for name, value in pair.items())
        for name in pair:
            if name.startswith(RANK_PREFIX) and name != RANK_FIXED_BUFFER:
                hard[name].add_(0.25)
        return pair, hard

    def _optimizer_pair(self, pair_model, hard_model):
        rank_ids = list(range(39))
        patch_ids = list(range(39, 48))

        def optimizer(model, *, rank_delta):
            rank_names = [
                name
                for name in model
                if name.startswith(RANK_PREFIX) and name != RANK_FIXED_BUFFER
            ]
            state = {}
            for name, parameter_id in zip(rank_names, rank_ids):
                state[parameter_id] = {
                    "step": torch.tensor(50.0),
                    "exp_avg": torch.full_like(model[name], rank_delta),
                    "exp_avg_sq": torch.ones_like(model[name]),
                }
            for name, parameter_id in zip(PATCH_PARAMETER_NAMES, patch_ids):
                state[parameter_id] = {
                    "step": torch.tensor(50.0),
                    "exp_avg": torch.zeros_like(model[name]),
                    "exp_avg_sq": torch.ones_like(model[name]),
                }
            return {
                "param_groups": [
                    {
                        "params": rank_ids,
                        "stage_b_data_driven_branch": "rank",
                        "lr": 3e-5,
                        "initial_lr": 3e-5,
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
                    },
                    {
                        "params": patch_ids,
                        "stage_b_data_driven_branch": "patch",
                        "lr": 3e-4,
                        "initial_lr": 3e-4,
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
                    },
                ],
                "state": state,
            }

        return optimizer(pair_model, rank_delta=0.0), optimizer(
            hard_model, rank_delta=0.5
        )

    def _saved_args(self, role):
        hardgap = role == HARDGAP3_ROLE
        args = {
            "stage_b_data_driven_score": True,
            "stage_b_data_driven_experiment_id": "DD1",
            "stage_b_data_driven_variant_id": (
                HARDGAP3_VARIANT if hardgap else PAIRTOP1_VARIANT
            ),
            "stage_b_data_driven_train_mode": "rank_patch_only",
            "stage_b_data_driven_category_complete": True,
            "stage_b_data_driven_confidence_trained": False,
            "stage_b_data_driven_rank_supervision": EXPECTED_RANK_SUPERVISION,
            "stage_b_data_driven_rank_weight": 0.0,
            "stage_b_data_driven_assignment_weight": 1.0,
            "stage_b_data_driven_deployment_weight": 1.0 if hardgap else 0.0,
            "stage_b_data_driven_patch_weight": 1.0,
            "stage_b_data_driven_strict_sample_identity": True,
            "stage_b_data_driven_category_gate": False,
            "stage_b_data_driven_category_gate_max_gap": 3.0,
            "stage_b_data_driven_patch_score_clip": 5.0,
            "stage_b_data_driven_assignment_dataset_scope": EXPECTED_SCOPE,
            "stage_b_data_driven_assignment_expected_rows": EXPECTED_ROWS,
            "stage_b_data_driven_assignment_expected_valid_rows": (
                EXPECTED_VALID_ROWS
            ),
            "stage_b_data_driven_assignment_dataset_config_sha256": (
                EXPECTED_DATASET_CONFIG_SHA256
            ),
            "stage_b_data_driven_assignment_receipt_sha256": (
                EXPECTED_ASSIGNMENT_RECEIPT_SHA256
            ),
            "stage_b_data_driven_assignment_manifest_sha256": (
                EXPECTED_MANIFEST_SHA256
            ),
            "stage_b_data_driven_base_initializer_sha256": (
                EXPECTED_INITIALIZER_SHA256
            ),
            "stage_b_data_driven_initializer_pair_receipt_sha256": (
                EXPECTED_INITIALIZER_PAIR_SHA256
            ),
            "stage_b_data_driven_no_teacher_contract": (
                "b58_only_random_independent_heads_v1"
            ),
            "stage_b_gdino_score_adapter": False,
            "stage_b_u0_patch_rank": False,
            "stage_b_v7": False,
            "stage_b_v11_fixed_text": False,
            "stage_b_legacy_global_gate": False,
            "seed": 42,
            "start_epoch": 0,
            "resume": "",
            "max_train_iters": 50,
            "iter_checkpoint_interval": 50,
            "num_workers": 4,
            "prefetch_factor": 1,
            "persistent_workers": False,
            "gradient_accumulation_steps": 1,
            "amp": True,
            "amp_init_scale": 8192.0,
            "save_log": True,
            "world_size": 1,
            "distributed": False,
            "batch_size": 64,
            "epochs": 1,
            "fix_size": True,
            "strong_aug": False,
            "data_aug_hflip_prob": 0.0,
            "config_file": (
                "config/ablations/"
                "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2_probe_u50.py"
                if hardgap
                else "config/ablations/"
                "cfg_stageb_data_driven_dd1_pairtop1_fair_v2.py"
            ),
            "pretrain_model_path": (
                "outputs/paper_cvpr_v1/data_driven_initializers/fair_v2_seed42/"
                "checkpoint_dd_a1_relational_v2_init.pth"
            ),
        }
        if hardgap:
            args.update(
                {
                    "stage_b_data_driven_probe_scope": (
                        "full_official_assignment_u50_v1"
                    ),
                    "stage_b_data_driven_probe_fresh_start": True,
                    "stage_b_data_driven_probe_expected_max_train_iters": 50,
                    "stage_b_data_driven_probe_expected_iter_checkpoint_interval": 50,
                    "stage_b_data_driven_probe_expected_num_workers": 4,
                    "stage_b_data_driven_probe_expected_prefetch_factor": 1,
                    "stage_b_data_driven_probe_expected_gradient_accumulation_steps": 1,
                    "stage_b_data_driven_probe_expected_amp": True,
                    "stage_b_data_driven_probe_expected_save_log": True,
                    "stage_b_data_driven_probe_expected_world_size": 1,
                    "stage_b_data_driven_probe_expected_distributed": False,
                }
            )
        return args

    def test_only_rank_model_and_optimizer_state_may_change(self):
        pair_model, hard_model = self._model_pair()
        model_audit = _audit_model_pair(
            pair_model,
            hard_model,
            expected_model_tensor_count=len(pair_model),
        )
        self.assertEqual(model_audit["rank_trainable_tensors_changed"], 39)
        pair_optimizer, hard_optimizer = self._optimizer_pair(
            pair_model, hard_model
        )
        optimizer_audit = _audit_optimizer_pair(
            pair_optimizer,
            hard_optimizer,
            pair_model,
            hard_model,
        )
        self.assertEqual(optimizer_audit["rank_optimizer_states_changed"], 39)
        self.assertEqual(optimizer_audit["patch_optimizer_state_count"], 9)

    def test_non_rank_or_patch_optimizer_drift_fails_closed(self):
        pair_model, hard_model = self._model_pair()
        hard_model["backbone.weight"][0] = 99.0
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "non-rank model tensors differ"
        ):
            _audit_model_pair(
                pair_model,
                hard_model,
                expected_model_tensor_count=len(pair_model),
            )

        pair_model, hard_model = self._model_pair()
        pair_optimizer, hard_optimizer = self._optimizer_pair(
            pair_model, hard_model
        )
        hard_optimizer["state"][39]["exp_avg"].add_(1.0)
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "patch optimizer states differ"
        ):
            _audit_optimizer_pair(
                pair_optimizer,
                hard_optimizer,
                pair_model,
                hard_model,
            )

    def test_optimizer_branch_or_step_drift_fails_closed(self):
        pair_model, hard_model = self._model_pair()
        pair_optimizer, hard_optimizer = self._optimizer_pair(
            pair_model, hard_model
        )
        hard_optimizer["param_groups"][0]["stage_b_data_driven_branch"] = "patch"
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "branch labels/order drifted"
        ):
            _audit_optimizer_pair(
                pair_optimizer,
                hard_optimizer,
                pair_model,
                hard_model,
            )

        pair_optimizer, hard_optimizer = self._optimizer_pair(
            pair_model, hard_model
        )
        hard_optimizer["state"][0]["step"] = torch.tensor(49.0)
        with self.assertRaisesRegex(PairTop1HardGap3ProbeAuditError, "must equal 50"):
            _audit_optimizer_pair(
                pair_optimizer,
                hard_optimizer,
                pair_model,
                hard_model,
            )

    def test_saved_contract_distinguishes_only_deployment_weight(self):
        pair_args = self._saved_args(PAIRTOP1_ROLE)
        hard_args = self._saved_args(HARDGAP3_ROLE)
        _validate_scalar_training_contract(pair_args, role=PAIRTOP1_ROLE)
        _validate_scalar_training_contract(hard_args, role=HARDGAP3_ROLE)
        hard_args["stage_b_data_driven_deployment_weight"] = 0.0
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "exact contract drifted"
        ):
            _validate_scalar_training_contract(hard_args, role=HARDGAP3_ROLE)

    def test_nonfinite_criterion_or_scaler_fails_closed(self):
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "non-finite tensor"
        ):
            _audit_finite_tree(
                {"criterion_contract_version": torch.tensor(float("nan"))},
                label="criterion",
            )
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "non-finite float"
        ):
            _audit_finite_tree(
                {"scale": float("inf")},
                label="scaler",
            )

    def test_exact_criterion_and_zero_skip_scaler_contracts(self):
        criterion = {
            "fpr_positive_queue": torch.zeros(4096, dtype=torch.float32),
            "fpr_positive_queue_count": torch.tensor(0, dtype=torch.int64),
            "fpr_positive_queue_cursor": torch.tensor(0, dtype=torch.int64),
            "criterion_contract_version": torch.tensor(4, dtype=torch.int64),
            "rank_supervision_contract_id": torch.tensor(4, dtype=torch.int64),
        }
        scaler = {
            "scale": 8192.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 50,
        }
        _audit_criterion_state(criterion, label="probe")
        _audit_scaler_state(scaler, label="probe")

        criterion["fpr_positive_queue_count"] = torch.tensor(
            1, dtype=torch.int64
        )
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "must equal 0"
        ):
            _audit_criterion_state(criterion, label="probe")

        scaler["_growth_tracker"] = 49
        with self.assertRaisesRegex(
            PairTop1HardGap3ProbeAuditError, "zero-skip evidence"
        ):
            _audit_scaler_state(scaler, label="probe")


if __name__ == "__main__":
    unittest.main()
