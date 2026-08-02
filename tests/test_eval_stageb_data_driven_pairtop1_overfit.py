import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from main import DeterministicEpochSampler
from tools.eval_stageb_data_driven_pairtop1_overfit import (
    EXPECTED_AMP_SCALE,
    EXPECTED_ROWS,
    EXPECTED_UPDATES,
    HARDGAP3_VARIANT,
    VARIANT,
    _audit_hardgap3_deployment_outputs,
    _audit_optimizer_state,
    _audit_terminal_checkpoint,
    _audit_training_log,
    _pairtop1_gates,
)
from models.GroundingDINO.stage_b_data_driven_score import (
    data_driven_category_gate_mask,
)
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _file_record(path: Path):
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _gate_counts():
    return {
        "data_rows": 64,
        "runtime_rows": 64,
        "query_collision_rows": 0,
        "correct_rows": 61,
        "margin_rows": 61,
        "margin_directions": 122,
        "deployment_correct_rows": 61,
    }


def _optimizer():
    def state():
        return {
            "step": torch.tensor(float(EXPECTED_UPDATES)),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.ones(2),
        }

    rank_ids = list(range(39))
    patch_ids = list(range(39, 48))
    return {
        "param_groups": [
            {
                "params": rank_ids,
                "stage_b_data_driven_branch": "rank",
            },
            {
                "params": patch_ids,
                "stage_b_data_driven_branch": "patch",
            },
        ],
        "state": {param_id: state() for param_id in rank_ids + patch_ids},
    }


def _write_training_log(path: Path, *, skipped_row=None):
    lines = []
    for index in range(EXPECTED_UPDATES - 1):
        skipped = 1.0 if index == skipped_row else 0.0
        lines.append(
            "amp_step_skipped: %.4f amp_scale: %.4f optimizer_step: 1.0000 "
            "loss_stage_b_data_driven_deployment_hard: 0.1000 max mem: 23147"
            % (skipped, EXPECTED_AMP_SCALE)
        )
    lines.append(
        "Saved iteration checkpoint (optimizer_updates=500, "
        "reason=max_train_iters)."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PairTop1GateThresholdTest(unittest.TestCase):
    def test_exact_preregistered_boundary_passes(self):
        self.assertTrue(all(_pairtop1_gates(_gate_counts(), rows=64).values()))

    def test_all_64_still_passes(self):
        counts = _gate_counts()
        counts.update(
            correct_rows=64,
            margin_rows=64,
            margin_directions=128,
            deployment_correct_rows=64,
        )
        self.assertTrue(all(_pairtop1_gates(counts, rows=64).values()))

    def test_each_preregistered_metric_fails_one_below_boundary(self):
        for key, value, gate in (
            ("correct_rows", 60, "reciprocal_correct_at_least_61_of_64"),
            ("margin_rows", 60, "reciprocal_margin_at_least_61_of_64"),
            ("margin_directions", 121, "direction_margin_at_least_122_of_128"),
            (
                "deployment_correct_rows",
                60,
                "deployment_gap3_correct_at_least_61_of_64",
            ),
        ):
            counts = _gate_counts()
            counts[key] = value
            gates = _pairtop1_gates(counts, rows=64)
            self.assertFalse(gates[gate], key)

    def test_hardgap3_additional_boundaries_are_preregistered(self):
        counts = _gate_counts()
        counts.update(
            deployment_correct_directions=122,
            deployment_hard_valid_directions=128,
            deployment_hard_margin_directions=122,
            deployment_gate_enabled=1,
            deployment_eligible_mask_exact=1,
            deployment_patch_score_exact=1,
            deployment_top1_wiring_directions=128,
            deployment_actual_raw_correct_rows=61,
            deployment_actual_raw_correct_directions=122,
            deployment_actual_clamped_correct_rows=61,
            deployment_actual_clamped_correct_directions=122,
            deployment_iou_correct_mask_disagreements=0,
            deployment_top1_iou_correct_disagreements=0,
        )
        gates = _pairtop1_gates(
            counts, rows=64, variant=HARDGAP3_VARIANT
        )
        self.assertTrue(all(gates.values()))
        for key, value, gate in (
            (
                "deployment_correct_directions",
                121,
                "deployment_gap3_directions_at_least_122_of_128",
            ),
            (
                "deployment_hard_valid_directions",
                127,
                "hardgap3_competitor_available_128_of_128",
            ),
            (
                "deployment_hard_margin_directions",
                121,
                "hardgap3_margin_at_least_122_of_128",
            ),
            (
                "deployment_gate_enabled",
                0,
                "actual_deployment_gate_enabled",
            ),
            (
                "deployment_eligible_mask_exact",
                0,
                "actual_deployment_eligible_mask_exact",
            ),
            (
                "deployment_patch_score_exact",
                0,
                "actual_deployment_patch_score_exact",
            ),
            (
                "deployment_top1_wiring_directions",
                127,
                "actual_deployment_top1_wiring_128_of_128",
            ),
            (
                "deployment_actual_raw_correct_rows",
                60,
                "actual_deployment_raw_rows_at_least_61_of_64",
            ),
            (
                "deployment_actual_raw_correct_directions",
                121,
                "actual_deployment_raw_directions_at_least_122_of_128",
            ),
            (
                "deployment_actual_clamped_correct_rows",
                60,
                "actual_deployment_clamped_rows_at_least_61_of_64",
            ),
            (
                "deployment_actual_clamped_correct_directions",
                121,
                "actual_deployment_clamped_directions_at_least_122_of_128",
            ),
        ):
            failed = dict(counts)
            failed[key] = value
            self.assertFalse(
                _pairtop1_gates(
                    failed, rows=64, variant=HARDGAP3_VARIANT
                )[gate]
            )


class HardGap3DeploymentAuditTest(unittest.TestCase):
    def _example(self):
        query_count = 20
        boxes = torch.full((1, query_count, 4), 0.1)
        boxes[0, 0] = torch.tensor([0.2, 0.2, 0.2, 0.2])
        boxes[0, 1] = torch.tensor([0.8, 0.8, 0.2, 0.2])
        patch = torch.tensor([[[10.0], [9.0]] + [[-1.0]] * 18])
        candidate_2d = torch.ones((1, query_count), dtype=torch.bool)
        eligible_2d, normalized = data_driven_category_gate_mask(
            patch, candidate_2d, max_gap=3.0, clip=5.0
        )
        self.assertEqual(int(eligible_2d.sum().item()), 2)
        candidate = candidate_2d[:, :, None].expand(-1, -1, 2).clone()
        eligible = eligible_2d[:, :, None].expand(-1, -1, 2).clone()
        normalized = normalized[:, :, None].expand(-1, -1, 2).clone()
        raw_rank = torch.zeros((1, query_count, 2))
        raw_rank[0, 0, 0] = 5.0
        raw_rank[0, 1, 1] = 5.0
        raw_rank[0, 0, 1] = 1.0
        raw_rank[0, 1, 0] = 1.0
        text_min = raw_rank.amin(dim=1, keepdim=True)
        text_max = raw_rank.amax(dim=1, keepdim=True)
        below_min = torch.nextafter(
            text_min, torch.full_like(text_min, -torch.inf)
        )
        deployed_rank = torch.where(
            eligible,
            raw_rank,
            below_min + raw_rank - text_max,
        )
        outputs = {
            "pred_boxes": boxes,
            "pred_logits_patch": patch,
            "stage_b_data_driven_text_rank_score": raw_rank,
            "stage_b_data_driven_rank_score": deployed_rank,
            "stage_b_data_driven_candidate_mask": candidate,
            "stage_b_data_driven_category_gate_eligible_mask": eligible,
            "stage_b_data_driven_category_gate_patch_score": normalized,
        }
        targets = [
            {
                "boxes": torch.stack((boxes[0, 0], boxes[0, 1])),
                "stage_b_data_driven_assignment_role": torch.tensor(
                    [0, 1], dtype=torch.int64
                ),
                "stage_b_data_driven_assignment_valid": torch.tensor(
                    True, dtype=torch.bool
                ),
            }
        ]
        runtime = {
            "training_flag": False,
            "evaluation_flag": True,
            "inference_only_override": True,
            "max_gap": 3.0,
            "patch_score_clip": 5.0,
        }
        return outputs, targets, runtime

    def test_real_deployment_outputs_are_independently_verified(self):
        outputs, targets, runtime = self._example()
        for patch_logits in (
            outputs["pred_logits_patch"],
            outputs["pred_logits_patch"][..., 0],
        ):
            with self.subTest(shape=tuple(patch_logits.shape)):
                outputs["pred_logits_patch"] = patch_logits
                audit = _audit_hardgap3_deployment_outputs(
                    outputs, targets, gate_runtime=runtime
                )
                counts = audit["counts"]
                self.assertEqual(counts["deployment_gate_enabled"], 1)
                self.assertEqual(counts["deployment_eligible_mask_exact"], 1)
                self.assertEqual(counts["deployment_patch_score_exact"], 1)
                self.assertEqual(counts["deployment_top1_wiring_directions"], 2)
                self.assertEqual(counts["deployment_actual_raw_correct_rows"], 1)
                self.assertEqual(
                    counts["deployment_actual_raw_correct_directions"], 2
                )
                self.assertEqual(
                    counts["deployment_actual_clamped_correct_rows"], 1
                )
                self.assertEqual(
                    counts["deployment_actual_clamped_correct_directions"], 2
                )
                self.assertEqual(
                    counts["deployment_iou_correct_mask_disagreements"], 0
                )

    def test_tampered_deployed_rank_cannot_pass_wiring_audit(self):
        outputs, targets, runtime = self._example()
        outputs["stage_b_data_driven_rank_score"] = outputs[
            "stage_b_data_driven_rank_score"
        ].clone()
        outputs["stage_b_data_driven_rank_score"][0, 2, :] = 100.0
        audit = _audit_hardgap3_deployment_outputs(
            outputs, targets, gate_runtime=runtime
        )
        self.assertEqual(
            audit["counts"]["deployment_top1_wiring_directions"], 0
        )


class PairTop1TerminalAuditTest(unittest.TestCase):
    def test_optimizer_audit_rejects_step_and_moment_drift(self):
        payload = {"optimizer": _optimizer()}
        self.assertEqual(_audit_optimizer_state(payload)["state_count"], 48)

        bad_step = copy.deepcopy(payload)
        bad_step["optimizer"]["state"][0]["step"] = torch.tensor(499.0)
        with self.assertRaisesRegex(RuntimeError, "expected 500"):
            _audit_optimizer_state(bad_step)

        bad_moment = copy.deepcopy(payload)
        bad_moment["optimizer"]["state"][47]["exp_avg"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _audit_optimizer_state(bad_moment)

        missing = copy.deepcopy(payload)
        del missing["optimizer"]["state"][47]
        with self.assertRaisesRegex(RuntimeError, "exactly cover"):
            _audit_optimizer_state(missing)

    def test_training_log_proves_zero_amp_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "info.txt"
            _write_training_log(path)
            audit = _audit_training_log(path)
            self.assertEqual(audit["metric_rows"], 499)
            self.assertEqual(audit["amp_skipped_steps"], 0)

            _write_training_log(path, skipped_row=7)
            with self.assertRaisesRegex(RuntimeError, "AMP skipped step"):
                _audit_training_log(path)

    def _payload(self, root: Path):
        output = root / "output"
        output.mkdir()
        config = root / "config.py"
        dataset = root / "datasets.json"
        base = root / "a1.pth"
        checkpoint = output / "checkpoint_iter.pth"
        config.write_text("# config\n", encoding="ascii")
        dataset.write_text("{}\n", encoding="ascii")
        base.write_bytes(b"a1")
        checkpoint.write_bytes(b"checkpoint")
        _write_training_log(output / "info.txt")

        sampling = DeterministicEpochSampler(EXPECTED_ROWS, seed=42).ledger_state(
            EXPECTED_UPDATES - 1
        )
        sampling.update(
            loader_seed=1042,
            loader_epoch_seed=1541,
            persistent_workers=False,
        )
        args = {
            "seed": 42,
            "batch_size": 64,
            "epochs": 500,
            "lr_drop": 500,
            "stage_b_data_driven_epoch_checkpoint_interval": 500,
            "max_train_iters": 500,
            "iter_checkpoint_interval": 500,
            "save_checkpoint_interval": 500,
            "num_workers": 0,
            "prefetch_factor": 1,
            "pin_memory": False,
            "persistent_workers": False,
            "gradient_accumulation_steps": 1,
            "amp": True,
            "amp_init_scale": 8192.0,
            "resume": "",
            "stage_b_data_driven_variant_id": VARIANT,
            "stage_b_data_driven_no_teacher_contract": (
                "b58_only_random_independent_heads_v1"
            ),
            "stage_b_data_driven_assignment_weight": 1.0,
            "stage_b_data_driven_category_gate": False,
            "stage_b_data_driven_category_gate_max_gap": 3.0,
            "stage_b_data_driven_patch_score_clip": 5.0,
            "stage_b_data_driven_positive_iou_threshold": 0.5,
            "stage_b_data_driven_rank_negative_iou_threshold": 0.3,
            "stage_b_data_driven_rank_weight": 0.0,
            "stage_b_data_driven_patch_weight": 1.0,
            "stage_b_gdino_score_adapter": False,
            "stage_b_u0_patch_rank": False,
            "stage_b_v7": False,
            "stage_b_v11_fixed_text": False,
            "stage_b_legacy_global_gate": False,
            "pretrain_model_path": str(base),
            "config_file": str(config),
            "datasets": str(dataset),
            "output_dir": str(output),
        }
        code_paths = (
            REPO_ROOT / "main.py",
            REPO_ROOT / "engine.py",
            REPO_ROOT / "models/GroundingDINO/groundingdino.py",
            REPO_ROOT / "models/GroundingDINO/stage_b_data_driven_score.py",
        )
        args["stage_b_data_driven_config_import_chain"] = [
            _file_record(config)
        ]
        args["stage_b_data_driven_dataset_config"] = _file_record(dataset)
        args["stage_b_data_driven_training_provenance"] = {
            "schema": "pivot.stageb.data_driven_training_provenance/v1",
            "code_files": [_file_record(path) for path in code_paths],
            "dataset_asset_files": [_file_record(dataset)],
        }
        payload = {
            "model": {"weight": torch.ones(2)},
            "criterion": {"contract": torch.tensor(4)},
            "optimizer": _optimizer(),
            "lr_scheduler": {},
            "scaler": {
                "scale": 8192.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 500,
            },
            "epoch": 499,
            "iteration": 0,
            "optimizer_updates": 500,
            "epoch_finished": True,
            "args": args,
            "checkpoint_reason": "max_train_iters",
            "stage_b_data_driven_sampling_state": sampling,
        }
        return payload, config, dataset, checkpoint, base

    def test_terminal_checkpoint_passes_and_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, config, dataset, checkpoint, base = self._payload(root)
            cfg = SLConfig({"eval": True})
            audit = _audit_terminal_checkpoint(
                payload,
                cfg=cfg,
                config_path=config,
                dataset_path=dataset,
                checkpoint_path=checkpoint,
                base_path=base,
            )
            self.assertEqual(audit["status"], "passed")
            self.assertTrue(audit["zero_amp_skips_proven"])

            bad = copy.deepcopy(payload)
            bad["scaler"]["_growth_tracker"] = 499
            with self.assertRaisesRegex(RuntimeError, "zero AMP skips"):
                _audit_terminal_checkpoint(
                    bad,
                    cfg=SLConfig({"eval": True}),
                    config_path=config,
                    dataset_path=dataset,
                    checkpoint_path=checkpoint,
                    base_path=base,
                )

            bad = copy.deepcopy(payload)
            bad["args"]["stage_b_gdino_score_adapter"] = True
            with self.assertRaisesRegex(RuntimeError, "saved runtime drifted"):
                _audit_terminal_checkpoint(
                    bad,
                    cfg=SLConfig({"eval": True}),
                    config_path=config,
                    dataset_path=dataset,
                    checkpoint_path=checkpoint,
                    base_path=base,
                )

    def test_hardgap3_terminal_checkpoint_requires_exact_separate_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, config, dataset, checkpoint, base = self._payload(root)
            payload["args"]["stage_b_data_driven_variant_id"] = HARDGAP3_VARIANT
            payload["args"]["stage_b_data_driven_deployment_weight"] = 1.0
            cfg = SLConfig(
                {
                    "eval": True,
                    "stage_b_data_driven_variant_id": HARDGAP3_VARIANT,
                }
            )
            audit = _audit_terminal_checkpoint(
                payload,
                cfg=cfg,
                config_path=config,
                dataset_path=dataset,
                checkpoint_path=checkpoint,
                base_path=base,
            )
            self.assertEqual(audit["status"], "passed")

            del payload["args"]["stage_b_data_driven_deployment_weight"]
            with self.assertRaisesRegex(
                RuntimeError, "deployment_weight"
            ):
                _audit_terminal_checkpoint(
                    payload,
                    cfg=cfg,
                    config_path=config,
                    dataset_path=dataset,
                    checkpoint_path=checkpoint,
                    base_path=base,
                )

if __name__ == "__main__":
    unittest.main()
