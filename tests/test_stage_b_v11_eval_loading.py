import inspect
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from tools import eval_refcoco_stageb as ref_eval
from tools import eval_stageb_tn_val as tn_eval
from tools import eval_text_groundingdino_refcoco_tn as combined_eval
from tools.stageb_ref_split_contract import REF_SPLITS


class _EvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.stage_b_fixed_text_scorer = nn.Linear(2, 2)


class StageBV11EvalLoadingTest(unittest.TestCase):
    def test_formal_word_veto_eval_rebinds_probe_admission(self):
        from tools import (
            run_stageb_confidence_adapter_veto_probe_evaluation as promotion,
        )

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "admission.json"
            report.write_text("{}\n", encoding="ascii")
            audit = {
                "status": "verified",
                "decision": "admit_to_formal_training",
            }
            cfg = types.SimpleNamespace(
                stage_b_dense_duty_confidence_probe_admission_contract=(
                    "u300_word_veto_strict1607_v1"
                ),
                stage_b_dense_duty_confidence_probe_admission_report=str(report),
            )
            with (
                mock.patch.object(promotion, "REPORT", report),
                mock.patch.object(
                    promotion,
                    "verify_admission_report",
                    return_value=audit,
                ) as verify,
            ):
                ref_eval._bind_dense_duty_formal_probe_admission(cfg)
            verify.assert_called_once_with(report.resolve())
            self.assertEqual(
                cfg.stage_b_dense_duty_confidence_probe_admission_audit,
                audit,
            )

    def test_formal_v4_absolute_cap_eval_rebinds_its_probe_admission(self):
        from tools import (
            run_stageb_confidence_adapter_veto_cap_probe_evaluation as promotion,
        )

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "admission.json"
            report.write_text("{}\n", encoding="ascii")
            audit = {
                "status": "verified",
                "decision": "admit_to_formal_training",
            }
            cfg = types.SimpleNamespace(
                stage_b_dense_duty_confidence_probe_admission_contract=(
                    "u300_word_veto_absolute_cap_strict1607_v4"
                ),
                stage_b_dense_duty_confidence_probe_admission_report=str(report),
            )
            with (
                mock.patch.object(promotion, "REPORT", report),
                mock.patch.object(
                    promotion,
                    "verify_admission_report",
                    return_value=audit,
                ) as verify,
            ):
                ref_eval._bind_dense_duty_formal_probe_admission(cfg)
            verify.assert_called_once_with(report.resolve())
            self.assertEqual(
                cfg.stage_b_dense_duty_confidence_probe_admission_audit,
                audit,
            )

    def test_formal_v5_gated_pool_eval_rebinds_its_probe_admission(self):
        import tools

        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "admission.json"
            report.write_text("{}\n", encoding="ascii")
            audit = {
                "status": "verified",
                "decision": "admit_to_formal_training",
            }
            promotion = types.SimpleNamespace(
                REPORT=report,
                verify_admission_report=mock.Mock(return_value=audit),
            )
            cfg = types.SimpleNamespace(
                stage_b_dense_duty_confidence_probe_admission_contract=(
                    "u300_word_veto_gated_pool_absolute_cap_strict1607_v5"
                ),
                stage_b_dense_duty_confidence_probe_admission_report=str(report),
            )
            with mock.patch.object(
                tools,
                "run_stageb_confidence_adapter_veto_gated_pool_probe_evaluation",
                promotion,
                create=True,
            ):
                ref_eval._bind_dense_duty_formal_probe_admission(cfg)
            promotion.verify_admission_report.assert_called_once_with(report.resolve())
            self.assertEqual(
                cfg.stage_b_dense_duty_confidence_probe_admission_audit,
                audit,
            )

    def test_dense_duty_routes_ref_to_rank_and_tn_to_confidence(self):
        cfg = types.SimpleNamespace(
            stage_b_v11_fixed_text=True,
            stage_b_v7=False,
            stage_b_v15_decoupled_confidence=True,
            stage_b_v22_score_ownership="independent_decoders_two_phase",
        )
        rank = torch.tensor([[[0.1], [0.9]]])
        confidence = torch.tensor([[[0.8], [0.2]]])
        outputs = {
            "stage_b_v15_dense_rank_score": rank,
            "stage_b_v7_final_score": confidence,
        }

        self.assertTrue(torch.equal(ref_eval._slot_scores(outputs, cfg, 0.0), rank))
        self.assertTrue(
            torch.equal(tn_eval._slot_scores(outputs, cfg, 0.0), confidence)
        )

    def test_v11_only_config_uses_compatibility_scorer_scores(self):
        cfg = types.SimpleNamespace(
            stage_b_v11_fixed_text=True,
            stage_b_v7=False,
        )
        expected = torch.tensor([[[0.2], [0.8]]])
        outputs = {"stage_b_v7_final_score": expected}

        self.assertTrue(ref_eval._uses_stage_b_post_candidate_scorer(cfg))
        self.assertTrue(torch.equal(ref_eval._slot_scores(outputs, cfg, 0.0), expected))
        self.assertTrue(torch.equal(tn_eval._slot_scores(outputs, cfg, 0.0), expected))
        self.assertIn(
            "_uses_stage_b_post_candidate_scorer",
            inspect.getsource(ref_eval._forward),
        )

    def test_eval_loader_rejects_missing_v11_scorer_state(self):
        model = _EvalModel()
        cfg = types.SimpleNamespace(
            modelname="test-v11",
            stage_b_v11_fixed_text=True,
        )

        def build(_cfg):
            return model, None, None

        state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("stage_b_fixed_text_scorer.")
        }
        with (
            mock.patch.object(ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build),
            mock.patch.object(
                ref_eval, "_torch_load_compat", return_value={"model": state}
            ),
            self.assertRaisesRegex(ValueError, "missing="),
        ):
            ref_eval._load_model(cfg, "truncated.pth", torch.device("cpu"))

    def test_dense_duty_partial_rank_loader_uses_diagnostic_validator(self):
        model = _EvalModel()
        cfg = types.SimpleNamespace(
            modelname="test-v11",
            stage_b_dense_duty=True,
            stage_b_dense_duty_partial_rank_diagnostic=True,
            stage_b_v11_fixed_text=False,
        )
        state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        audit = {
            "optimizer_updates": 3033,
            "checkpoint_reason": "signal",
            "diagnostic_only": True,
        }

        def build(_cfg):
            return model, None, None

        with (
            mock.patch.object(ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build),
            mock.patch.object(
                ref_eval, "_torch_load_compat", return_value={"model": state}
            ),
            mock.patch.object(
                ref_eval,
                "_validate_dense_duty_partial_rank_diagnostic_checkpoint",
                return_value=audit,
            ) as validate,
        ):
            loaded = ref_eval._load_model(
                cfg, "rank-partial.pth", torch.device("cpu")
            )

        self.assertIs(loaded, model)
        validate.assert_called_once()
        self.assertEqual(
            cfg.stage_b_dense_duty_partial_rank_diagnostic_optimizer_updates,
            3033,
        )
        self.assertEqual(
            cfg.stage_b_dense_duty_partial_rank_diagnostic_checkpoint_reason,
            "signal",
        )

    def test_dense_duty_partial_confidence_loader_uses_diagnostic_validator(self):
        model = _EvalModel()
        cfg = types.SimpleNamespace(
            modelname="test-v11",
            stage_b_dense_duty=True,
            stage_b_dense_duty_partial_confidence_diagnostic=True,
            stage_b_v11_fixed_text=False,
        )
        state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        audit = {
            "optimizer_updates": 4388,
            "expected_optimizer_updates": 4412,
            "checkpoint_reason": "signal",
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "confidence_evaluated": True,
        }

        def build(_cfg):
            return model, None, None

        with (
            mock.patch.object(ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build),
            mock.patch.object(
                ref_eval, "_torch_load_compat", return_value={"model": state}
            ),
            mock.patch.object(
                ref_eval,
                "_validate_dense_duty_partial_confidence_diagnostic_checkpoint",
                return_value=audit,
            ) as validate,
        ):
            loaded = ref_eval._load_model(
                cfg, "confidence-partial.pth", torch.device("cpu")
            )

        self.assertIs(loaded, model)
        validate.assert_called_once()
        self.assertEqual(
            cfg.stage_b_dense_duty_partial_confidence_diagnostic_optimizer_updates,
            4388,
        )
        self.assertEqual(
            cfg.stage_b_dense_duty_partial_confidence_diagnostic_expected_optimizer_updates,
            4412,
        )
        self.assertEqual(
            cfg.stage_b_dense_duty_partial_confidence_diagnostic_checkpoint_reason,
            "signal",
        )

    def test_dense_duty_partial_diagnostic_modes_are_mutually_exclusive(self):
        model = _EvalModel()
        cfg = types.SimpleNamespace(
            modelname="test-v11",
            stage_b_dense_duty=True,
            stage_b_dense_duty_partial_rank_diagnostic=True,
            stage_b_dense_duty_partial_confidence_diagnostic=True,
            stage_b_v11_fixed_text=False,
        )

        def build(_cfg):
            return model, None, None

        with (
            mock.patch.object(ref_eval.MODULE_BUILD_FUNCS, "get", return_value=build),
            mock.patch.object(
                ref_eval,
                "_torch_load_compat",
                return_value={"model": model.state_dict()},
            ),
            self.assertRaisesRegex(RuntimeError, "mutually exclusive"),
        ):
            ref_eval._load_model(cfg, "partial.pth", torch.device("cpu"))

    def test_partial_confidence_checkpoint_validator_rejects_terminal_update(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_20260730.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        cfg.stage_b_dense_duty_evaluation_scope = "probe"
        equal_keys = (
            "stage_b_dense_duty_no_stageb_teacher",
            "stage_b_v22_score_ownership",
            "stage_b_dense_duty_base_checkpoint_sha256",
            "stage_b_dense_duty_text_checkpoint_sha256",
            "stage_b_dense_duty_tn_manifest_sha256",
            "stage_b_dense_duty_dataset_config_sha256",
            "stage_b_v11_candidate_topk",
            "stage_b_v11_num_layers",
            "stage_b_v15_patch_rank_fusion",
            "stage_b_v15_patch_rank_weight",
            "stage_b_dense_duty_confidence_adapter_dim",
            "stage_b_dense_duty_confidence_init_seed",
            "stage_b_dense_duty_confidence_token_contract",
            "stage_b_dense_duty_confidence_pool_feature_contract",
            "stage_b_dense_duty_rank_source_checkpoint_path",
            "stage_b_dense_duty_rank_source_checkpoint_sha256",
            "stage_b_dense_duty_rank_source_optimizer_updates",
            "stage_b_dense_duty_rank_source_checkpoint_reason",
            "stage_b_dense_duty_rank_source_rank_sha256",
            "stage_b_dense_duty_rank_source_transferred_sha256",
            "stage_b_dense_duty_forward_pack_factor",
            "stage_b_dense_duty_logical_loss_batch_size",
            "stage_b_dense_duty_expected_forward_batch_size",
            "stage_b_dense_duty_expected_logical_batches_per_epoch",
            "stage_b_dense_duty_expected_physical_forwards_per_epoch",
        )
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary) / "train"
            train_dir.mkdir()
            canonical = train_dir / "checkpoint_iter.pth"
            snapshot = Path(temporary) / "snapshot.pth"
            canonical.write_bytes(b"same-checkpoint")
            snapshot.write_bytes(b"same-checkpoint")
            saved_args = {
                key: getattr(cfg, key)
                for key in equal_keys
            }
            saved_args.update(
                {
                    "output_dir": str(train_dir),
                    "max_train_iters": 4412,
                    "stage_b_dense_duty_execution_scope": "probe",
                    "stage_b_dense_duty_runtime_audit": {
                        "successful_optimizer_steps": 4388,
                        "optimizer_step_boundaries": 4388,
                        "amp_skipped_optimizer_steps": 0,
                        "nonfinite_gradient_boundaries": 0,
                        "zero_gradient_successful_steps": 0,
                        "max_active_grad_norm_preclip": 1.0,
                        "peak_reserved_bytes": 1,
                    },
                }
            )
            payload = {
                "args": saved_args,
                "optimizer_updates": 4388,
                "checkpoint_reason": "signal",
            }

            def resume(value, _args, *, checkpoint_path):
                return {
                    "phase": "confidence",
                    "optimizer_updates": value["optimizer_updates"],
                    "checkpoint_reason": value["checkpoint_reason"],
                    "rank_handoff": {"status": "passed"},
                }

            def audit(value, *, checkpoint_path):
                return {
                    "status": "passed",
                    "phase": "confidence",
                    "optimizer_updates": value["optimizer_updates"],
                    "lineage": {
                        "no_stage_b_teacher": True,
                        "execution_scope": "probe",
                    },
                }

            with (
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload",
                    side_effect=resume,
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                    side_effect=audit,
                ),
            ):
                result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    payload,
                    cfg,
                    checkpoint_path=snapshot,
                )
                self.assertTrue(result["diagnostic_only"])
                self.assertFalse(result["formal_gate_eligible"])
                self.assertTrue(result["confidence_evaluated"])
                self.assertEqual(result["remaining_optimizer_updates"], 24)

                payload["optimizer_updates"] = 4412
                saved_args["stage_b_dense_duty_runtime_audit"].update(
                    successful_optimizer_steps=4412,
                    optimizer_step_boundaries=4412,
                )
                with self.assertRaisesRegex(RuntimeError, "non-terminal"):
                    ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )

    def test_word_veto_u300_terminal_probe_checkpoint_is_diagnostic_only(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_probe_20260730.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        equal_keys = (
            "stage_b_dense_duty_no_stageb_teacher",
            "stage_b_v22_score_ownership",
            "stage_b_dense_duty_base_checkpoint_sha256",
            "stage_b_dense_duty_text_checkpoint_sha256",
            "stage_b_dense_duty_tn_manifest_sha256",
            "stage_b_dense_duty_dataset_config_sha256",
            "stage_b_v11_candidate_topk",
            "stage_b_v11_num_layers",
            "stage_b_v15_patch_rank_fusion",
            "stage_b_v15_patch_rank_weight",
            "stage_b_dense_duty_confidence_adapter_dim",
            "stage_b_dense_duty_confidence_init_seed",
            "stage_b_dense_duty_confidence_token_contract",
            "stage_b_dense_duty_confidence_pool_feature_contract",
            "stage_b_dense_duty_rank_source_checkpoint_path",
            "stage_b_dense_duty_rank_source_checkpoint_sha256",
            "stage_b_dense_duty_rank_source_optimizer_updates",
            "stage_b_dense_duty_rank_source_checkpoint_reason",
            "stage_b_dense_duty_rank_source_rank_sha256",
            "stage_b_dense_duty_rank_source_transferred_sha256",
            "stage_b_dense_duty_forward_pack_factor",
            "stage_b_dense_duty_logical_loss_batch_size",
            "stage_b_dense_duty_expected_forward_batch_size",
            "stage_b_dense_duty_expected_logical_batches_per_epoch",
            "stage_b_dense_duty_expected_physical_forwards_per_epoch",
            "stage_b_dense_duty_confidence_revision",
            "stage_b_dense_duty_confidence_phrase_aggregation",
            "stage_b_dense_duty_confidence_word_softmin_temperature",
            "stage_b_dense_duty_confidence_veto_gate_scale",
            "stage_b_dense_duty_positive_trust_contract",
            "stage_b_dense_duty_confidence_tn_scope",
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "stage_b_dense_duty_confidence_probe_admission_report",
        )
        source_closure = {"sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary) / "train"
            train_dir.mkdir()
            canonical = train_dir / "checkpoint_iter.pth"
            snapshot = Path(temporary) / "snapshot.pth"
            canonical.write_bytes(b"same-checkpoint")
            snapshot.write_bytes(b"same-checkpoint")
            saved_args = {key: getattr(cfg, key) for key in equal_keys}
            saved_args.update(
                {
                    "output_dir": str(train_dir),
                    "max_train_iters": 300,
                    "stage_b_dense_duty_execution_scope": "probe",
                    "stage_b_dense_duty_source_closure": source_closure,
                    "stage_b_dense_duty_runtime_audit": {
                        "successful_optimizer_steps": 300,
                        "optimizer_step_boundaries": 300,
                        "amp_skipped_optimizer_steps": 0,
                        "nonfinite_gradient_boundaries": 0,
                        "zero_gradient_successful_steps": 0,
                        "max_active_grad_norm_preclip": 1.0,
                        "peak_reserved_bytes": 1,
                    },
                }
            )
            payload = {
                "args": saved_args,
                "optimizer_updates": 300,
                "checkpoint_reason": "max_train_iters",
            }

            with (
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload",
                    return_value={
                        "phase": "confidence",
                        "optimizer_updates": 300,
                        "checkpoint_reason": "max_train_iters",
                        "rank_handoff": {"status": "passed"},
                    },
                ) as strict_resume,
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_evaluation_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 300,
                        "checkpoint_reason": "max_train_iters",
                        "rank_handoff": {"status": "passed"},
                    },
                ) as terminal_evaluation,
                mock.patch(
                    "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 300,
                        "lineage": {
                            "no_stage_b_teacher": True,
                            "execution_scope": "probe",
                        },
                    },
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.build_source_closure",
                    return_value=source_closure,
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_source_closure",
                    side_effect=lambda value: value,
                ),
            ):
                result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    payload,
                    cfg,
                    checkpoint_path=snapshot,
                )
                terminal_evaluation.assert_called_once()
                strict_resume.assert_not_called()
                cfg.stage_b_dense_duty_confidence_word_softmin_temperature = 0.2
                with self.assertRaisesRegex(RuntimeError, "terminal U300 probe"):
                    ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )

        self.assertTrue(result["terminal_checkpoint"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["formal_gate_eligible"])
        self.assertEqual(result["remaining_optimizer_updates"], 0)
        self.assertEqual(result["source_closure"], source_closure)

    def test_v4_absolute_cap_u300_terminal_checkpoint_is_diagnostic_only(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_cap_probe_20260731.py"
        ).resolve()
        cfg = types.SimpleNamespace(
            _filename=str(config_path),
            stage_b_dense_duty_evaluation_scope="probe",
            stage_b_dense_duty_confidence_expected_optimizer_updates=300,
            stage_b_dense_duty_confidence_revision=(
                "word_veto_coverage_absolute_cap_v4"
            ),
            stage_b_dense_duty_confidence_phrase_aggregation=(
                "trace_activated_word_veto_absolute_cap_v4"
            ),
            stage_b_dense_duty_raw_veto_gate_weight=1.0,
            stage_b_dense_duty_raw_veto_positive_margin=0.1,
            stage_b_dense_duty_raw_veto_tn_margin=0.15,
            stage_b_dense_duty_raw_veto_query_scope=(
                "tn_all_admitted_positive_carrier_v2"
            ),
            stage_b_dense_duty_confidence_word_softmin_temperature=0.1,
            stage_b_dense_duty_confidence_veto_gate_offset=0.05,
            stage_b_dense_duty_confidence_veto_gate_scale=0.1,
            stage_b_dense_duty_confidence_veto_coverage_offset=0.1,
            stage_b_dense_duty_confidence_veto_coverage_ramp=0.8,
            stage_b_dense_duty_confidence_veto_cap_temperature=0.1,
            stage_b_dense_duty_confidence_veto_cap_initial_ceiling=-0.1,
            stage_b_dense_duty_positive_trust_contract=(
                "net_total_confidence_delta_v1"
            ),
            stage_b_dense_duty_confidence_tn_scope="direct_trace_valid_v1",
            stage_b_v15_exclude_canonical_from_score=True,
            stage_b_dense_duty_no_stageb_teacher=True,
            stage_b_v22_score_ownership=(
                "rank_tower_stopgrad_token_adapter_two_phase"
            ),
            stage_b_dense_duty_confidence_probe_admission_contract=(
                "disabled_for_probe_v1"
            ),
            stage_b_dense_duty_confidence_probe_admission_report="",
        )
        saved_keys = (
            "stage_b_dense_duty_no_stageb_teacher",
            "stage_b_v22_score_ownership",
            "stage_b_dense_duty_confidence_revision",
            "stage_b_dense_duty_confidence_phrase_aggregation",
            "stage_b_dense_duty_raw_veto_gate_weight",
            "stage_b_dense_duty_raw_veto_positive_margin",
            "stage_b_dense_duty_raw_veto_tn_margin",
            "stage_b_dense_duty_raw_veto_query_scope",
            "stage_b_dense_duty_confidence_word_softmin_temperature",
            "stage_b_dense_duty_confidence_veto_gate_offset",
            "stage_b_dense_duty_confidence_veto_gate_scale",
            "stage_b_dense_duty_confidence_veto_coverage_offset",
            "stage_b_dense_duty_confidence_veto_coverage_ramp",
            "stage_b_dense_duty_confidence_veto_cap_temperature",
            "stage_b_dense_duty_confidence_veto_cap_initial_ceiling",
            "stage_b_dense_duty_positive_trust_contract",
            "stage_b_dense_duty_confidence_tn_scope",
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "stage_b_dense_duty_confidence_probe_admission_report",
        )
        source_closure = {"sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary) / "train"
            train_dir.mkdir()
            canonical = train_dir / "checkpoint_iter.pth"
            snapshot = Path(temporary) / "snapshot.pth"
            canonical.write_bytes(b"same-v4-checkpoint")
            snapshot.write_bytes(b"same-v4-checkpoint")
            saved_args = {key: getattr(cfg, key) for key in saved_keys}
            saved_args.update(
                {
                    "output_dir": str(train_dir),
                    "max_train_iters": 300,
                    "stage_b_dense_duty_execution_scope": "probe",
                    "stage_b_dense_duty_source_closure": source_closure,
                    "stage_b_dense_duty_runtime_audit": {
                        "successful_optimizer_steps": 300,
                        "optimizer_step_boundaries": 300,
                        "amp_skipped_optimizer_steps": 0,
                        "nonfinite_gradient_boundaries": 0,
                        "zero_gradient_successful_steps": 0,
                        "max_active_grad_norm_preclip": 1.0,
                        "peak_reserved_bytes": 1,
                    },
                }
            )
            payload = {
                "args": saved_args,
                "optimizer_updates": 300,
                "checkpoint_reason": "max_train_iters",
            }
            terminal_resume = {
                "status": "passed",
                "phase": "confidence",
                "optimizer_updates": 300,
                "checkpoint_reason": "max_train_iters",
                "rank_handoff": {"status": "passed"},
            }

            with (
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload"
                ) as strict_resume,
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_evaluation_checkpoint_payload",
                    return_value=terminal_resume,
                ) as terminal_evaluation,
                mock.patch(
                    "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 300,
                        "lineage": {
                            "no_stage_b_teacher": True,
                            "execution_scope": "probe",
                        },
                    },
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.build_source_closure",
                    return_value=source_closure,
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_source_closure",
                    side_effect=lambda value: value,
                ),
            ):
                result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    payload,
                    cfg,
                    checkpoint_path=snapshot,
                )
                terminal_evaluation.assert_called_once()
                strict_resume.assert_not_called()

                cfg.stage_b_dense_duty_confidence_veto_coverage_ramp = 0.7
                with self.assertRaisesRegex(RuntimeError, "terminal U300 probe"):
                    ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )

        self.assertTrue(result["terminal_checkpoint"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["formal_gate_eligible"])
        self.assertEqual(result["remaining_optimizer_updates"], 0)
        self.assertEqual(result["source_closure"], source_closure)

    def test_v5_gated_pool_u300_terminal_checkpoint_is_diagnostic_only(self):
        config_path = Path(__file__).resolve()
        cfg = types.SimpleNamespace(
            _filename=str(config_path),
            stage_b_dense_duty_evaluation_scope="probe",
            stage_b_dense_duty_confidence_expected_optimizer_updates=300,
            stage_b_dense_duty_confidence_revision=(
                "word_veto_gated_pool_absolute_cap_v5"
            ),
            stage_b_dense_duty_confidence_phrase_aggregation=(
                "trace_activated_word_veto_gated_pool_absolute_cap_v5"
            ),
            stage_b_dense_duty_raw_veto_gate_weight=1.0,
            stage_b_dense_duty_raw_veto_positive_margin=0.1,
            stage_b_dense_duty_raw_veto_tn_margin=0.15,
            stage_b_dense_duty_raw_veto_query_scope=(
                "tn_all_admitted_positive_carrier_v2"
            ),
            stage_b_dense_duty_confidence_word_softmin_temperature=0.1,
            stage_b_dense_duty_confidence_veto_gate_offset=0.05,
            stage_b_dense_duty_confidence_veto_gate_scale=0.1,
            stage_b_dense_duty_confidence_veto_coverage_offset=0.1,
            stage_b_dense_duty_confidence_veto_coverage_ramp=0.8,
            stage_b_dense_duty_confidence_veto_cap_temperature=0.1,
            stage_b_dense_duty_confidence_veto_cap_initial_ceiling=-0.1,
            stage_b_dense_duty_positive_trust_contract=(
                "net_total_confidence_delta_v1"
            ),
            stage_b_dense_duty_confidence_tn_scope="direct_trace_valid_v1",
            stage_b_v15_exclude_canonical_from_score=True,
            stage_b_dense_duty_no_stageb_teacher=True,
            stage_b_v22_score_ownership=(
                "rank_tower_stopgrad_token_adapter_two_phase"
            ),
            stage_b_dense_duty_confidence_probe_admission_contract=(
                "disabled_for_probe_v1"
            ),
            stage_b_dense_duty_confidence_probe_admission_report="",
        )
        contract_keys = tuple(
            key for key in vars(cfg) if not key.startswith("_")
        )
        source_closure = {"sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary) / "train"
            train_dir.mkdir()
            canonical = train_dir / "checkpoint_iter.pth"
            snapshot = Path(temporary) / "snapshot.pth"
            canonical.write_bytes(b"same-v5-checkpoint")
            snapshot.write_bytes(b"same-v5-checkpoint")
            saved_args = {key: getattr(cfg, key) for key in contract_keys}
            saved_args.update(
                {
                    "output_dir": str(train_dir),
                    "max_train_iters": 300,
                    "stage_b_dense_duty_execution_scope": "probe",
                    "stage_b_dense_duty_source_closure": source_closure,
                    "stage_b_dense_duty_runtime_audit": {
                        "successful_optimizer_steps": 300,
                        "optimizer_step_boundaries": 300,
                        "amp_skipped_optimizer_steps": 0,
                        "nonfinite_gradient_boundaries": 0,
                        "zero_gradient_successful_steps": 0,
                        "max_active_grad_norm_preclip": 1.0,
                        "peak_reserved_bytes": 1,
                    },
                }
            )
            payload = {
                "args": saved_args,
                "optimizer_updates": 300,
                "checkpoint_reason": "max_train_iters",
            }
            with (
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload"
                ) as strict_resume,
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_evaluation_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 300,
                        "checkpoint_reason": "max_train_iters",
                        "rank_handoff": {"status": "passed"},
                    },
                ) as terminal_evaluation,
                mock.patch(
                    "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 300,
                        "lineage": {
                            "no_stage_b_teacher": True,
                            "execution_scope": "probe",
                        },
                    },
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.build_source_closure",
                    return_value=source_closure,
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_source_closure",
                    side_effect=lambda value: value,
                ),
            ):
                result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    payload,
                    cfg,
                    checkpoint_path=snapshot,
                )
            terminal_evaluation.assert_called_once()
            strict_resume.assert_not_called()

        self.assertTrue(result["terminal_checkpoint"])
        self.assertEqual(result["remaining_optimizer_updates"], 0)
        self.assertEqual(result["source_closure"], source_closure)

    def test_partial_rank_cli_contract_is_ref8_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=True,
                config=str(
                    Path(
                        "config/ablations/cfg_stageb_dense_duty_rank_20260728.py"
                    ).resolve()
                ),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["rank-partial.pth"],
                skip_tn=True,
                skip_ref=False,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            cfg = types.SimpleNamespace(
                stage_b_dense_duty=True,
                stage_b_dense_duty_phase="rank",
            )
            combined_eval._validate_partial_dense_duty_rank_diagnostic_args(
                args, cfg
            )
            args.skip_tn = False
            with self.assertRaisesRegex(ValueError, "Ref-only"):
                combined_eval._validate_partial_dense_duty_rank_diagnostic_args(
                    args, cfg
                )

    def test_partial_confidence_cli_and_metadata_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=False,
                partial_dense_duty_confidence_diagnostic=True,
                config=str(
                    Path(
                        "config/ablations/"
                        "cfg_stageb_dense_duty_confidence_adapter_20260730.py"
                    ).resolve()
                ),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["confidence-partial.pth"],
                tn_jsonl=str(
                    Path(
                        "data/eval_manifests/"
                        "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                        "eval_manifest.jsonl"
                    ).resolve()
                ),
                tn_splits=["refcocop_val", "refcocog_umd_val"],
                skip_tn=False,
                skip_ref=False,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                threshold_tprs=[0.75, 0.9, 0.95],
                score_thresholds=[0.5],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            cfg = types.SimpleNamespace(
                stage_b_dense_duty=True,
                stage_b_dense_duty_phase="confidence",
                stage_b_v22_train_phase="confidence",
                stage_b_v22_score_ownership=(
                    "rank_tower_stopgrad_token_adapter_two_phase"
                ),
                stage_b_dense_duty_partial_confidence_diagnostic=True,
                stage_b_dense_duty_partial_confidence_diagnostic_optimizer_updates=4388,
                stage_b_dense_duty_partial_confidence_diagnostic_expected_optimizer_updates=4412,
                stage_b_dense_duty_partial_confidence_diagnostic_checkpoint_reason="signal",
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )
            with mock.patch.object(
                combined_eval, "file_record", return_value={"sha256": "a" * 64}
            ):
                provenance = combined_eval._evaluation_summary_provenance(
                    cfg=cfg,
                    args=args,
                    checkpoint=Path(__file__),
                    data_root=Path(temporary),
                )

            self.assertTrue(provenance["diagnostic_only"])
            self.assertFalse(provenance["formal_gate_eligible"])
            self.assertTrue(provenance["confidence_evaluated"])
            self.assertFalse(provenance["terminal_checkpoint"])
            self.assertEqual(provenance["optimizer_updates"], 4388)
            self.assertEqual(provenance["remaining_optimizer_updates"], 24)
            self.assertEqual(
                provenance["score_ownership"],
                "rank_tower_stopgrad_token_adapter_two_phase",
            )

            args.skip_ref = True
            args.tn_jsonl = str(
                Path(
                    "data/eval_manifests/"
                    "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                    "semantic_stageb_union_image_disjoint_manifest.jsonl"
                ).resolve()
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )
            args.tn_jsonl = str(
                Path(
                    "data/eval_manifests/"
                    "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                    "eval_manifest.jsonl"
                ).resolve()
            )
            with self.assertRaisesRegex(ValueError, "strict1607"):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

            args.skip_tn = True
            with self.assertRaisesRegex(ValueError, "require TN"):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

    def test_v4_absolute_cap_cli_contract_accepts_exact_config_and_rejects_drift(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_cap_probe_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=False,
                partial_dense_duty_confidence_diagnostic=True,
                config=str(config_path),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["confidence-v4-terminal.pth"],
                tn_jsonl=str(
                    Path(
                        "data/eval_manifests/"
                        "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                        "semantic_stageb_union_image_disjoint_manifest.jsonl"
                    ).resolve()
                ),
                tn_splits=["refcocop_val", "refcocog_umd_val"],
                skip_tn=False,
                skip_ref=True,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                threshold_tprs=[0.75, 0.9, 0.95],
                score_thresholds=[0.5],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )

            cfg.stage_b_dense_duty_confidence_veto_cap_temperature = 0.2
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

    def test_v5_gated_pool_cli_contract_accepts_exact_config_and_rejects_drift(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_probe_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=False,
                partial_dense_duty_confidence_diagnostic=True,
                config=str(config_path),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["confidence-v5-terminal.pth"],
                tn_jsonl=str(
                    Path(
                        "data/eval_manifests/"
                        "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                        "semantic_stageb_union_image_disjoint_manifest.jsonl"
                    ).resolve()
                ),
                tn_splits=["refcocop_val", "refcocog_umd_val"],
                skip_tn=False,
                skip_ref=True,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                threshold_tprs=[0.75, 0.9, 0.95],
                score_thresholds=[0.5],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )
            cfg.stage_b_dense_duty_confidence_veto_gate_offset = 0.0
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )
            cfg.stage_b_dense_duty_confidence_veto_gate_offset = 0.05
            cfg.stage_b_dense_duty_confidence_phrase_aggregation = (
                "trace_activated_word_veto_absolute_cap_v4"
            )
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

    def test_v18_tail_paired_u50_cli_contract_is_fixed_and_fail_closed(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_probe_u0050_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=False,
                partial_dense_duty_confidence_diagnostic=True,
                config=str(config_path),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["confidence-v18-u50-terminal.pth"],
                tn_jsonl=str(
                    Path(
                        "data/eval_manifests/"
                        "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                        "semantic_stageb_union_image_disjoint_manifest.jsonl"
                    ).resolve()
                ),
                tn_splits=["refcocop_val", "refcocog_umd_val"],
                skip_tn=False,
                skip_ref=True,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                threshold_tprs=[0.75, 0.9, 0.95],
                score_thresholds=[0.5],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )

            cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 300
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )
            cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 50
            cfg.stage_b_dense_duty_raw_veto_query_scope = (
                "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6"
            )
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

    def test_v18_tail_paired_u50_terminal_checkpoint_is_diagnostic_only(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_probe_u0050_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        source_closure = {"sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary) / "train"
            train_dir.mkdir()
            canonical = train_dir / "checkpoint_iter.pth"
            snapshot = Path(temporary) / "snapshot.pth"
            canonical.write_bytes(b"same-v18-u50-checkpoint")
            snapshot.write_bytes(b"same-v18-u50-checkpoint")
            saved_args = dict(cfg._cfg_dict)
            saved_args.update(
                {
                    "output_dir": str(train_dir),
                    "max_train_iters": 50,
                    "stage_b_dense_duty_execution_scope": "probe",
                    "stage_b_dense_duty_source_closure": source_closure,
                    "stage_b_dense_duty_runtime_audit": {
                        "successful_optimizer_steps": 50,
                        "optimizer_step_boundaries": 50,
                        "amp_skipped_optimizer_steps": 0,
                        "nonfinite_gradient_boundaries": 0,
                        "zero_gradient_successful_steps": 0,
                        "max_active_grad_norm_preclip": 1.0,
                        "peak_reserved_bytes": 1,
                    },
                }
            )
            payload = {
                "args": saved_args,
                "optimizer_updates": 50,
                "checkpoint_reason": "max_train_iters",
            }

            def resume(value, _args, *, checkpoint_path):
                return {
                    "phase": "confidence",
                    "optimizer_updates": value["optimizer_updates"],
                    "checkpoint_reason": value["checkpoint_reason"],
                    "rank_handoff": {"status": "passed"},
                }

            with (
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload",
                    side_effect=resume,
                ) as strict_resume,
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_evaluation_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 50,
                        "checkpoint_reason": "max_train_iters",
                        "rank_handoff": {"status": "passed"},
                    },
                ) as terminal_evaluation,
                mock.patch(
                    "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 50,
                        "lineage": {
                            "no_stage_b_teacher": True,
                            "execution_scope": "probe",
                        },
                    },
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.build_source_closure",
                    return_value=source_closure,
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_source_closure",
                    side_effect=lambda value: value,
                ),
            ):
                result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    payload,
                    cfg,
                    checkpoint_path=snapshot,
                )
                terminal_evaluation.assert_called_once()
                strict_resume.assert_not_called()

                payload["optimizer_updates"] = 49
                payload["checkpoint_reason"] = "signal"
                saved_args["stage_b_dense_duty_runtime_audit"].update(
                    successful_optimizer_steps=49,
                    optimizer_step_boundaries=49,
                )
                with self.assertRaisesRegex(RuntimeError, "terminal U50 v18 probe"):
                    ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )
                payload["optimizer_updates"] = 50
                payload["checkpoint_reason"] = "max_train_iters"
                saved_args["stage_b_dense_duty_runtime_audit"].update(
                    successful_optimizer_steps=50,
                    optimizer_step_boundaries=50,
                )
                cfg.stage_b_dense_duty_raw_veto_query_scope = (
                    "tn_all_admitted_tail_weighted_carrier_"
                    "positive_carrier_paired_v6"
                )
                with self.assertRaisesRegex(RuntimeError, "terminal U50 v18 probe"):
                    ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )

        self.assertTrue(result["terminal_checkpoint"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["formal_gate_eligible"])
        self.assertEqual(result["expected_optimizer_updates"], 50)
        self.assertEqual(result["remaining_optimizer_updates"], 0)
        self.assertEqual(result["source_closure"], source_closure)

    def test_v15_v17_u50_cli_contracts_are_fixed_and_fail_closed(self):
        specs = (
            (
                "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
                "carrier_affine_probe_u0050_20260731.py",
                "word_veto_gated_pool_carrier_affine_v15",
                "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
            ),
            (
                "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
                "tail_carrier_probe_u0050_20260731.py",
                "word_veto_gated_pool_tail_carrier_v17",
                "tn_all_admitted_tail_weighted_carrier_"
                "positive_carrier_paired_v6",
            ),
        )
        for filename, revision, expected_scope in specs:
            with self.subTest(revision=revision), tempfile.TemporaryDirectory() as temporary:
                config_path = Path("config/ablations", filename).resolve()
                cfg = combined_eval.SLConfig.fromfile(str(config_path))
                self.assertEqual(
                    cfg.stage_b_dense_duty_confidence_revision,
                    revision,
                )
                self.assertEqual(
                    cfg.stage_b_dense_duty_raw_veto_query_scope,
                    expected_scope,
                )
                args = types.SimpleNamespace(
                    partial_dense_duty_rank_diagnostic=False,
                    partial_dense_duty_confidence_diagnostic=True,
                    config=str(config_path),
                    output_dir=str(Path(temporary) / "fresh"),
                    ckpts=[f"confidence-{revision}-u50-terminal.pth"],
                    tn_jsonl=str(
                        Path(
                            "data/eval_manifests/"
                            "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                            "semantic_stageb_union_image_disjoint_manifest.jsonl"
                        ).resolve()
                    ),
                    tn_splits=["refcocop_val", "refcocog_umd_val"],
                    skip_tn=False,
                    skip_ref=True,
                    ref_splits=list(REF_SPLITS),
                    device="cuda:0",
                    batch_size=16,
                    num_workers=4,
                    seed=42,
                    amp=True,
                    topk=[1],
                    threshold_tprs=[0.75, 0.9, 0.95],
                    score_thresholds=[0.5],
                    max_ref_batches=0,
                    max_tn_batches=0,
                    no_per_example_records=False,
                    screen_calibration_manifest=False,
                    direct_prebuilt_tn=False,
                    category_gate_max_gaps=None,
                    category_gate_include_base_expert=False,
                    candidate_count_control=0,
                    holdout_level="none",
                    exclude_train_jsonl=[],
                )
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

                cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 300
                with self.assertRaisesRegex(
                    ValueError, "word-veto probe config contract is incomplete"
                ):
                    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                        args, cfg
                    )
                cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 50
                cfg.stage_b_dense_duty_raw_veto_query_scope = "wrong-scope"
                with self.assertRaisesRegex(
                    ValueError, "word-veto probe config contract is incomplete"
                ):
                    combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                        args, cfg
                    )

    def test_v15_v17_u50_terminal_checkpoints_are_diagnostic_only(self):
        specs = (
            (
                "v15",
                "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
                "carrier_affine_probe_u0050_20260731.py",
            ),
            (
                "v17",
                "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
                "tail_carrier_probe_u0050_20260731.py",
            ),
        )
        source_closure = {"sha256": "a" * 64}
        for label, filename in specs:
            with self.subTest(revision=label), tempfile.TemporaryDirectory() as temporary:
                config_path = Path("config/ablations", filename).resolve()
                cfg = combined_eval.SLConfig.fromfile(str(config_path))
                train_dir = Path(temporary) / "train"
                train_dir.mkdir()
                canonical = train_dir / "checkpoint_iter.pth"
                snapshot = Path(temporary) / "snapshot.pth"
                canonical.write_bytes(f"same-{label}-u50-checkpoint".encode("ascii"))
                snapshot.write_bytes(f"same-{label}-u50-checkpoint".encode("ascii"))
                saved_args = dict(cfg._cfg_dict)
                saved_args.update(
                    {
                        "output_dir": str(train_dir),
                        "max_train_iters": 50,
                        "stage_b_dense_duty_execution_scope": "probe",
                        "stage_b_dense_duty_source_closure": source_closure,
                        "stage_b_dense_duty_runtime_audit": {
                            "successful_optimizer_steps": 50,
                            "optimizer_step_boundaries": 50,
                            "amp_skipped_optimizer_steps": 0,
                            "nonfinite_gradient_boundaries": 0,
                            "zero_gradient_successful_steps": 0,
                            "max_active_grad_norm_preclip": 1.0,
                            "peak_reserved_bytes": 1,
                        },
                    }
                )
                payload = {
                    "args": saved_args,
                    "optimizer_updates": 50,
                    "checkpoint_reason": "max_train_iters",
                }

                def resume(value, _args, *, checkpoint_path):
                    return {
                        "phase": "confidence",
                        "optimizer_updates": value["optimizer_updates"],
                        "checkpoint_reason": value["checkpoint_reason"],
                        "rank_handoff": {"status": "passed"},
                    }

                def audit(value, *, checkpoint_path):
                    return {
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": value["optimizer_updates"],
                        "lineage": {
                            "no_stage_b_teacher": True,
                            "execution_scope": "probe",
                        },
                    }

                with (
                    mock.patch(
                        "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload",
                        side_effect=resume,
                    ) as strict_resume,
                    mock.patch(
                        "util.stage_b_dense_duty_audit.validate_evaluation_checkpoint_payload",
                        return_value={
                            "status": "passed",
                            "phase": "confidence",
                            "optimizer_updates": 50,
                            "checkpoint_reason": "max_train_iters",
                            "rank_handoff": {"status": "passed"},
                        },
                    ) as terminal_evaluation,
                    mock.patch(
                        "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                        side_effect=audit,
                    ),
                    mock.patch(
                        "util.stage_b_dense_duty_audit.build_source_closure",
                        return_value=source_closure,
                    ),
                    mock.patch(
                        "util.stage_b_dense_duty_audit.validate_source_closure",
                        side_effect=lambda value: value,
                    ),
                    mock.patch.object(
                        ref_eval,
                        "_validate_historical_dense_duty_u50_source_archive",
                        return_value={"status": "verified", "revision": label},
                    ) as historical_archive,
                ):
                    result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )
                    terminal_evaluation.assert_called_once()
                    strict_resume.assert_not_called()
                    historical_archive.assert_called_once()

                    payload["optimizer_updates"] = 49
                    payload["checkpoint_reason"] = "signal"
                    saved_args["stage_b_dense_duty_runtime_audit"].update(
                        successful_optimizer_steps=49,
                        optimizer_step_boundaries=49,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, f"terminal U50 {label} probe"
                    ):
                        ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                            payload,
                            cfg,
                            checkpoint_path=snapshot,
                        )

                self.assertTrue(result["terminal_checkpoint"])
                self.assertTrue(result["diagnostic_only"])
                self.assertFalse(result["formal_gate_eligible"])
                self.assertEqual(result["expected_optimizer_updates"], 50)
                self.assertEqual(result["remaining_optimizer_updates"], 0)
                self.assertEqual(result["source_closure"], source_closure)
                self.assertEqual(
                    result["historical_source_archive"]["status"],
                    "verified",
                )

    def test_v15_v17_exact_source_archives_match_saved_closures(self):
        for revision, expected_count in (
            ("word_veto_gated_pool_carrier_affine_v15", 138),
            ("word_veto_gated_pool_tail_carrier_v17", 139),
        ):
            with self.subTest(revision=revision):
                spec = ref_eval._HISTORICAL_DENSE_DUTY_U50_SOURCES[revision]
                checkpoint = Path(spec["checkpoint"]).resolve(strict=True)
                contract_path = (
                    checkpoint.parent / "stage_b_dense_duty_training_contract.json"
                )
                contract = json.loads(contract_path.read_text(encoding="ascii"))
                source_closure = contract["values"][
                    "stage_b_dense_duty_source_closure"
                ]
                result = ref_eval._validate_historical_dense_duty_u50_source_archive(
                    revision=revision,
                    config_path=Path(spec["config"]).resolve(strict=True),
                    checkpoint_path=checkpoint,
                    checkpoint_sha256=spec["checkpoint_sha256"],
                    source_closure=source_closure,
                )
                self.assertEqual(result["status"], "verified")
                self.assertEqual(result["file_count"], expected_count)

                drifted_closure = dict(source_closure)
                drifted_closure["sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "source closure drifted"):
                    ref_eval._validate_historical_dense_duty_u50_source_archive(
                        revision=revision,
                        config_path=Path(spec["config"]).resolve(strict=True),
                        checkpoint_path=checkpoint,
                        checkpoint_sha256=spec["checkpoint_sha256"],
                        source_closure=drifted_closure,
                    )

    def test_v18_u100_cli_contract_is_fixed_and_fail_closed(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_probe_u0100_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=False,
                partial_dense_duty_confidence_diagnostic=True,
                config=str(config_path),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["confidence-v18-u100-terminal.pth"],
                tn_jsonl=str(
                    Path(
                        "data/eval_manifests/"
                        "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                        "semantic_stageb_union_image_disjoint_manifest.jsonl"
                    ).resolve()
                ),
                tn_splits=["refcocop_val", "refcocog_umd_val"],
                skip_tn=False,
                skip_ref=True,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                threshold_tprs=[0.75, 0.9, 0.95],
                score_thresholds=[0.5],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )

            cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 50
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )
            cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 100
            cfg.stage_b_dense_duty_raw_veto_query_scope = "wrong-scope"
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

    def test_v18_u100_terminal_checkpoint_is_diagnostic_only(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_probe_u0100_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        source_closure = {"sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary) / "train"
            train_dir.mkdir()
            canonical = train_dir / "checkpoint_iter.pth"
            snapshot = Path(temporary) / "snapshot.pth"
            canonical.write_bytes(b"same-v18-u100-checkpoint")
            snapshot.write_bytes(b"same-v18-u100-checkpoint")
            saved_args = dict(cfg._cfg_dict)
            saved_args.update(
                {
                    "output_dir": str(train_dir),
                    "max_train_iters": 100,
                    "stage_b_dense_duty_execution_scope": "probe",
                    "stage_b_dense_duty_source_closure": source_closure,
                    "stage_b_dense_duty_runtime_audit": {
                        "successful_optimizer_steps": 100,
                        "optimizer_step_boundaries": 100,
                        "amp_skipped_optimizer_steps": 0,
                        "nonfinite_gradient_boundaries": 0,
                        "zero_gradient_successful_steps": 0,
                        "max_active_grad_norm_preclip": 1.0,
                        "peak_reserved_bytes": 1,
                    },
                }
            )
            payload = {
                "args": saved_args,
                "optimizer_updates": 100,
                "checkpoint_reason": "max_train_iters",
            }

            def resume(value, _args, *, checkpoint_path):
                return {
                    "phase": "confidence",
                    "optimizer_updates": value["optimizer_updates"],
                    "checkpoint_reason": value["checkpoint_reason"],
                    "rank_handoff": {"status": "passed"},
                }

            def audit(value, *, checkpoint_path):
                return {
                    "status": "passed",
                    "phase": "confidence",
                    "optimizer_updates": value["optimizer_updates"],
                    "lineage": {
                        "no_stage_b_teacher": True,
                        "execution_scope": "probe",
                    },
                }

            with (
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload",
                    side_effect=resume,
                ) as strict_resume,
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_evaluation_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 100,
                        "checkpoint_reason": "max_train_iters",
                        "rank_handoff": {"status": "passed"},
                    },
                ) as terminal_evaluation,
                mock.patch(
                    "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                    side_effect=audit,
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_source_closure",
                    side_effect=lambda value: value,
                ),
                mock.patch.object(
                    ref_eval,
                    "_validate_historical_dense_duty_u100_source_archive",
                    return_value={
                        "status": "verified",
                        "revision": "v18",
                        "file_count": 140,
                    },
                ) as historical_archive,
            ):
                result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    payload,
                    cfg,
                    checkpoint_path=snapshot,
                )
                terminal_evaluation.assert_called_once()
                strict_resume.assert_not_called()
                historical_archive.assert_called_once()

                payload["optimizer_updates"] = 99
                payload["checkpoint_reason"] = "signal"
                saved_args["stage_b_dense_duty_runtime_audit"].update(
                    successful_optimizer_steps=99,
                    optimizer_step_boundaries=99,
                )
                with self.assertRaisesRegex(RuntimeError, "terminal U100 v18 probe"):
                    ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )

        self.assertTrue(result["terminal_checkpoint"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["formal_gate_eligible"])
        self.assertEqual(result["expected_optimizer_updates"], 100)
        self.assertEqual(result["remaining_optimizer_updates"], 0)
        self.assertEqual(result["historical_source_archive"]["file_count"], 140)

    def test_v18_u100_exact_source_archive_matches_saved_closure(self):
        revision = "word_veto_gated_pool_tail_paired_v18"
        spec = ref_eval._HISTORICAL_DENSE_DUTY_U100_SOURCES[revision]
        checkpoint = Path(spec["checkpoint"]).resolve(strict=True)
        contract_path = checkpoint.parent / "stage_b_dense_duty_training_contract.json"
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        source_closure = contract["values"]["stage_b_dense_duty_source_closure"]
        result = ref_eval._validate_historical_dense_duty_u100_source_archive(
            revision=revision,
            config_path=Path(spec["config"]).resolve(strict=True),
            checkpoint_path=checkpoint,
            checkpoint_sha256=spec["checkpoint_sha256"],
            source_closure=source_closure,
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["optimizer_updates"], 100)
        self.assertEqual(result["file_count"], 140)

        drifted_closure = dict(source_closure)
        drifted_closure["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source closure drifted"):
            ref_eval._validate_historical_dense_duty_u100_source_archive(
                revision=revision,
                config_path=Path(spec["config"]).resolve(strict=True),
                checkpoint_path=checkpoint,
                checkpoint_sha256=spec["checkpoint_sha256"],
                source_closure=drifted_closure,
            )

    def test_v19_rank_channel_u50_cli_contract_is_fixed_and_fail_closed(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_rank_channel_probe_u0050_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=False,
                partial_dense_duty_confidence_diagnostic=True,
                config=str(config_path),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["confidence-v19-u50-terminal.pth"],
                tn_jsonl=str(
                    Path(
                        "data/eval_manifests/"
                        "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                        "semantic_stageb_union_image_disjoint_manifest.jsonl"
                    ).resolve()
                ),
                tn_splits=["refcocop_val", "refcocog_umd_val"],
                skip_tn=False,
                skip_ref=True,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                threshold_tprs=[0.75, 0.9, 0.95],
                score_thresholds=[0.5],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )

            cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 100
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )
            cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 50
            cfg.stage_b_dense_duty_confidence_rank_evidence_contract = (
                "zero_init_carrier_token_rank_affine_v5"
            )
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

    def test_v19_rank_channel_u50_terminal_checkpoint_is_diagnostic_only(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_rank_channel_probe_u0050_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        source_closure = {"sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            train_dir = Path(temporary) / "train"
            train_dir.mkdir()
            canonical = train_dir / "checkpoint_iter.pth"
            snapshot = Path(temporary) / "snapshot.pth"
            canonical.write_bytes(b"same-v19-u50-checkpoint")
            snapshot.write_bytes(b"same-v19-u50-checkpoint")
            saved_args = dict(cfg._cfg_dict)
            saved_args.update(
                {
                    "output_dir": str(train_dir),
                    "max_train_iters": 50,
                    "stage_b_dense_duty_execution_scope": "probe",
                    "stage_b_dense_duty_source_closure": source_closure,
                    "stage_b_dense_duty_runtime_audit": {
                        "successful_optimizer_steps": 50,
                        "optimizer_step_boundaries": 50,
                        "amp_skipped_optimizer_steps": 0,
                        "nonfinite_gradient_boundaries": 0,
                        "zero_gradient_successful_steps": 0,
                        "max_active_grad_norm_preclip": 1.0,
                        "peak_reserved_bytes": 1,
                    },
                }
            )
            payload = {
                "args": saved_args,
                "optimizer_updates": 50,
                "checkpoint_reason": "max_train_iters",
            }

            def resume(value, _args, *, checkpoint_path):
                return {
                    "phase": "confidence",
                    "optimizer_updates": value["optimizer_updates"],
                    "checkpoint_reason": value["checkpoint_reason"],
                    "rank_handoff": {"status": "passed"},
                }

            def audit(value, *, checkpoint_path):
                return {
                    "status": "passed",
                    "phase": "confidence",
                    "optimizer_updates": value["optimizer_updates"],
                    "lineage": {
                        "no_stage_b_teacher": True,
                        "execution_scope": "probe",
                    },
                }

            with (
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_strict_resume_checkpoint_payload",
                    side_effect=resume,
                ) as strict_resume,
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_evaluation_checkpoint_payload",
                    return_value={
                        "status": "passed",
                        "phase": "confidence",
                        "optimizer_updates": 50,
                        "checkpoint_reason": "max_train_iters",
                        "rank_handoff": {"status": "passed"},
                    },
                ) as terminal_evaluation,
                mock.patch(
                    "util.stage_b_dense_duty_audit.audit_checkpoint_payload",
                    side_effect=audit,
                ),
                mock.patch(
                    "util.stage_b_dense_duty_audit.validate_source_closure",
                    side_effect=lambda value: value,
                ),
                mock.patch.object(
                    ref_eval,
                    "_validate_historical_dense_duty_u50_source_archive",
                    return_value={
                        "status": "verified",
                        "revision": "v19",
                        "file_count": 141,
                    },
                ) as historical_archive,
            ):
                result = ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    payload,
                    cfg,
                    checkpoint_path=snapshot,
                )
                terminal_evaluation.assert_called_once()
                strict_resume.assert_not_called()
                historical_archive.assert_called_once()

                payload["optimizer_updates"] = 49
                payload["checkpoint_reason"] = "signal"
                saved_args["stage_b_dense_duty_runtime_audit"].update(
                    successful_optimizer_steps=49,
                    optimizer_step_boundaries=49,
                )
                with self.assertRaisesRegex(RuntimeError, "terminal U50 v19 probe"):
                    ref_eval._validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                        payload,
                        cfg,
                        checkpoint_path=snapshot,
                    )

        self.assertTrue(result["terminal_checkpoint"])
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["formal_gate_eligible"])
        self.assertEqual(result["expected_optimizer_updates"], 50)
        self.assertEqual(result["remaining_optimizer_updates"], 0)
        self.assertEqual(result["historical_source_archive"]["file_count"], 141)

    def test_v19_rank_channel_exact_source_archive_matches_saved_closure(self):
        revision = "word_veto_gated_pool_tail_paired_rank_channel_v19"
        spec = ref_eval._HISTORICAL_DENSE_DUTY_U50_SOURCES[revision]
        checkpoint = Path(spec["checkpoint"]).resolve(strict=True)
        contract_path = checkpoint.parent / "stage_b_dense_duty_training_contract.json"
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        source_closure = contract["values"]["stage_b_dense_duty_source_closure"]
        result = ref_eval._validate_historical_dense_duty_u50_source_archive(
            revision=revision,
            config_path=Path(spec["config"]).resolve(strict=True),
            checkpoint_path=checkpoint,
            checkpoint_sha256=spec["checkpoint_sha256"],
            source_closure=source_closure,
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["optimizer_updates"], 50)
        self.assertEqual(result["file_count"], 141)

        drifted_closure = dict(source_closure)
        drifted_closure["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source closure drifted"):
            ref_eval._validate_historical_dense_duty_u50_source_archive(
                revision=revision,
                config_path=Path(spec["config"]).resolve(strict=True),
                checkpoint_path=checkpoint,
                checkpoint_sha256=spec["checkpoint_sha256"],
                source_closure=drifted_closure,
            )

    def test_v20_signed_rank_pool_u50_cli_contract_is_fixed_and_fail_closed(self):
        config_path = Path(
            "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_signed_rank_pool_probe_u0050_20260731.py"
        ).resolve()
        cfg = combined_eval.SLConfig.fromfile(str(config_path))
        with tempfile.TemporaryDirectory() as temporary:
            args = types.SimpleNamespace(
                partial_dense_duty_rank_diagnostic=False,
                partial_dense_duty_confidence_diagnostic=True,
                config=str(config_path),
                output_dir=str(Path(temporary) / "fresh"),
                ckpts=["confidence-v20-u50-terminal.pth"],
                tn_jsonl=str(
                    Path(
                        "data/eval_manifests/"
                        "stageb_vlm_verified_strict_ann_umd_val_20260711/"
                        "semantic_stageb_union_image_disjoint_manifest.jsonl"
                    ).resolve()
                ),
                tn_splits=["refcocop_val", "refcocog_umd_val"],
                skip_tn=False,
                skip_ref=True,
                ref_splits=list(REF_SPLITS),
                device="cuda:0",
                batch_size=16,
                num_workers=4,
                seed=42,
                amp=True,
                topk=[1],
                threshold_tprs=[0.75, 0.9, 0.95],
                score_thresholds=[0.5],
                max_ref_batches=0,
                max_tn_batches=0,
                no_per_example_records=False,
                screen_calibration_manifest=False,
                direct_prebuilt_tn=False,
                category_gate_max_gaps=None,
                category_gate_include_base_expert=False,
                candidate_count_control=0,
                holdout_level="none",
                exclude_train_jsonl=[],
            )
            combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                args, cfg
            )

            cfg.stage_b_dense_duty_confidence_pool_feature_contract = (
                "patch_statistics_only_v1"
            )
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )
            cfg.stage_b_dense_duty_confidence_pool_feature_contract = (
                "detached_rank_query_plus_patch_statistics_signed_residual_v2"
            )
            cfg.stage_b_dense_duty_confidence_expected_optimizer_updates = 100
            with self.assertRaisesRegex(
                ValueError, "word-veto probe config contract is incomplete"
            ):
                combined_eval._validate_partial_dense_duty_confidence_diagnostic_args(
                    args, cfg
                )

    def test_v20_signed_rank_pool_exact_source_archive_matches_saved_closure(self):
        revision = "word_veto_gated_pool_tail_paired_signed_rank_pool_v20"
        spec = ref_eval._HISTORICAL_DENSE_DUTY_U50_SOURCES[revision]
        checkpoint = Path(spec["checkpoint"]).resolve(strict=True)
        contract_path = checkpoint.parent / "stage_b_dense_duty_training_contract.json"
        contract = json.loads(contract_path.read_text(encoding="ascii"))
        source_closure = contract["values"]["stage_b_dense_duty_source_closure"]
        result = ref_eval._validate_historical_dense_duty_u50_source_archive(
            revision=revision,
            config_path=Path(spec["config"]).resolve(strict=True),
            checkpoint_path=checkpoint,
            checkpoint_sha256=spec["checkpoint_sha256"],
            source_closure=source_closure,
        )
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["optimizer_updates"], 50)
        self.assertEqual(result["file_count"], 142)

        drifted_closure = dict(source_closure)
        drifted_closure["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source closure drifted"):
            ref_eval._validate_historical_dense_duty_u50_source_archive(
                revision=revision,
                config_path=Path(spec["config"]).resolve(strict=True),
                checkpoint_path=checkpoint,
                checkpoint_sha256=spec["checkpoint_sha256"],
                source_closure=drifted_closure,
            )


if __name__ == "__main__":
    unittest.main()
