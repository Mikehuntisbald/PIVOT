import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from engine import (
    _clip_stage_b_dense_duty_grad_norms,
    _record_stage_b_dense_duty_runtime_audit,
    _select_dense_duty_confidence_loss_logits,
    _sum_weighted_training_losses,
)
from main import _prepare_stage_b_dense_duty_state_fingerprint

from util.stage_b_dense_duty_audit import (
    CHECKPOINT_AUDIT_SCHEMA,
    FINGERPRINT_ARG,
    SOURCE_CLOSURE_ARG,
    STRICT_RESUME_CHECKPOINT_NAME,
    TRAINING_CONTRACT_ARG,
    _RESUME_CONTRACT_KEYS,
    _validate_candidate_sample_runtime_audit,
    audit_checkpoint_payload,
    build_source_closure,
    build_training_contract,
    fingerprint_model,
    validate_confidence_adapter_rank_source_audit,
    validate_evaluation_checkpoint_payload,
    validate_rank_handoff_audit,
    validate_resume_training_contract,
    validate_strict_resume_checkpoint_payload,
    write_json_atomic,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_closure(phase: str):
    return build_source_closure(
        _REPO_ROOT
        / f"config/ablations/cfg_stageb_dense_duty_{phase}_20260728.py",
        repo_root=_REPO_ROOT,
    )


def _drifted_code_closure():
    closure = copy.deepcopy(_source_closure("confidence")["code"])
    closure["files"][0]["sha256"] = "0" * 64
    closure["sha256"] = hashlib.sha256(
        json.dumps(
            closure["files"],
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return closure


class _TinyDenseDutyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_a = nn.Linear(3, 3)
        self.stage_b_fixed_text_scorer = nn.Module()
        self.stage_b_fixed_text_scorer.rank_tower = nn.Linear(3, 2)
        self.stage_b_fixed_text_scorer.confidence_tower = nn.Linear(3, 2)
        self.stage_b_fixed_text_scorer.confidence_pool = nn.Linear(2, 1)

    def set_phase(self, phase: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        owner = (
            self.stage_b_fixed_text_scorer.rank_tower
            if phase == "rank"
            else self.stage_b_fixed_text_scorer
        )
        if phase == "rank":
            parameters = owner.parameters()
        else:
            parameters = list(
                self.stage_b_fixed_text_scorer.confidence_tower.parameters()
            ) + list(self.stage_b_fixed_text_scorer.confidence_pool.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)


class _TinySplitConfidenceScorer(nn.Module):
    def __init__(
        self,
        ownership_mode: str = "complete",
        head_contract: str = "split_token_veto_global_absolute_v2",
    ):
        super().__init__()
        self.token_veto = nn.Parameter(torch.zeros(2))
        self.global_absolute = nn.Parameter(torch.zeros(2))
        self.global_veto = (
            nn.Parameter(torch.zeros(2))
            if head_contract == "split_token_veto_global_trust_veto_v4"
            else None
        )
        self.ownership_mode = ownership_mode
        self.confidence_adapter = SimpleNamespace(
            head_gradient_contract=head_contract
        )

    def token_veto_parameters(self):
        return (self.token_veto,)

    def global_absolute_parameters(self):
        if self.ownership_mode == "overlap":
            return (self.token_veto, self.global_absolute)
        if self.ownership_mode == "incomplete":
            return ()
        return (self.global_absolute,) + (
            (self.global_veto,) if self.global_veto is not None else ()
        )

    def global_trust_parameters(self):
        return (self.global_absolute,)

    def global_veto_parameters(self):
        return (self.global_veto,) if self.global_veto is not None else ()

    def expected_live_confidence_parameter_tensor_counts(self):
        if self.confidence_adapter.head_gradient_contract not in {
            "split_token_veto_fulltext_global_absolute_v7",
            "split_token_veto_local_candidate_global_absolute_v8",
        }:
            raise RuntimeError("tiny live contract is defined only for V53/V55")
        return {"token_veto": 1, "global_absolute": 1}


class _TinySplitDenseDutyModel(nn.Module):
    def __init__(
        self,
        ownership_mode: str = "complete",
        head_contract: str = "split_token_veto_global_absolute_v2",
    ):
        super().__init__()
        self.stage_b_fixed_text_scorer = _TinySplitConfidenceScorer(
            ownership_mode,
            head_contract,
        )


class _TinyV52ConfidenceScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_veto = nn.Parameter(torch.zeros(2))
        self.candidate_absolute = nn.Parameter(torch.zeros(2))
        self.sample_calibrator = nn.Parameter(torch.zeros(2))
        self.confidence_adapter = SimpleNamespace(
            head_gradient_contract=(
                "split_token_veto_candidate_absolute_sample_calibrator_v6"
            )
        )

    def token_veto_parameters(self):
        return (self.token_veto,)

    def candidate_absolute_parameters(self):
        return (self.candidate_absolute,)

    def sample_calibrator_parameters(self):
        return (self.sample_calibrator,)

    def expected_live_confidence_parameter_tensor_counts(self):
        return {
            "token_veto": 1,
            "candidate_absolute": 1,
            "sample_calibrator": 1,
        }


class _TinyV52DenseDutyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_b_fixed_text_scorer = _TinyV52ConfidenceScorer()


def _v52_clip_stats(max_norm: float = 1.0):
    model = _TinyV52DenseDutyModel()
    scorer = model.stage_b_fixed_text_scorer
    scorer.token_veto.grad = torch.tensor([3.0, 4.0])
    scorer.candidate_absolute.grad = torch.tensor([6.0, 8.0])
    scorer.sample_calibrator.grad = torch.tensor([0.0, 2.0])
    return model, _clip_stage_b_dense_duty_grad_norms(model, max_norm)


def _healthy_v52_runtime_audit(steps: int = 400):
    runtime = {
        "schema": "pivot.stageb.dense_duty_runtime_audit/v1",
        "clip_contract_schema": (
            "pivot.stageb.dense_duty_three_owner_clip_contract/v1"
        ),
        "clip_contract_checked_steps": steps,
        "clip_contract_max_norm": 0.1,
        "clip_contract_tolerance": 1.0e-5,
        "max_active_pre_decomposition_residual": 0.0,
        "max_active_post_decomposition_residual": 0.0,
        "max_owner_clip_residual": 0.0,
        "max_active_monotonic_residual": 0.0,
        "owner_clip_violation_steps": 0,
        "active_pre_decomposition_violation_steps": 0,
        "active_post_decomposition_violation_steps": 0,
        "live_tensor_count_violation_steps": 0,
        "active_monotonic_violation_steps": 0,
    }
    for owner, count in (
        ("token_veto", 21),
        ("candidate_absolute", 39),
        ("sample_calibrator", 7),
    ):
        runtime[f"last_{owner}_grad_norm_preclip"] = 1.0
        runtime[f"max_{owner}_grad_norm_preclip"] = 2.0
        runtime[f"nonfinite_{owner}_gradient_boundaries"] = 0
        runtime[f"zero_{owner}_gradient_successful_steps"] = 0
        runtime[f"expected_{owner}_tensor_count"] = count
        runtime[f"last_observed_{owner}_tensor_count"] = count
    return runtime


def _payload(model: _TinyDenseDutyModel, initial, phase="rank"):
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer_updates": 1,
        "args": {
            "stage_b_dense_duty": True,
            "stage_b_dense_duty_phase": phase,
            FINGERPRINT_ARG: initial,
            "stage_b_dense_duty_lineage_audit": {"status": "test"},
        },
    }


def _strict_resume_fixture(directory: Path, *, phase="rank"):
    model = _TinyDenseDutyModel()
    model.set_phase(phase)
    initial = fingerprint_model(model, phase=phase)
    values = {key: 1 for key in _RESUME_CONTRACT_KEYS}
    values.update(
        {
            "stage_b_dense_duty": True,
            "stage_b_dense_duty_phase": phase,
            "stage_b_v22_train_phase": phase,
            "stage_b_dense_duty_execution_scope": "probe",
            "stage_b_dense_duty_rank_dataset_config_sha256": "e" * 64,
            "gradient_accumulation_steps": 4,
            "max_train_iters": 10,
            "iter_checkpoint_interval": 2,
            "world_size": 1,
            "distributed": False,
            "find_unused_params": False,
            SOURCE_CLOSURE_ARG: _source_closure(phase),
        }
    )
    saved_args = dict(values)
    saved_args[FINGERPRINT_ARG] = initial
    saved_args["stage_b_dense_duty_runtime_audit"] = {
        "schema": "pivot.stageb.dense_duty_runtime_audit/v1",
        "optimizer_step_boundaries": 2,
        "successful_optimizer_steps": 2,
    }
    if phase == "confidence":
        saved_args["stage_b_dense_duty_rank_source_checkpoint_audit"] = {
            "schema": CHECKPOINT_AUDIT_SCHEMA,
            "status": "passed",
            "phase": "rank",
            "optimizer_updates": 4,
            "ownership": {
                "active_changed": True,
                "frozen_unchanged": True,
            },
            "lineage": {
                "execution_scope": "probe",
                "no_stage_b_teacher": True,
                "dataset_config": {"sha256": "e" * 64},
            },
            "source_closure": _source_closure("rank"),
        }
    saved_args[TRAINING_CONTRACT_ARG] = build_training_contract(saved_args)
    rng_state = {
        "python": (3, (), None),
        "numpy": ("MT19937", (), 0, 0, 0.0),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    payload = {
        "model": copy.deepcopy(model.state_dict()),
        "criterion": {},
        "optimizer": {},
        "lr_scheduler": {},
        "scaler": {},
        "epoch": 0,
        "iteration": 8,
        "optimizer_updates": 2,
        "epoch_finished": False,
        "rng_state": copy.deepcopy(rng_state),
        "epoch_rng_state": copy.deepcopy(rng_state),
        "args": saved_args,
        "checkpoint_reason": "interval",
    }
    checkpoint_path = directory / STRICT_RESUME_CHECKPOINT_NAME
    checkpoint_path.touch()
    args = SimpleNamespace(**values, output_dir=str(directory))
    return args, payload, checkpoint_path


class DenseDutyCheckpointAuditTest(unittest.TestCase):
    def test_confidence_adapter_rank_source_rejects_empty_lineage(self):
        values = {
            "stage_b_v22_score_ownership": (
                "rank_tower_stopgrad_token_adapter_two_phase"
            ),
            "stage_b_dense_duty_rank_dataset_config_sha256": "e" * 64,
            "stage_b_dense_duty_rank_source_optimizer_updates": 6551,
        }
        with self.assertRaisesRegex(RuntimeError, "invalid rank handoff audit"):
            validate_confidence_adapter_rank_source_audit({}, values)

        rank_audit = {
            "schema": CHECKPOINT_AUDIT_SCHEMA,
            "status": "passed",
            "phase": "rank",
            "optimizer_updates": 6551,
            "ownership": {
                "active_changed": True,
                "frozen_unchanged": True,
            },
            "lineage": {
                "execution_scope": "formal",
                "no_stage_b_teacher": True,
                "dataset_config": {"sha256": "e" * 64},
            },
        }
        self.assertEqual(
            validate_confidence_adapter_rank_source_audit(rank_audit, values),
            rank_audit,
        )

    def test_zero_weight_loss_graph_is_not_traversed(self):
        value = torch.tensor(2.0, requires_grad=True)
        finite_loss = value.square()
        zero_with_nonfinite_derivative = torch.sqrt(value * 0.0)
        loss = _sum_weighted_training_losses(
            {
                "finite": finite_loss,
                "disabled": zero_with_nonfinite_derivative,
            },
            {"finite": 1.0, "disabled": 0.0},
        )
        loss.backward()
        self.assertTrue(torch.isfinite(value.grad).item())
        self.assertEqual(value.grad.item(), 4.0)

    def test_rank_change_with_frozen_state_unchanged_passes(self):
        model = _TinyDenseDutyModel()
        model.set_phase("rank")
        initial = fingerprint_model(model, phase="rank")
        with torch.no_grad():
            model.stage_b_fixed_text_scorer.rank_tower.weight.add_(0.25)
        audit = audit_checkpoint_payload(_payload(model, initial))
        self.assertEqual(audit["status"], "passed")
        self.assertTrue(audit["ownership"]["active_changed"])
        self.assertTrue(audit["ownership"]["frozen_unchanged"])

    def test_frozen_stage_a_change_is_rejected(self):
        model = _TinyDenseDutyModel()
        model.set_phase("rank")
        initial = fingerprint_model(model, phase="rank")
        with torch.no_grad():
            model.stage_b_fixed_text_scorer.rank_tower.weight.add_(0.25)
            model.stage_a.weight.add_(0.25)
        with self.assertRaisesRegex(RuntimeError, "frozen state changed"):
            audit_checkpoint_payload(_payload(model, initial))

    def test_reported_update_without_active_change_is_rejected(self):
        model = _TinyDenseDutyModel()
        model.set_phase("rank")
        initial = fingerprint_model(model, phase="rank")
        with self.assertRaisesRegex(RuntimeError, "active state did not change"):
            audit_checkpoint_payload(_payload(model, initial))

    def test_confidence_owner_includes_pool_but_not_rank(self):
        model = _TinyDenseDutyModel()
        model.set_phase("confidence")
        initial = fingerprint_model(model, phase="confidence")
        names = initial["active_parameter_names"]
        self.assertTrue(any("confidence_tower" in name for name in names))
        self.assertTrue(any("confidence_pool" in name for name in names))
        self.assertFalse(any("rank_tower" in name for name in names))
        with torch.no_grad():
            model.stage_b_fixed_text_scorer.confidence_pool.bias.add_(0.5)
        audit = audit_checkpoint_payload(
            _payload(model, initial, phase="confidence")
        )
        self.assertEqual(audit["phase"], "confidence")

    def test_atomic_json_writer_publishes_complete_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            write_json_atomic(output, {"status": "passed"})
            self.assertEqual(output.read_text(encoding="ascii"), '{\n  "status": "passed"\n}\n')
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_runtime_audit_distinguishes_signal_and_zero_gradient_steps(self):
        model = _TinyDenseDutyModel()
        model.set_phase("rank")
        loss = model.stage_b_fixed_text_scorer.rank_tower(
            torch.ones(1, 3)
        ).sum()
        loss.backward()
        signal_stats = _clip_stage_b_dense_duty_grad_norms(model, 0.1)
        args = SimpleNamespace(stage_b_dense_duty=True)
        _record_stage_b_dense_duty_runtime_audit(
            args,
            torch.device("cpu"),
            optimizer_step_boundary=True,
            optimizer_step_succeeded=True,
            branch_grad_norms=signal_stats,
            amp_scale=64.0,
        )
        self.assertGreater(
            args.stage_b_dense_duty_runtime_audit[
                "last_active_grad_norm_preclip"
            ],
            0.0,
        )

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        zero_stats = _clip_stage_b_dense_duty_grad_norms(model, 0.1)
        _record_stage_b_dense_duty_runtime_audit(
            args,
            torch.device("cpu"),
            optimizer_step_boundary=True,
            optimizer_step_succeeded=True,
            branch_grad_norms=zero_stats,
            amp_scale=32.0,
        )
        audit = args.stage_b_dense_duty_runtime_audit
        self.assertEqual(audit["successful_optimizer_steps"], 2)
        self.assertEqual(audit["zero_gradient_successful_steps"], 1)
        self.assertEqual(audit["amp_skipped_optimizer_steps"], 0)
        self.assertEqual(audit["min_amp_scale"], 32.0)

    def test_split_confidence_heads_are_clipped_independently(self):
        model = _TinySplitDenseDutyModel()
        scorer = model.stage_b_fixed_text_scorer
        scorer.token_veto.grad = torch.tensor([3.0, 4.0])
        scorer.global_absolute.grad = torch.tensor([6.0, 8.0])

        stats = _clip_stage_b_dense_duty_grad_norms(model, 1.0)

        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_token_veto_preclip"], 5.0
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_absolute_preclip"], 10.0
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_token_veto_postclip"], 1.0, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_absolute_postclip"],
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_active_postclip"],
            2.0**0.5,
            places=5,
        )
        self.assertEqual(stats["grad_tensor_count_dense_duty_token_veto"], 1.0)
        self.assertEqual(
            stats["grad_tensor_count_dense_duty_global_absolute"], 1.0
        )

    def test_v53_two_owners_have_exact_independent_clip_audit(self):
        model = _TinySplitDenseDutyModel(
            head_contract="split_token_veto_fulltext_global_absolute_v7"
        )
        scorer = model.stage_b_fixed_text_scorer
        scorer.token_veto.grad = torch.tensor([3.0, 4.0])
        scorer.global_absolute.grad = torch.tensor([6.0, 8.0])

        stats = _clip_stage_b_dense_duty_grad_norms(model, 1.0)
        self.assertEqual(stats["dense_duty_clip_contract_checked"], 1)
        self.assertEqual(stats["dense_duty_clip_contract_owner_count"], 2)
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_token_veto_postclip"], 1.0, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_absolute_postclip"], 1.0, places=5
        )
        for owner in ("token_veto", "global_absolute"):
            self.assertEqual(
                stats[f"dense_duty_clip_contract_expected_{owner}_tensor_count"],
                1,
            )
            self.assertEqual(
                stats[f"dense_duty_clip_contract_observed_{owner}_tensor_count"],
                1,
            )
        for suffix in (
            "live_tensor_count",
            "pre_decomposition",
            "post_decomposition",
            "owner_clip",
            "active_monotonic",
        ):
            self.assertEqual(
                stats[f"dense_duty_clip_contract_{suffix}_violation"], 0
            )

        args = SimpleNamespace(stage_b_dense_duty=True)
        _record_stage_b_dense_duty_runtime_audit(
            args,
            torch.device("cpu"),
            optimizer_step_boundary=True,
            optimizer_step_succeeded=True,
            branch_grad_norms=stats,
        )
        audit = args.stage_b_dense_duty_runtime_audit
        self.assertEqual(
            audit["clip_contract_schema"],
            "pivot.stageb.dense_duty_two_owner_clip_contract/v1",
        )
        self.assertEqual(audit["expected_token_veto_tensor_count"], 1)
        self.assertEqual(audit["expected_global_absolute_tensor_count"], 1)

    def test_v55_two_owners_use_the_same_exact_independent_clip_audit(self):
        model = _TinySplitDenseDutyModel(
            head_contract="split_token_veto_local_candidate_global_absolute_v8"
        )
        scorer = model.stage_b_fixed_text_scorer
        scorer.token_veto.grad = torch.tensor([3.0, 4.0])
        scorer.global_absolute.grad = torch.tensor([6.0, 8.0])

        stats = _clip_stage_b_dense_duty_grad_norms(model, 1.0)

        self.assertEqual(stats["dense_duty_clip_contract_checked"], 1)
        self.assertEqual(stats["dense_duty_clip_contract_owner_count"], 2)
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_token_veto_postclip"], 1.0, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_absolute_postclip"], 1.0, places=5
        )
        self.assertEqual(
            stats["dense_duty_clip_contract_expected_token_veto_tensor_count"],
            1,
        )
        self.assertEqual(
            stats[
                "dense_duty_clip_contract_expected_global_absolute_tensor_count"
            ],
            1,
        )

    def test_split_joint_clip_preserves_head_stats_and_clips_union_once(self):
        model = _TinySplitDenseDutyModel(
            head_contract="split_token_veto_global_absolute_joint_clip_v3"
        )
        scorer = model.stage_b_fixed_text_scorer
        scorer.token_veto.grad = torch.tensor([3.0, 4.0])
        scorer.global_absolute.grad = torch.tensor([6.0, 8.0])

        stats = _clip_stage_b_dense_duty_grad_norms(model, 1.0)

        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_token_veto_preclip"], 5.0
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_absolute_preclip"], 10.0
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_token_veto_postclip"],
            5.0 / (125.0**0.5),
            places=5,
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_absolute_postclip"],
            10.0 / (125.0**0.5),
            places=5,
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_active_postclip"], 1.0, places=5
        )
        self.assertEqual(stats["grad_tensor_count_dense_duty_token_veto"], 1.0)
        self.assertEqual(
            stats["grad_tensor_count_dense_duty_global_absolute"], 1.0
        )

    def test_global_trust_veto_subowners_share_the_global_clip(self):
        model = _TinySplitDenseDutyModel(
            head_contract="split_token_veto_global_trust_veto_v4"
        )
        scorer = model.stage_b_fixed_text_scorer
        scorer.token_veto.grad = torch.tensor([3.0, 4.0])
        scorer.global_absolute.grad = torch.tensor([0.0, 6.0])
        scorer.global_veto.grad = torch.tensor([0.0, 8.0])

        stats = _clip_stage_b_dense_duty_grad_norms(model, 1.0)

        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_token_veto_postclip"], 1.0, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_absolute_postclip"], 1.0, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_trust_postclip"], 0.6, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_global_veto_postclip"], 0.8, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_active_postclip"], 2.0**0.5, places=5
        )
        self.assertEqual(stats["grad_tensor_count_dense_duty_global_trust"], 1.0)
        self.assertEqual(stats["grad_tensor_count_dense_duty_global_veto"], 1.0)

    def test_v52_three_owners_are_clipped_independently_with_zero_violations(self):
        model, stats = _v52_clip_stats()
        scorer = model.stage_b_fixed_text_scorer

        expected_preclip = {
            "token_veto": 5.0,
            "candidate_absolute": 10.0,
            "sample_calibrator": 2.0,
        }
        for owner, expected in expected_preclip.items():
            self.assertAlmostEqual(
                stats[f"grad_norm_dense_duty_{owner}_preclip"], expected
            )
            self.assertAlmostEqual(
                stats[f"grad_norm_dense_duty_{owner}_postclip"], 1.0, places=5
            )
            self.assertEqual(
                stats[f"grad_tensor_count_dense_duty_{owner}"], 1.0
            )
            self.assertEqual(
                stats[f"dense_duty_clip_contract_expected_{owner}_tensor_count"],
                1,
            )
            self.assertEqual(
                stats[f"dense_duty_clip_contract_observed_{owner}_tensor_count"],
                1,
            )

        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_active_preclip"], 129.0**0.5, places=5
        )
        self.assertAlmostEqual(
            stats["grad_norm_dense_duty_active_postclip"], 3.0**0.5, places=5
        )
        self.assertAlmostEqual(float(scorer.token_veto.grad.norm()), 1.0, places=5)
        self.assertAlmostEqual(
            float(scorer.candidate_absolute.grad.norm()), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(scorer.sample_calibrator.grad.norm()), 1.0, places=5
        )
        self.assertEqual(stats["dense_duty_clip_contract_checked"], 1)
        for suffix in (
            "live_tensor_count",
            "pre_decomposition",
            "post_decomposition",
            "owner_clip",
            "active_monotonic",
        ):
            self.assertEqual(
                stats[f"dense_duty_clip_contract_{suffix}_violation"], 0
            )

    def test_v52_runtime_recorder_latches_one_bad_step_fail_closed(self):
        _model, good_stats = _v52_clip_stats()
        args = SimpleNamespace(stage_b_dense_duty=True)
        violation_fields = {
            "owner_clip_violation_steps": (
                "dense_duty_clip_contract_owner_clip_violation"
            ),
            "active_pre_decomposition_violation_steps": (
                "dense_duty_clip_contract_pre_decomposition_violation"
            ),
            "active_post_decomposition_violation_steps": (
                "dense_duty_clip_contract_post_decomposition_violation"
            ),
            "live_tensor_count_violation_steps": (
                "dense_duty_clip_contract_live_tensor_count_violation"
            ),
            "active_monotonic_violation_steps": (
                "dense_duty_clip_contract_active_monotonic_violation"
            ),
        }

        def record(stats):
            _record_stage_b_dense_duty_runtime_audit(
                args,
                torch.device("cpu"),
                optimizer_step_boundary=True,
                optimizer_step_succeeded=True,
                branch_grad_norms=stats,
            )

        record(good_stats)
        record(good_stats)
        audit = args.stage_b_dense_duty_runtime_audit
        self.assertEqual(audit["clip_contract_checked_steps"], 2)
        for counter in violation_fields:
            self.assertEqual(audit[counter], 0)

        bad_stats = dict(good_stats)
        for stats_key in violation_fields.values():
            bad_stats[stats_key] = 1
        record(bad_stats)
        record(good_stats)

        audit = args.stage_b_dense_duty_runtime_audit
        self.assertEqual(
            audit["clip_contract_schema"],
            "pivot.stageb.dense_duty_three_owner_clip_contract/v1",
        )
        self.assertEqual(audit["optimizer_step_boundaries"], 4)
        self.assertEqual(audit["successful_optimizer_steps"], 4)
        self.assertEqual(audit["clip_contract_checked_steps"], 4)
        for counter in violation_fields:
            self.assertEqual(audit[counter], 1)
        for owner in (
            "token_veto",
            "candidate_absolute",
            "sample_calibrator",
        ):
            self.assertEqual(audit[f"expected_{owner}_tensor_count"], 1)
            self.assertEqual(audit[f"last_observed_{owner}_tensor_count"], 1)

    def test_v52_checkpoint_runtime_contract_accepts_only_exact_healthy_steps(self):
        runtime = _healthy_v52_runtime_audit()
        validated = _validate_candidate_sample_runtime_audit(
            runtime, expected_steps=400
        )
        self.assertEqual(validated, runtime)

        mutations = {
            "clip_contract_checked_steps": 399,
            "owner_clip_violation_steps": 1,
            "active_pre_decomposition_violation_steps": 1,
            "active_post_decomposition_violation_steps": 1,
            "live_tensor_count_violation_steps": 1,
            "active_monotonic_violation_steps": 1,
            "last_observed_candidate_absolute_tensor_count": 38,
            "expected_sample_calibrator_tensor_count": 8,
            "zero_token_veto_gradient_successful_steps": 1,
            "last_sample_calibrator_grad_norm_preclip": 0.0,
            "max_owner_clip_residual": 1.0e-3,
            "clip_contract_max_norm": 0.2,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                drifted = copy.deepcopy(runtime)
                drifted[field] = value
                with self.assertRaises(RuntimeError):
                    _validate_candidate_sample_runtime_audit(
                        drifted, expected_steps=400
                    )

    def test_global_trust_veto_engine_route_binding_is_tn_isolated(self):
        trust = torch.randn(2, 3, 2, requires_grad=True)
        veto = torch.randn(2, 3, 2, requires_grad=True)
        positive_route = trust - veto
        negative_route = trust.detach() - veto
        candidate_logits = torch.zeros_like(positive_route)
        positive, negative = _select_dense_duty_confidence_loss_logits(
            outputs={
                "stage_b_dense_duty_positive_confidence_logits": positive_route,
                "stage_b_dense_duty_negative_confidence_logits": negative_route,
            },
            candidate_logits=candidate_logits,
            confidence_logits=positive_route,
            head_gradient_contract="split_token_veto_global_trust_veto_v4",
        )
        self.assertTrue(torch.equal(positive, positive_route[..., 0]))
        self.assertTrue(torch.equal(negative, negative_route[..., 1]))

        torch.nn.functional.softplus(negative).mean().backward()
        self.assertIsNone(trust.grad)
        self.assertIsNotNone(veto.grad)
        self.assertTrue(bool(veto.grad.ne(0).any().item()))

        trust = torch.randn(2, 3, 2, requires_grad=True)
        veto = torch.randn(2, 3, 2, requires_grad=True)
        positive_route = trust - veto
        negative_route = trust.detach() - veto
        positive, _negative = _select_dense_duty_confidence_loss_logits(
            outputs={
                "stage_b_dense_duty_positive_confidence_logits": positive_route,
                "stage_b_dense_duty_negative_confidence_logits": negative_route,
            },
            candidate_logits=torch.zeros_like(positive_route),
            confidence_logits=positive_route,
            head_gradient_contract="split_token_veto_global_trust_veto_v4",
        )
        torch.nn.functional.softplus(-positive).mean().backward()
        self.assertIsNotNone(trust.grad)
        self.assertTrue(bool(trust.grad.ne(0).any().item()))

    def test_split_confidence_head_clipping_rejects_ownership_drift(self):
        for ownership_mode in ("overlap", "incomplete"):
            with self.subTest(ownership_mode=ownership_mode):
                model = _TinySplitDenseDutyModel(ownership_mode)
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                with self.assertRaisesRegex(
                    RuntimeError, "empty, overlapping, or incomplete"
                ):
                    _clip_stage_b_dense_duty_grad_norms(model, 1.0)

    def test_runtime_audit_records_each_split_confidence_head(self):
        args = SimpleNamespace(stage_b_dense_duty=True)
        _record_stage_b_dense_duty_runtime_audit(
            args,
            torch.device("cpu"),
            optimizer_step_boundary=True,
            optimizer_step_succeeded=True,
            branch_grad_norms={
                "grad_norm_dense_duty_active_preclip": 5.0,
                "grad_norm_dense_duty_token_veto_preclip": 3.0,
                "grad_norm_dense_duty_global_absolute_preclip": 4.0,
            },
        )
        _record_stage_b_dense_duty_runtime_audit(
            args,
            torch.device("cpu"),
            optimizer_step_boundary=True,
            optimizer_step_succeeded=True,
            branch_grad_norms={
                "grad_norm_dense_duty_active_preclip": 2.0,
                "grad_norm_dense_duty_token_veto_preclip": 0.0,
                "grad_norm_dense_duty_global_absolute_preclip": 2.0,
            },
        )

        audit = args.stage_b_dense_duty_runtime_audit
        self.assertEqual(audit["last_token_veto_grad_norm_preclip"], 0.0)
        self.assertEqual(audit["max_token_veto_grad_norm_preclip"], 3.0)
        self.assertEqual(audit["last_global_absolute_grad_norm_preclip"], 2.0)
        self.assertEqual(audit["max_global_absolute_grad_norm_preclip"], 4.0)
        self.assertEqual(audit["zero_token_veto_gradient_successful_steps"], 1)
        self.assertEqual(
            audit.get("zero_global_absolute_gradient_successful_steps", 0), 0
        )

    def test_probe_evaluation_requires_confidence_and_audited_rank_handoff(self):
        model = _TinyDenseDutyModel()
        model.set_phase("confidence")
        initial = fingerprint_model(model, phase="confidence")
        with torch.no_grad():
            model.stage_b_fixed_text_scorer.confidence_pool.bias.add_(0.5)
        payload = _payload(model, initial, phase="confidence")
        payload["checkpoint_reason"] = "max_train_iters"
        payload["args"].update(
            {
                "stage_b_dense_duty_execution_scope": "probe",
                "stage_b_dense_duty_no_stageb_teacher": True,
                "stage_b_v22_score_ownership": "independent_decoders_two_phase",
                "stage_b_dense_duty_base_checkpoint_sha256": "a" * 64,
                "stage_b_dense_duty_text_checkpoint_sha256": "b" * 64,
                "stage_b_dense_duty_tn_manifest_sha256": "c" * 64,
                "stage_b_dense_duty_dataset_config_sha256": "d" * 64,
                "stage_b_v11_candidate_topk": 50,
                "stage_b_v11_num_layers": 6,
                "stage_b_v15_patch_rank_fusion": False,
                "stage_b_v15_patch_rank_weight": 0.0,
                "stage_b_dense_duty_rank_source_checkpoint_audit": {
                    "schema": CHECKPOINT_AUDIT_SCHEMA,
                    "status": "passed",
                    "phase": "rank",
                    "optimizer_updates": 4,
                    "ownership": {
                        "active_changed": True,
                        "frozen_unchanged": True,
                    },
                    "lineage": {
                        "execution_scope": "probe",
                        "no_stage_b_teacher": True,
                        "dataset_config": {"sha256": "e" * 64},
                    },
                },
            }
        )
        cfg = SimpleNamespace(
            stage_b_dense_duty_evaluation_scope="probe",
            stage_b_dense_duty_rank_expected_optimizer_updates=10,
            stage_b_dense_duty_confidence_expected_optimizer_updates=5,
            stage_b_dense_duty_no_stageb_teacher=True,
            stage_b_v22_score_ownership="independent_decoders_two_phase",
            stage_b_dense_duty_base_checkpoint_sha256="a" * 64,
            stage_b_dense_duty_text_checkpoint_sha256="b" * 64,
            stage_b_dense_duty_tn_manifest_sha256="c" * 64,
            stage_b_dense_duty_dataset_config_sha256="d" * 64,
            stage_b_dense_duty_rank_dataset_config_sha256="e" * 64,
            stage_b_v11_candidate_topk=50,
            stage_b_v11_num_layers=6,
            stage_b_v15_patch_rank_fusion=False,
            stage_b_v15_patch_rank_weight=0.0,
        )
        audit = validate_evaluation_checkpoint_payload(payload, cfg)
        self.assertEqual(audit["evaluation_scope"], "probe")
        self.assertEqual(audit["rank_handoff"]["optimizer_updates"], 4)

        rank_model = _TinyDenseDutyModel()
        rank_model.set_phase("rank")
        rank_initial = fingerprint_model(rank_model, phase="rank")
        with torch.no_grad():
            rank_model.stage_b_fixed_text_scorer.rank_tower.bias.add_(0.5)
        rank_payload = _payload(rank_model, rank_initial, phase="rank")
        with self.assertRaisesRegex(RuntimeError, "confidence checkpoint"):
            validate_evaluation_checkpoint_payload(rank_payload, cfg)

    def test_formal_evaluation_enforces_measured_runtime_contract(self):
        model = _TinyDenseDutyModel()
        model.set_phase("confidence")
        initial = fingerprint_model(model, phase="confidence")
        with torch.no_grad():
            model.stage_b_fixed_text_scorer.confidence_pool.bias.add_(0.5)
        payload = _payload(model, initial, phase="confidence")
        payload.update(
            {
                "optimizer_updates": 5,
                "checkpoint_reason": "max_train_iters",
            }
        )
        payload["args"].update(
            {
                "stage_b_dense_duty_execution_scope": "formal",
                "stage_b_dense_duty_no_stageb_teacher": True,
                "stage_b_v22_score_ownership": "independent_decoders_two_phase",
                "stage_b_dense_duty_base_checkpoint_sha256": "a" * 64,
                "stage_b_dense_duty_text_checkpoint_sha256": "b" * 64,
                "stage_b_dense_duty_tn_manifest_sha256": "c" * 64,
                "stage_b_dense_duty_dataset_config_sha256": "d" * 64,
                "stage_b_v11_candidate_topk": 50,
                "stage_b_v11_num_layers": 6,
                "stage_b_v15_patch_rank_fusion": False,
                "stage_b_v15_patch_rank_weight": 0.0,
                SOURCE_CLOSURE_ARG: _source_closure("confidence"),
                "max_train_iters": 5,
                "batch_size": 16,
                "gradient_accumulation_steps": 4,
                "stage_b_v11_expression_microbatch": 16,
                "stage_b_dense_duty_runtime_audit": {
                    "schema": "pivot.stageb.dense_duty_runtime_audit/v1",
                    "optimizer_step_boundaries": 5,
                    "successful_optimizer_steps": 5,
                    "zero_gradient_successful_steps": 0,
                    "max_active_grad_norm_preclip": 1.0,
                    "peak_reserved_bytes": 1024,
                },
                "stage_b_dense_duty_rank_source_checkpoint_audit": {
                    "schema": CHECKPOINT_AUDIT_SCHEMA,
                    "status": "passed",
                    "phase": "rank",
                    "optimizer_updates": 10,
                    "ownership": {
                        "active_changed": True,
                        "frozen_unchanged": True,
                    },
                    "lineage": {
                        "execution_scope": "formal",
                        "no_stage_b_teacher": True,
                        "dataset_config": {"sha256": "e" * 64},
                    },
                    "source_closure": _source_closure("rank"),
                },
            }
        )
        cfg = SimpleNamespace(
            stage_b_dense_duty_evaluation_scope="formal",
            stage_b_dense_duty_rank_expected_optimizer_updates=10,
            stage_b_dense_duty_confidence_expected_optimizer_updates=5,
            stage_b_dense_duty_expected_physical_batch_size=16,
            stage_b_dense_duty_expected_gradient_accumulation_steps=4,
            stage_b_dense_duty_expected_expression_microbatch=16,
            stage_b_dense_duty_no_stageb_teacher=True,
            stage_b_v22_score_ownership="independent_decoders_two_phase",
            stage_b_dense_duty_base_checkpoint_sha256="a" * 64,
            stage_b_dense_duty_text_checkpoint_sha256="b" * 64,
            stage_b_dense_duty_tn_manifest_sha256="c" * 64,
            stage_b_dense_duty_dataset_config_sha256="d" * 64,
            stage_b_dense_duty_rank_dataset_config_sha256="e" * 64,
            stage_b_v11_candidate_topk=50,
            stage_b_v11_num_layers=6,
            stage_b_v15_patch_rank_fusion=False,
            stage_b_v15_patch_rank_weight=0.0,
        )
        audit = validate_evaluation_checkpoint_payload(payload, cfg)
        self.assertEqual(audit["evaluation_scope"], "formal")

        payload["args"].update(
            {
                "stage_b_dense_duty_confidence_revision": (
                    "word_veto_candidate_split_global_trust_veto_v49"
                ),
                "stage_b_dense_duty_confidence_head_gradient_contract": (
                    "split_token_veto_global_trust_veto_v4"
                ),
            }
        )
        payload["args"]["stage_b_dense_duty_runtime_audit"].update(
            {
                "last_global_trust_grad_norm_preclip": 1.0,
                "max_global_trust_grad_norm_preclip": 2.0,
                "last_global_veto_grad_norm_preclip": 1.5,
                "max_global_veto_grad_norm_preclip": 2.5,
                "nonfinite_global_trust_gradient_boundaries": 0,
                "nonfinite_global_veto_gradient_boundaries": 0,
                "zero_global_trust_gradient_successful_steps": 0,
                "zero_global_veto_gradient_successful_steps": 0,
            }
        )
        audit = validate_evaluation_checkpoint_payload(payload, cfg)
        self.assertEqual(audit["evaluation_scope"], "formal")
        for field, value in (
            ("last_global_trust_grad_norm_preclip", 0.0),
            ("max_global_veto_grad_norm_preclip", float("inf")),
            ("nonfinite_global_trust_gradient_boundaries", 1),
            ("zero_global_veto_gradient_successful_steps", 1),
        ):
            with self.subTest(field=field):
                broken = copy.deepcopy(payload)
                broken["args"]["stage_b_dense_duty_runtime_audit"][field] = value
                with self.assertRaisesRegex(RuntimeError, "formal v31 checkpoint"):
                    validate_evaluation_checkpoint_payload(broken, cfg)

        with self.assertRaisesRegex(
            RuntimeError, "formal evaluation code source closure drifted"
        ):
            validate_evaluation_checkpoint_payload(
                payload,
                cfg,
                current_code_source_closure=_drifted_code_closure(),
            )

        payload["args"]["gradient_accumulation_steps"] = 8
        with self.assertRaisesRegex(RuntimeError, "runtime contract"):
            validate_evaluation_checkpoint_payload(payload, cfg)

    def test_rank_handoff_with_malformed_dataset_lineage_is_rejected_cleanly(self):
        model = _TinyDenseDutyModel()
        model.set_phase("confidence")
        initial = fingerprint_model(model, phase="confidence")
        with torch.no_grad():
            model.stage_b_fixed_text_scorer.confidence_pool.bias.add_(0.5)
        payload = _payload(model, initial, phase="confidence")
        payload["checkpoint_reason"] = "max_train_iters"
        payload["args"].update(
            {
                "stage_b_dense_duty_execution_scope": "probe",
                "stage_b_dense_duty_no_stageb_teacher": True,
                "stage_b_v22_score_ownership": "independent_decoders_two_phase",
                "stage_b_dense_duty_base_checkpoint_sha256": "a" * 64,
                "stage_b_dense_duty_text_checkpoint_sha256": "b" * 64,
                "stage_b_dense_duty_tn_manifest_sha256": "c" * 64,
                "stage_b_dense_duty_dataset_config_sha256": "d" * 64,
                "stage_b_v11_candidate_topk": 50,
                "stage_b_v11_num_layers": 6,
                "stage_b_v15_patch_rank_fusion": False,
                "stage_b_v15_patch_rank_weight": 0.0,
                "stage_b_dense_duty_rank_source_checkpoint_audit": {
                    "schema": CHECKPOINT_AUDIT_SCHEMA,
                    "status": "passed",
                    "phase": "rank",
                    "optimizer_updates": 4,
                    "ownership": {
                        "active_changed": True,
                        "frozen_unchanged": True,
                    },
                    "lineage": {
                        "execution_scope": "probe",
                        "no_stage_b_teacher": True,
                        "dataset_config": "invalid",
                    },
                },
            }
        )
        cfg = SimpleNamespace(
            stage_b_dense_duty_evaluation_scope="probe",
            stage_b_dense_duty_rank_expected_optimizer_updates=10,
            stage_b_dense_duty_confidence_expected_optimizer_updates=5,
            stage_b_dense_duty_no_stageb_teacher=True,
            stage_b_v22_score_ownership="independent_decoders_two_phase",
            stage_b_dense_duty_base_checkpoint_sha256="a" * 64,
            stage_b_dense_duty_text_checkpoint_sha256="b" * 64,
            stage_b_dense_duty_tn_manifest_sha256="c" * 64,
            stage_b_dense_duty_dataset_config_sha256="d" * 64,
            stage_b_dense_duty_rank_dataset_config_sha256="e" * 64,
            stage_b_v11_candidate_topk=50,
            stage_b_v11_num_layers=6,
            stage_b_v15_patch_rank_fusion=False,
            stage_b_v15_patch_rank_weight=0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "invalid rank handoff audit"):
            validate_evaluation_checkpoint_payload(payload, cfg)

    def test_resume_contract_rejects_dataset_or_loss_drift(self):
        values = {key: 1 for key in _RESUME_CONTRACT_KEYS}
        values[SOURCE_CLOSURE_ARG] = _source_closure("rank")
        first = build_training_contract(values)
        self.assertEqual(first, validate_resume_training_contract(values, values))
        drifted = dict(values)
        drifted["stage_b_v11_listwise_weight"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "listwise_weight"):
            validate_resume_training_contract(drifted, values)

    def test_strict_resume_accepts_only_complete_atomic_update_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            args, payload, checkpoint_path = _strict_resume_fixture(
                Path(directory)
            )
            audit = validate_strict_resume_checkpoint_payload(
                payload,
                args,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(audit["phase"], "rank")
            self.assertEqual(audit["optimizer_updates"], 2)
            self.assertEqual(audit["checkpoint_reason"], "interval")

            missing_state = copy.deepcopy(payload)
            del missing_state["scaler"]
            with self.assertRaisesRegex(RuntimeError, "complete training state"):
                validate_strict_resume_checkpoint_payload(
                    missing_state,
                    args,
                    checkpoint_path=checkpoint_path,
                )

            off_boundary = copy.deepcopy(payload)
            off_boundary["iteration"] = 6
            with self.assertRaisesRegex(RuntimeError, "optimizer-update boundary"):
                validate_strict_resume_checkpoint_payload(
                    off_boundary,
                    args,
                    checkpoint_path=checkpoint_path,
                )

    def test_strict_resume_rejects_packed_cursor_beyond_physical_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            args, payload, checkpoint_path = _strict_resume_fixture(
                Path(directory)
            )
            packed = {
                "stage_b_dense_duty_forward_pack_factor": 2,
                "stage_b_dense_duty_logical_loss_batch_size": 16,
                "stage_b_dense_duty_expected_forward_batch_size": 32,
                "stage_b_dense_duty_expected_logical_batches_per_epoch": 887,
                "stage_b_dense_duty_expected_physical_forwards_per_epoch": 444,
            }
            for key, value in packed.items():
                setattr(args, key, value)
                payload["args"][key] = value
            args.gradient_accumulation_steps = 2
            payload["args"]["gradient_accumulation_steps"] = 2
            payload["args"][TRAINING_CONTRACT_ARG] = build_training_contract(
                payload["args"]
            )
            payload["iteration"] = 600

            with self.assertRaisesRegex(RuntimeError, "physical-forward epoch"):
                validate_strict_resume_checkpoint_payload(
                    payload,
                    args,
                    checkpoint_path=checkpoint_path,
                )

    def test_strict_resume_rejects_nonatomic_or_terminal_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            args, payload, checkpoint_path = _strict_resume_fixture(
                Path(directory)
            )
            non_atomic_path = Path(directory) / "checkpoint.pth"
            non_atomic_path.touch()
            with self.assertRaisesRegex(RuntimeError, "atomic checkpoint_iter"):
                validate_strict_resume_checkpoint_payload(
                    payload,
                    args,
                    checkpoint_path=non_atomic_path,
                )

            other_output = Path(directory) / "other-output"
            other_output.mkdir()
            args.output_dir = str(other_output)
            with self.assertRaisesRegex(RuntimeError, "in its output_dir"):
                validate_strict_resume_checkpoint_payload(
                    payload,
                    args,
                    checkpoint_path=checkpoint_path,
                )
            args.output_dir = str(directory)

            terminal = copy.deepcopy(payload)
            terminal.update(
                {
                    "iteration": 0,
                    "optimizer_updates": 10,
                    "epoch_finished": True,
                    "checkpoint_reason": "max_train_iters",
                }
            )
            with self.assertRaisesRegex(RuntimeError, "unfinished phase"):
                validate_strict_resume_checkpoint_payload(
                    terminal,
                    args,
                    checkpoint_path=checkpoint_path,
                )

    def test_strict_resume_rejects_mutated_saved_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            args, payload, checkpoint_path = _strict_resume_fixture(
                Path(directory)
            )
            payload["args"]["stage_b_v11_listwise_weight"] = 0.5
            with self.assertRaisesRegex(RuntimeError, "saved training contract"):
                validate_strict_resume_checkpoint_payload(
                    payload,
                    args,
                    checkpoint_path=checkpoint_path,
                )

    def test_strict_resume_rejects_runtime_update_counter_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            args, payload, checkpoint_path = _strict_resume_fixture(
                Path(directory)
            )
            payload["args"]["stage_b_dense_duty_runtime_audit"][
                "successful_optimizer_steps"
            ] = 1
            with self.assertRaisesRegex(RuntimeError, "optimizer progress"):
                validate_strict_resume_checkpoint_payload(
                    payload,
                    args,
                    checkpoint_path=checkpoint_path,
                )

    def test_strict_confidence_resume_returns_validated_rank_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            args, payload, checkpoint_path = _strict_resume_fixture(
                Path(directory), phase="confidence"
            )
            audit = validate_strict_resume_checkpoint_payload(
                payload,
                args,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(audit["rank_handoff"]["phase"], "rank")
            self.assertEqual(audit["rank_handoff"]["optimizer_updates"], 4)

    def test_confidence_resume_restores_rank_handoff_audit(self):
        model = _TinyDenseDutyModel()
        model.set_phase("confidence")
        initial = fingerprint_model(model, phase="confidence")
        with torch.no_grad():
            model.stage_b_fixed_text_scorer.confidence_pool.bias.add_(0.5)

        values = {key: 1 for key in _RESUME_CONTRACT_KEYS}
        values.update(
            {
                "stage_b_dense_duty": True,
                "stage_b_dense_duty_phase": "confidence",
                "stage_b_dense_duty_execution_scope": "probe",
                "stage_b_dense_duty_rank_dataset_config_sha256": "e" * 64,
                "stage_b_dense_duty_rank_expected_optimizer_updates": 10,
                SOURCE_CLOSURE_ARG: _source_closure("confidence"),
            }
        )
        rank_audit = {
            "schema": CHECKPOINT_AUDIT_SCHEMA,
            "status": "passed",
            "phase": "rank",
            "optimizer_updates": 4,
            "ownership": {
                "active_changed": True,
                "frozen_unchanged": True,
            },
            "lineage": {
                "execution_scope": "probe",
                "no_stage_b_teacher": True,
                "dataset_config": {"sha256": "e" * 64},
            },
            "source_closure": _source_closure("rank"),
        }
        saved_args = dict(values)
        saved_args.update(
            {
                FINGERPRINT_ARG: initial,
                "stage_b_dense_duty_rank_source_checkpoint_audit": rank_audit,
            }
        )
        args = SimpleNamespace(
            **values,
            eval=False,
            output_dir="",
            rank=0,
        )
        _prepare_stage_b_dense_duty_state_fingerprint(
            model,
            args,
            logger=None,
            resume_checkpoint={
                "args": saved_args,
                "optimizer_updates": 1,
            },
        )
        self.assertEqual(
            args.stage_b_dense_duty_rank_source_checkpoint_audit,
            rank_audit,
        )
        with self.assertRaisesRegex(RuntimeError, "rank-to-confidence code"):
            validate_rank_handoff_audit(
                rank_audit,
                execution_scope="probe",
                rank_dataset_sha256="e" * 64,
                code_source_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
