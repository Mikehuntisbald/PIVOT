import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from models.GroundingDINO.stage_b_gdino_score_adapter import StageBGDINOScoreAdapter
from tools.make_stageb_gdino_adapter_p0 import (
    _functional_identity_check,
    build_p0_model_state,
    create_p0,
    verify_p0,
    verify_p0_sidecar,
)
from tools.stageb_fixed_protocol_audit import (
    BASELINE_CONFIG,
    BASELINE_DATASETS,
    FINAL_STAGEA_CHECKPOINT,
)
from tools.stageb_gdino_adapter_probe_audit import (
    ADAPTER_PREFIX,
    CONFIDENCE_MILESTONES,
    RANK_MILESTONES,
    ProbeAuditError,
    SCHEMA,
    _cmd_inspect,
    _cmd_segment_lineage,
    _cmd_verify_evaluation,
    _expected_previous_iteration,
    _resolve_adapter_find_unused_params,
    _milestone_payload,
    _preflight_equivalent,
    _replay_milestone_audit,
    _validate_branch_isolation,
    _validate_rank_initial,
    _validate_previous_audit,
    _validate_segment_lineage_chain,
    _validate_training_checkpoint_common,
    file_record,
    model_hash_record,
    validate_phase_static,
)
from tools.stageb_dependency_audit import config_import_chain
from tools.verify_stageb_p0_record_parity import ParityError, compare_record_groups


def _adapter_state(adapter):
    return {
        ADAPTER_PREFIX + key: value.detach().clone()
        for key, value in adapter.state_dict().items()
    }


def _preflight(phase, base_hash, rank_hash=None, confidence_hash=None):
    initial = {"base_model_sha256": base_hash}
    if rank_hash is not None:
        initial["rank_sha256"] = rank_hash
    if confidence_hash is not None:
        initial["confidence_sha256"] = confidence_hash
    if phase == "rank":
        config = "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py"
        datasets = "config/datasets_stageb_gdino_adapter_rank_three_ref.json"
    else:
        config = "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py"
        datasets = "config/datasets_stageb_gdino_adapter_dataft_pairs.json"
    root = Path(__file__).resolve().parents[1]
    return {
        "phase": phase,
        "initial_checkpoint": initial,
        "static": {
            "config": {"path": str((root / config).resolve())},
            "datasets": {"path": str((root / datasets).resolve())},
        },
        "launch": {"global_batch": 8},
    }


def _training_checkpoint(
    path,
    *,
    phase,
    model,
    source,
    target=50,
    initialization_mode="pretrain",
):
    root = Path(__file__).resolve().parents[1]
    if phase == "rank":
        mode = "rank_only"
        scope = ""
        mode_code = 1
        scope_code = 0
        branch = "rank"
        lr = 3.0e-5
        objective_code = 0
        criterion_trust_margin = 0.0
        criterion_trust_weight = 0.0
        queue_size = 0
        queue_min_count = 0
        config = root / "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py"
        datasets = root / "config/datasets_stageb_gdino_adapter_rank_three_ref.json"
    else:
        mode = "confidence_only"
        scope = "benchmark_dataft_alltn"
        mode_code = 2
        scope_code = 2
        branch = "confidence"
        lr = 3.0e-4
        objective_code = 2
        criterion_trust_margin = 0.02
        criterion_trust_weight = 1.0
        queue_size = 512
        queue_min_count = 256
        config = root / "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py"
        datasets = root / "config/datasets_stageb_gdino_adapter_dataft_pairs.json"
    queue_count = 0 if queue_size == 0 else min(queue_size, target * 8)
    queue_ptr = 0 if queue_size == 0 else (target * 8) % queue_size
    payload = {
        "model": model,
        "criterion": {
            "criterion_train_mode_code": torch.tensor(mode_code),
            "criterion_scope_code": torch.tensor(scope_code),
            "criterion_confidence_objective_code": torch.tensor(objective_code),
            "criterion_positive_trust_margin": torch.tensor(
                criterion_trust_margin
            ),
            "criterion_positive_trust_weight": torch.tensor(
                criterion_trust_weight
            ),
            "criterion_queue_size": torch.tensor(queue_size),
            "criterion_queue_min_count": torch.tensor(queue_min_count),
            "fpr_positive_queue": torch.zeros(queue_size),
            "fpr_negative_queue": torch.zeros(queue_size),
            "fpr_queue_count": torch.tensor(queue_count),
            "fpr_queue_ptr": torch.tensor(queue_ptr),
        },
        "optimizer": {
            "state": {},
            "param_groups": [
                {"params": [0], "lr": lr, "stage_b_gdino_branch": branch}
            ],
        },
        "lr_scheduler": {},
        "scaler": {},
        "rng_state": {"torch": torch.tensor([1], dtype=torch.uint8)},
        "epoch_rng_state": {"torch": torch.tensor([2], dtype=torch.uint8)},
        "epoch": 0,
        "iteration": target,
        "epoch_finished": False,
        "checkpoint_reason": "max_train_iters",
        "args": {
            "config_file": str(config),
            "datasets": str(datasets),
            "world_size": 2,
            "batch_size": 4,
            "distributed": True,
            "amp": True,
            "max_train_iters": target,
            "data_aug_hflip_prob": 0.0,
            "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
            "stage_b_gdino_gate_pool_temperature": 0.01,
            "stage_b_gdino_gate_topk": 3,
            "stage_b_gdino_fpr_temperature": 0.1,
            "stage_b_gdino_fpr_margin": 0.0,
            "stage_b_gdino_paired_margin": 0.05,
            "stage_b_gdino_paired_margin_weight": (
                0.0 if phase == "rank" else 0.25
            ),
            "stage_b_gdino_positive_trust_margin": 0.02,
            "stage_b_gdino_positive_trust_weight": 1.0,
            "stage_b_gdino_queue_size": queue_size,
            "stage_b_gdino_queue_min_count": queue_min_count,
            "pretrain_model_path": (
                str(source) if initialization_mode == "pretrain" else ""
            ),
            "resume": str(source) if initialization_mode == "resume" else "",
            "stage_b_gdino_adapter_train_mode": mode,
            "stage_b_gdino_tn_scope": scope,
        },
    }
    torch.save(payload, path)


def _record(task, index, n, *, value=0.5):
    split = "global" if task == "tn" else "refcoco_val"
    row = {
        "task": task,
        "manifest_key": "tn_global" if task == "tn" else "ref:refcoco_val",
        "manifest_sha256": "a" * 64,
        "manifest_n": n,
        "manifest_index": index,
        "sample_id": f"sample:{index}",
        "split": split,
        "image_id": index,
        "ann_id": 100 + index,
        "ref_id": 200 + index,
        "sent_id": 300 + index,
        "valid": True,
    }
    if task == "tn":
        row.update(
            {
                "pos_score": value,
                "neg_score": value - 0.1,
                "pos_iou": 0.7,
                "neg_iou": 0.2,
            }
        )
    else:
        row.update(
            {
                "top1_iou": value,
                "all_query_best_iou": min(1.0, value + 0.1),
                "correct50": value >= 0.5,
            }
        )
    return row


def _build_rank_milestone_chain(root, targets=(50, 100, 250, 500)):
    initial = root / "initial.pth"
    initial.write_bytes(b"rank initial")
    adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
    model = {"base.weight": torch.randn(2, 3), **_adapter_state(adapter)}
    model[ADAPTER_PREFIX + "rank_output.bias"].add_(0.25)
    hashes = model_hash_record(model)
    preflight = _preflight("rank", hashes["base_model_sha256"])
    preflight.update({"schema": SCHEMA, "kind": "phase_preflight"})
    preflight["initial_checkpoint"].update(file_record(initial))
    preflight_path = root / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    chain = {"initial": initial, "preflight": preflight_path}
    previous_checkpoint = None
    previous_audit = None
    for target in targets:
        checkpoint = root / f"checkpoint_{target}.pth"
        lineage = root / f"lineage_{target}.json"
        audit = root / f"audit_{target}.json"
        source = initial if previous_checkpoint is None else previous_checkpoint
        mode = "pretrain" if previous_checkpoint is None else "resume"
        _training_checkpoint(
            checkpoint,
            phase="rank",
            model=model,
            source=source,
            target=target,
            initialization_mode=mode,
        )
        lineage.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "kind": "segment_lineage",
                    "phase": "rank",
                    "expected_target": target,
                    "initialization_mode": mode,
                    "ancestry": (
                        "phase_initial"
                        if previous_checkpoint is None
                        else "previous_milestone"
                    ),
                    "source_checkpoint": file_record(source),
                    "preflight": file_record(preflight_path),
                    "previous_audit": (
                        file_record(previous_audit) if previous_audit else None
                    ),
                    "recovery_inspection": None,
                }
            ),
            encoding="utf-8",
        )
        payload = _milestone_payload(
            phase="rank",
            checkpoint=checkpoint,
            preflight_path=preflight_path,
            iteration=target,
            source=source,
            previous_path=previous_audit,
            segment_lineage_path=lineage,
        )
        audit.write_text(json.dumps(payload), encoding="utf-8")
        chain[target] = {
            "checkpoint": checkpoint,
            "lineage": lineage,
            "audit": audit,
        }
        previous_checkpoint = checkpoint
        previous_audit = audit
    return chain


class AdapterProbeOrchestrationTest(unittest.TestCase):
    def test_find_unused_params_uses_cli_default_and_rejects_true(self):
        missing = type("Cfg", (), {})()
        explicit_false = type("Cfg", (), {"find_unused_params": False})()
        explicit_true = type("Cfg", (), {"find_unused_params": True})()
        self.assertIs(
            _resolve_adapter_find_unused_params(missing, phase="rank"), False
        )
        self.assertIs(
            _resolve_adapter_find_unused_params(
                explicit_false, phase="confidence"
            ),
            False,
        )
        with self.assertRaisesRegex(ProbeAuditError, "find_unused_params"):
            _resolve_adapter_find_unused_params(explicit_true, phase="rank")

    def test_phase_specific_milestones_keep_confidence_bounded(self):
        self.assertEqual(
            RANK_MILESTONES,
            (50, 100, 250, 500, 1000, 2000, 5000),
        )
        self.assertEqual(CONFIDENCE_MILESTONES, (50, 100, 250, 500))
        self.assertEqual(_expected_previous_iteration("rank", 5000), 2000)
        self.assertEqual(_expected_previous_iteration("rank", 1000), 500)
        self.assertEqual(_expected_previous_iteration("confidence", 500), 250)
        with self.assertRaisesRegex(ProbeAuditError, "confidence milestone iteration"):
            _expected_previous_iteration("confidence", 1000)

    def test_rank_and_confidence_static_contract_disable_horizontal_flip(self):
        for phase in ("rank", "confidence"):
            with self.subTest(phase=phase):
                static = validate_phase_static(phase)
                self.assertEqual(
                    static["resolved_config"]["data_aug_hflip_prob"], 0.0
                )
                self.assertEqual(
                    static["resolved_config"][
                        "stage_b_gdino_confidence_objective"
                    ],
                    "detached_recent_q05_trust",
                )
                expected_queue = 0 if phase == "rank" else 512
                self.assertEqual(
                    static["resolved_config"]["stage_b_gdino_queue_size"],
                    expected_queue,
                )

    def test_confidence_pair_dataset_disables_internal_negative_resampling(self):
        root = Path(__file__).resolve().parents[1]
        dataset = json.loads(
            (root / "config/datasets_stageb_gdino_adapter_dataft_pairs.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(dataset["train"][0]["neg_episode_prob"], 0.0)
        validate_phase_static("confidence")

    def test_p0_is_functionally_identity_and_preserves_baseline_state(self):
        torch.manual_seed(7)
        adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
        _functional_identity_check(adapter)
        baseline = {"base.weight": torch.randn(2, 3)}
        merged = build_p0_model_state(baseline, adapter)
        self.assertTrue(torch.equal(merged["base.weight"], baseline["base.weight"]))
        record = model_hash_record(merged)
        self.assertTrue(record["rank_final_zero"])
        self.assertTrue(record["confidence_final_zero"])

    def test_p0_create_verify_and_sidecar_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            baseline_dir.mkdir()
            baseline = baseline_dir / "checkpoint0000.pth"
            repo = Path(__file__).resolve().parents[1]
            torch.save(
                {
                    "model": {"base.weight": torch.randn(2, 3)},
                    "epoch": 0,
                    "epoch_finished": True,
                    "iteration": 0,
                    "checkpoint_reason": "epoch_end",
                    "args": {
                        "batch_size": 9,
                        "world_size": 2,
                        "distributed": True,
                        "amp": True,
                        "seed": 42,
                        "epochs": 1,
                        "stage_b": False,
                        "patch_only": False,
                        "resume": "",
                        "pretrain_model_path": str(repo / FINAL_STAGEA_CHECKPOINT),
                        "config_file": str(repo / BASELINE_CONFIG),
                        "datasets": str(repo / BASELINE_DATASETS),
                        "output_dir": str(baseline_dir),
                    },
                },
                baseline,
            )
            (baseline_dir / "protocol_train_complete.json").write_text(
                json.dumps(
                    {
                        "authoritative_checkpoint": {
                            "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = (repo / "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py").resolve()
            p0 = root / "p0" / "checkpoint_p0.pth"
            created = create_p0(
                baseline_checkpoint=baseline,
                output=p0,
                config=config,
                seed=42,
            )
            verified = verify_p0(
                baseline_checkpoint=baseline,
                p0_checkpoint=p0,
                config=config,
            )
            self.assertEqual(created, verified)
            sidecar_verification = verify_p0_sidecar(
                p0_checkpoint=p0,
                audit=verified,
                sidecar=Path(str(p0) + ".audit.json"),
            )
            self.assertEqual(
                sidecar_verification["kind"],
                "p0_checkpoint_and_sidecar_verified",
            )
            self.assertEqual(sidecar_verification["audit"], verified)

    def test_hashes_separate_base_rank_and_confidence_branches(self):
        adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
        state = {"base.weight": torch.randn(2, 3), **_adapter_state(adapter)}
        before = model_hash_record(state)
        state[ADAPTER_PREFIX + "rank_output.bias"].add_(1.0)
        after_rank = model_hash_record(state)
        self.assertEqual(before["base_model_sha256"], after_rank["base_model_sha256"])
        self.assertNotEqual(before["rank_sha256"], after_rank["rank_sha256"])
        self.assertEqual(before["confidence_sha256"], after_rank["confidence_sha256"])
        state[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(1.0)
        after_gate = model_hash_record(state)
        self.assertEqual(after_rank["rank_sha256"], after_gate["rank_sha256"])
        self.assertNotEqual(
            after_rank["confidence_sha256"], after_gate["confidence_sha256"]
        )

    def test_rank_and_confidence_checkpoint_phase_audits_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pth"
            source.write_bytes(b"source")
            adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
            rank_model = {"base.weight": torch.randn(2, 3), **_adapter_state(adapter)}
            rank_model[ADAPTER_PREFIX + "rank_output.bias"].add_(0.25)
            rank_hashes = model_hash_record(rank_model)
            rank_path = root / "rank.pth"
            _training_checkpoint(
                rank_path, phase="rank", model=rank_model, source=source
            )
            rank_preflight = _preflight("rank", rank_hashes["base_model_sha256"])
            rank_result = _validate_training_checkpoint_common(
                phase="rank",
                checkpoint=rank_path,
                preflight=rank_preflight,
                expected_target=50,
                source_checkpoint=source,
                require_exact_iteration=True,
            )
            _validate_branch_isolation(
                phase="rank",
                record=rank_result["record"],
                initial=rank_preflight["initial_checkpoint"],
                previous=None,
            )
            nonzero_gate = dict(rank_result["record"])
            nonzero_gate["confidence_final_zero"] = False
            with self.assertRaisesRegex(ProbeAuditError, "zero-init confidence"):
                _validate_branch_isolation(
                    phase="rank",
                    record=nonzero_gate,
                    initial=rank_preflight["initial_checkpoint"],
                    previous=None,
                )

            confidence_model = copy.deepcopy(rank_model)
            confidence_model[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.5)
            confidence_path = root / "confidence.pth"
            _training_checkpoint(
                confidence_path,
                phase="confidence",
                model=confidence_model,
                source=rank_path,
            )
            confidence_preflight = _preflight(
                "confidence",
                rank_hashes["base_model_sha256"],
                rank_hashes["rank_sha256"],
                rank_hashes["confidence_sha256"],
            )
            confidence_result = _validate_training_checkpoint_common(
                phase="confidence",
                checkpoint=confidence_path,
                preflight=confidence_preflight,
                expected_target=50,
                source_checkpoint=rank_path,
                require_exact_iteration=True,
            )
            _validate_branch_isolation(
                phase="confidence",
                record=confidence_result["record"],
                initial=confidence_preflight["initial_checkpoint"],
                previous=None,
            )
            bad_objective_path = root / "confidence_bad_objective.pth"
            bad_objective = torch.load(confidence_path, weights_only=False)
            bad_objective["criterion"][
                "criterion_confidence_objective_code"
            ] = torch.tensor(1)
            torch.save(bad_objective, bad_objective_path)
            with self.assertRaisesRegex(
                ProbeAuditError, "criterion_confidence_objective_code"
            ):
                _validate_training_checkpoint_common(
                    phase="confidence",
                    checkpoint=bad_objective_path,
                    preflight=confidence_preflight,
                    expected_target=50,
                    source_checkpoint=rank_path,
                    require_exact_iteration=True,
                )

            cold_queue_path = root / "confidence_cold_queue.pth"
            cold_queue = torch.load(confidence_path, weights_only=False)
            cold_queue["criterion"]["fpr_queue_count"] = torch.tensor(100)
            cold_queue["criterion"]["fpr_queue_ptr"] = torch.tensor(100)
            torch.save(cold_queue, cold_queue_path)
            with self.assertRaisesRegex(ProbeAuditError, "queue is not warm"):
                _validate_training_checkpoint_common(
                    phase="confidence",
                    checkpoint=cold_queue_path,
                    preflight=confidence_preflight,
                    expected_target=50,
                    source_checkpoint=rank_path,
                    require_exact_iteration=True,
                )

            bad = torch.load(confidence_path, weights_only=False)
            bad["criterion"]["criterion_train_mode_code"] = torch.tensor(1)
            torch.save(bad, confidence_path)
            with self.assertRaisesRegex(ProbeAuditError, "train-mode code"):
                _validate_training_checkpoint_common(
                    phase="confidence",
                    checkpoint=confidence_path,
                    preflight=confidence_preflight,
                    expected_target=50,
                    source_checkpoint=rank_path,
                    require_exact_iteration=True,
                )

    def test_formal_evaluation_rejects_an_arbitrary_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "expected.pth"
            arbitrary = root / "arbitrary.pth"
            expected.write_bytes(b"expected")
            arbitrary.write_bytes(b"arbitrary")
            audit = root / "milestone.audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "milestone_checkpoint",
                        "phase": "rank",
                        "checkpoint": {"path": str(expected.resolve())},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProbeAuditError, "milestone iteration must be"):
                _cmd_verify_evaluation(
                    Namespace(checkpoint=str(arbitrary), audit=str(audit), output=None)
                )

    def test_confidence_preflight_replays_rank_milestone_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "rank.pth"
            torch.save({"model": {"base.weight": torch.ones(1)}}, checkpoint)
            audit = root / "rank.audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "milestone_checkpoint",
                        "phase": "rank",
                        "iteration": 50,
                        "checkpoint": file_record(checkpoint),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ProbeAuditError, "missing checkpoint/preflight/source"
            ):
                _validate_rank_initial(checkpoint, audit)

    def test_confidence_preflight_accepts_an_audited_r5000_source(self):
        checkpoint_record_value = {"path": "/tmp/r5000.pth", "sha256": "a" * 64}
        with patch(
            "tools.stageb_gdino_adapter_probe_audit._verify_milestone_checkpoint",
            return_value={
                "phase": "rank",
                "iteration": 5000,
                "checkpoint": checkpoint_record_value,
            },
        ):
            result = _validate_rank_initial(
                Path("/tmp/r5000.pth"), Path("/tmp/r5000.audit.json")
            )
        self.assertEqual(result, checkpoint_record_value)

    def test_milestone_replay_accepts_only_the_full_adjacent_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            chain = _build_rank_milestone_chain(Path(temporary))
            replayed = _replay_milestone_audit(chain[500]["audit"])
            self.assertEqual(replayed["iteration"], 500)
            self.assertEqual(
                replayed["previous_audit"], file_record(chain[250]["audit"])
            )

    def test_rank_milestone_replay_extends_adjacent_chain_to_r5000(self):
        with tempfile.TemporaryDirectory() as temporary:
            chain = _build_rank_milestone_chain(
                Path(temporary), targets=RANK_MILESTONES
            )
            replayed = _replay_milestone_audit(chain[5000]["audit"])
            self.assertEqual(replayed["iteration"], 5000)
            self.assertEqual(
                replayed["previous_audit"], file_record(chain[2000]["audit"])
            )

    def test_r5000_rejects_a_skipped_r2000_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            chain = _build_rank_milestone_chain(
                Path(temporary), targets=(50, 100, 250, 500, 1000)
            )
            with self.assertRaisesRegex(ProbeAuditError, "expected 2000, got 1000"):
                _validate_previous_audit(
                    "rank",
                    chain[1000]["audit"],
                    5000,
                    preflight_path=chain["preflight"],
                )

    def test_milestone_replay_rejects_a_skipped_previous_milestone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chain = _build_rank_milestone_chain(root, targets=(50, 100))
            checkpoint = root / "checkpoint_500.pth"
            lineage = root / "lineage_500.json"
            model = torch.load(chain[100]["checkpoint"], weights_only=False)["model"]
            _training_checkpoint(
                checkpoint,
                phase="rank",
                model=model,
                source=chain[100]["checkpoint"],
                target=500,
                initialization_mode="resume",
            )
            lineage.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "segment_lineage",
                        "phase": "rank",
                        "expected_target": 500,
                        "initialization_mode": "resume",
                        "ancestry": "previous_milestone",
                        "source_checkpoint": file_record(chain[100]["checkpoint"]),
                        "preflight": file_record(chain["preflight"]),
                        "previous_audit": file_record(chain[100]["audit"]),
                        "recovery_inspection": None,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProbeAuditError, "expected 250, got 100"):
                _milestone_payload(
                    phase="rank",
                    checkpoint=checkpoint,
                    preflight_path=chain["preflight"],
                    iteration=500,
                    source=chain[100]["checkpoint"],
                    previous_path=chain[100]["audit"],
                    segment_lineage_path=lineage,
                )

    def test_milestone_replay_rejects_forged_previous_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            chain = _build_rank_milestone_chain(
                Path(temporary), targets=(50, 100)
            )
            audit_path = chain[100]["audit"]
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["global_batch"] = 999
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ProbeAuditError, "payload drifted"):
                _replay_milestone_audit(audit_path)

    def test_milestone_replay_rejects_cycle_and_disconnected_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            chain = _build_rank_milestone_chain(
                Path(temporary), targets=(50, 100)
            )
            audit_path = chain[100]["audit"].resolve()
            with self.assertRaisesRegex(ProbeAuditError, "cycle detected"):
                _replay_milestone_audit(audit_path, replay_stack={audit_path})

            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["previous_audit"] = None
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ProbeAuditError, "disconnected"):
                _replay_milestone_audit(audit_path)

    def test_recovery_inspection_rejects_a_forked_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_source = root / "expected_source.pth"
            fork_source = root / "fork_source.pth"
            expected_source.write_bytes(b"expected")
            fork_source.write_bytes(b"fork")
            adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
            model = {"base.weight": torch.randn(2, 3), **_adapter_state(adapter)}
            model[ADAPTER_PREFIX + "rank_output.bias"].add_(0.25)
            hashes = model_hash_record(model)
            checkpoint = root / "live.pth"
            _training_checkpoint(
                checkpoint,
                phase="rank",
                model=model,
                source=fork_source,
                target=50,
            )
            preflight = _preflight("rank", hashes["base_model_sha256"])
            preflight["schema"] = SCHEMA
            preflight["kind"] = "phase_preflight"
            preflight["initial_checkpoint"].update(file_record(expected_source))
            preflight_path = root / "preflight.json"
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            lineage_path = root / "segment.lineage.json"
            _cmd_segment_lineage(
                Namespace(
                    phase="rank",
                    expected_target=50,
                    preflight=str(preflight_path),
                    source_checkpoint=str(expected_source),
                    initialization_mode="pretrain",
                    previous_audit=None,
                    recovery_inspection=None,
                    output=str(lineage_path),
                )
            )
            with self.assertRaisesRegex(ProbeAuditError, "lineage must use exactly"):
                _cmd_inspect(
                    Namespace(
                        phase="rank",
                        checkpoint=str(checkpoint),
                        preflight=str(preflight_path),
                        expected_target=50,
                        segment_lineage=str(lineage_path),
                        previous_audit=None,
                        output=None,
                        print_iteration=False,
                    )
                )

    def test_fake_recovery_chain_cannot_become_milestone_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.pth"
            fork = root / "fork.pth"
            recovery = root / "recovery.pth"
            initial.write_bytes(b"initial")
            fork.write_bytes(b"fork")
            recovery.write_bytes(b"recovery")
            preflight_path = root / "preflight.json"
            preflight_path.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "phase_preflight",
                        "phase": "rank",
                        "initial_checkpoint": file_record(initial),
                    }
                ),
                encoding="utf-8",
            )
            prior_lineage = root / "prior.lineage.json"
            prior_lineage.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "segment_lineage",
                        "phase": "rank",
                        "expected_target": 50,
                        "initialization_mode": "pretrain",
                        "ancestry": "phase_initial",
                        "source_checkpoint": file_record(fork),
                        "preflight": file_record(preflight_path),
                        "previous_audit": None,
                        "recovery_inspection": None,
                    }
                ),
                encoding="utf-8",
            )
            inspection = root / "inspection.json"
            inspection.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "live_checkpoint_inspection",
                        "phase": "rank",
                        "expected_target": 50,
                        "checkpoint": file_record(recovery),
                        "segment_lineage": file_record(prior_lineage),
                    }
                ),
                encoding="utf-8",
            )
            current_lineage = root / "current.lineage.json"
            current_lineage.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "segment_lineage",
                        "phase": "rank",
                        "expected_target": 50,
                        "initialization_mode": "resume",
                        "ancestry": "audited_live_recovery",
                        "source_checkpoint": file_record(recovery),
                        "preflight": file_record(preflight_path),
                        "previous_audit": None,
                        "recovery_inspection": file_record(inspection),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProbeAuditError, "initial segment lineage"):
                _validate_segment_lineage_chain(
                    lineage_path=current_lineage,
                    phase="rank",
                    target=50,
                    preflight_path=preflight_path,
                    previous_path=None,
                    source=recovery,
                )

    def test_recursive_config_chain_detects_parent_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_dir = root / "config"
            config_dir.mkdir()
            parent = config_dir / "parent.py"
            child = config_dir / "child.py"
            parent.write_text("value = 1\n", encoding="utf-8")
            child.write_text("from config.parent import *\n", encoding="utf-8")

            def records():
                return [
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in config_import_chain(child, root=root)
                ]

            before = {"config_import_chain": records()}
            parent.write_text("value = 2\n", encoding="utf-8")
            after = {"config_import_chain": records()}
            self.assertFalse(_preflight_equivalent(before, after))

    def test_rank_wrapper_runs_a_bounded_prefix_and_accepts_extended_selection(self):
        repo = Path(__file__).resolve().parents[1]
        wrapper = repo / "tools/run_stageb_gdino_adapter_two_phase_probe.sh"

        def dry_run(*arguments):
            completed = subprocess.run(
                [str(wrapper), *arguments, "--dry-run"],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return completed.stdout

        default_rank = dry_run("--phase", "rank")
        self.assertIn("--max_train_iters 500 ", default_rank)
        self.assertNotIn("--max_train_iters 1000 ", default_rank)

        extended_rank = dry_run(
            "--phase", "rank", "--rank-max-target", "5000"
        )
        for target in RANK_MILESTONES:
            self.assertIn(f"--max_train_iters {target} ", extended_rank)
        self.assertIn("checkpoint_iter_002000.pth", extended_rank)

        confidence = dry_run(
            "--phase", "confidence", "--rank-selection", "5000"
        )
        self.assertIn("checkpoint_iter_005000.pth", confidence)
        self.assertIn("--max_train_iters 500 ", confidence)
        self.assertNotIn("--max_train_iters 1000 ", confidence)

    def test_p0_sidecar_must_match_recomputed_checkpoint_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "p0.pth"
            sidecar = root / "p0.pth.audit.json"
            checkpoint.write_bytes(b"p0")
            sidecar.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ProbeAuditError, "does not exactly match"):
                verify_p0_sidecar(
                    p0_checkpoint=checkpoint,
                    audit={"schema": "stageb-gdino-adapter-p0-v1"},
                    sidecar=sidecar,
                )

    def test_eval_dry_run_derives_rank_and_semantic_training_configs(self):
        repo = Path(__file__).resolve().parents[1]
        wrapper = repo / "tools/run_stageb_gdino_adapter_probe_eval.sh"
        rank_config = (
            repo / "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py"
        ).resolve()
        semantic_config = (
            repo
            / "config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py"
        ).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank_preflight = root / "rank.preflight.json"
            rank_preflight.write_text(
                json.dumps({"static": {"config": {"path": str(rank_config)}}}),
                encoding="utf-8",
            )
            audits = (
                (
                    "rank",
                    {
                        "schema": SCHEMA,
                        "preflight": {"path": str(rank_preflight)},
                    },
                    rank_config,
                ),
                (
                    "semantic",
                    {
                        "schema": "stageb-gdino-adapter-semantic-confidence-probe-v1",
                        "config": {"path": str(semantic_config)},
                    },
                    semantic_config,
                ),
            )
            for label, payload, expected in audits:
                audit = root / f"{label}.audit.json"
                audit.write_text(json.dumps(payload), encoding="utf-8")
                completed = subprocess.run(
                    [
                        str(wrapper),
                        "--checkpoint",
                        str(root / "not-built.pth"),
                        "--checkpoint-audit",
                        str(audit),
                        "--label",
                        label,
                        "--dry-run",
                    ],
                    cwd=repo,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(f"--config {expected}", completed.stdout)
                self.assertEqual(
                    completed.stdout.count(
                        "tools/verify_stageb_fixed_eval_summary_binding.py"
                    ),
                    2,
                )
                self.assertIn("--eval-dir outputs/stageb_gdino_adapter_fixed_protocol_eval_20260711/baseline", completed.stdout)

    def test_eval_wrapper_pins_and_runs_summary_binding_before_comparison(self):
        repo = Path(__file__).resolve().parents[1]
        wrapper_path = repo / "tools/run_stageb_gdino_adapter_probe_eval.sh"
        auditor_path = repo / "tools/verify_stageb_fixed_eval_summary_binding.py"
        fixed_top1_path = repo / "tools/stageb_gdino_fixed_top1_probe_audit.py"
        selector_path = repo / "tools/stageb_gdino_fixed_top1_selection.py"
        comparator_path = repo / "tools/compare_stageb_fpr95_records.py"
        calibration_evaluator_path = (
            repo / "tools/eval_stageb_gdino_fixed_top1_calibration.py"
        )
        calibration_launcher_path = (
            repo / "tools/run_stageb_gdino_fixed_top1_calibration.sh"
        )
        semantic_path = repo / "tools/stageb_gdino_semantic_probe_audit.py"
        wrapper = wrapper_path.read_text(encoding="utf-8")
        auditor_sha = hashlib.sha256(auditor_path.read_bytes()).hexdigest()
        fixed_top1_sha = hashlib.sha256(fixed_top1_path.read_bytes()).hexdigest()
        semantic_sha = hashlib.sha256(semantic_path.read_bytes()).hexdigest()
        selector_sha = hashlib.sha256(selector_path.read_bytes()).hexdigest()
        comparator_sha = hashlib.sha256(comparator_path.read_bytes()).hexdigest()
        calibration_evaluator_sha = hashlib.sha256(
            calibration_evaluator_path.read_bytes()
        ).hexdigest()
        calibration_launcher_sha = hashlib.sha256(
            calibration_launcher_path.read_bytes()
        ).hexdigest()
        self.assertIn(
            f'EVAL_SUMMARY_AUDITOR_SHA256="{auditor_sha}"',
            wrapper,
        )
        self.assertIn(
            f'FIXED_TOP1_AUDITOR_SHA256="{fixed_top1_sha}"',
            wrapper,
        )
        self.assertIn(
            f'SEMANTIC_AUDITOR_SHA256="{semantic_sha}"',
            wrapper,
        )
        self.assertIn(f'FIXED_TOP1_SELECTOR_SHA256="{selector_sha}"', wrapper)
        self.assertIn(f'FPR_COMPARATOR_SHA256="{comparator_sha}"', wrapper)
        self.assertIn(
            f'FIXED_TOP1_CALIBRATION_EVALUATOR_SHA256="{calibration_evaluator_sha}"',
            wrapper,
        )
        self.assertIn(
            f'FIXED_TOP1_CALIBRATION_LAUNCHER_SHA256="{calibration_launcher_sha}"',
            wrapper,
        )
        self.assertIn("verify_semantic_auditor\n", wrapper)
        formal = wrapper[wrapper.index('mkdir -p "${COMPARISON_DIR}"') :]
        self.assertEqual(formal.count('"${EVAL_SUMMARY_AUDITOR}"'), 2)
        self.assertLess(
            formal.index('baseline_summary_binding.json'),
            formal.index("tools/stageb_fixed_protocol_audit.py compare-evals"),
        )
        self.assertLess(
            formal.index('candidate_summary_binding.json'),
            formal.index("tools/stageb_fixed_protocol_audit.py compare-evals"),
        )
        candidate_binding = formal[formal.index('candidate_summary_binding.json') - 500 :]
        self.assertIn('--trusted-lineage "${LINEAGE_OUTPUT}"', candidate_binding)
        self.assertIn(
            '--expected-baseline-checkpoint "${BASELINE_CHECKPOINT}"',
            candidate_binding,
        )
        self.assertLess(
            formal.index("validate_candidate_lineage"),
            formal.index('candidate_summary_binding.json'),
        )
        self.assertEqual(
            wrapper.count('validate_candidate_lineage "${LINEAGE_OUTPUT}"'), 1
        )
        self.assertEqual(
            wrapper.count('validate_candidate_lineage "${LINEAGE_POST_OUTPUT}"'), 1
        )
        self.assertIn("lineage_replay_equality.json", formal)
        final_writer = wrapper[wrapper.index("write_final_acceptance_status()") :]
        self.assertIn("validate_final_metric_input_binding", final_writer)
        self.assertIn('"metric_input_binding": metric_input_binding', final_writer)

    def test_eval_wrapper_final_acceptance_status_is_fail_closed(self):
        repo = Path(__file__).resolve().parents[1]
        wrapper = (
            repo / "tools/run_stageb_gdino_adapter_probe_eval.sh"
        ).read_text(encoding="utf-8")
        formal = wrapper[wrapper.index('mkdir -p "${COMPARISON_DIR}"') :]
        self.assertLess(
            formal.index('write_nonacceptance_status "pending"'),
            formal.index('baseline_summary_binding.json'),
        )
        self.assertIn(
            '"P0 parity is a control and can never claim final metric acceptance"',
            formal,
        )
        self.assertIn(
            '"diagnostic mode cannot claim final acceptance even when metrics pass"',
            formal,
        )
        strict_true_call = formal.index("write_final_acceptance_status")
        diagnostic_branch = formal.index('elif [[ "${DIAGNOSTIC}" == "1" ]]')
        self.assertGreater(strict_true_call, diagnostic_branch)
        self.assertIn('"final_acceptance_claimed": True', wrapper)
        self.assertIn('"final_acceptance_claimed": False', wrapper)

    def test_record_parity_requires_full_identity_and_exact_values(self):
        baseline = {"tn_global": [_record("tn", index, 3) for index in range(3)]}
        p0 = copy.deepcopy(baseline)
        report = compare_record_groups(baseline, p0)
        self.assertEqual(report["tn_global"]["valid"], 3)
        p0["tn_global"][1]["ann_id"] = 999
        with self.assertRaisesRegex(ParityError, "identity mismatch"):
            compare_record_groups(baseline, p0)
        p0 = copy.deepcopy(baseline)
        p0["tn_global"][1]["neg_score"] += 1e-6
        with self.assertRaisesRegex(ParityError, "value mismatch"):
            compare_record_groups(baseline, p0, atol=0.0)
        identity_report = compare_record_groups(
            baseline, p0, compare_values=False
        )
        self.assertTrue(identity_report["tn_global"]["identity_aligned"])
        self.assertIsNone(identity_report["tn_global"]["values_equal"])
        compare_record_groups(baseline, p0, atol=1e-5)

        baseline_ref = {"ref:refcoco_val": [_record("ref", 0, 1)]}
        p0_ref = copy.deepcopy(baseline_ref)
        p0_ref["ref:refcoco_val"][0]["all_query_best_iou"] += 1e-6
        with self.assertRaisesRegex(ParityError, "all_query_best_iou"):
            compare_record_groups(baseline_ref, p0_ref, atol=0.0)


if __name__ == "__main__":
    unittest.main()
