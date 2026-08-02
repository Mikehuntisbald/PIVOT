import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.GroundingDINO.stage_b_gdino_score_adapter import StageBGDINOScoreAdapter
from tools.stageb_gdino_adapter_probe_audit import (
    ADAPTER_PREFIX,
    SCHEMA as TWO_PHASE_SCHEMA,
    checkpoint_record,
    file_record,
    model_hash_record,
)
from tools.stageb_gdino_semantic_probe_audit import (
    SCHEMA,
    SemanticProbeError,
    _resolve_semantic_find_unused_params,
    _validate_source,
    make_preflight,
    make_segment_lineage,
    validate_checkpoint,
    validate_static,
    verify_evaluation_checkpoint,
)


def _adapter_state(adapter):
    return {
        ADAPTER_PREFIX + key: value.detach().clone()
        for key, value in adapter.state_dict().items()
    }


class StageBGDINOSemanticProbeOrchestrationTest(unittest.TestCase):
    def test_find_unused_params_uses_cli_default_and_rejects_true(self):
        missing = type("Cfg", (), {})()
        explicit_false = type("Cfg", (), {"find_unused_params": False})()
        explicit_true = type("Cfg", (), {"find_unused_params": True})()
        self.assertIs(_resolve_semantic_find_unused_params(missing), False)
        self.assertIs(
            _resolve_semantic_find_unused_params(explicit_false), False
        )
        with self.assertRaisesRegex(SemanticProbeError, "find_unused_params"):
            _resolve_semantic_find_unused_params(explicit_true)

    def _checkpoint(
        self,
        path,
        *,
        source,
        model,
        use_resume=False,
        iteration=50,
        max_target=50,
        reason="max_train_iters",
    ):
        root = Path(__file__).resolve().parents[1]
        criterion = {
            "criterion_train_mode_code": torch.tensor(2),
            "criterion_scope_code": torch.tensor(1),
            "criterion_confidence_objective_code": torch.tensor(2),
            "criterion_positive_trust_margin": torch.tensor(0.02),
            "criterion_positive_trust_weight": torch.tensor(1.0),
            "criterion_queue_size": torch.tensor(512),
            "criterion_queue_min_count": torch.tensor(256),
            "fpr_positive_queue": torch.zeros(512),
            "fpr_negative_queue": torch.zeros(512),
            "fpr_queue_count": torch.tensor(400),
            "fpr_queue_ptr": torch.tensor(400),
        }
        args = {
            "config_file": str(
                root
                / "config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py"
            ),
            "datasets": str(
                root
                / "config/datasets_stageb_gdino_adapter_semantic_verified_pairs.json"
            ),
            "world_size": 2,
            "batch_size": 4,
            "distributed": True,
            "amp": True,
            "max_train_iters": max_target,
            "pretrain_model_path": "" if use_resume else str(source),
            "resume": str(source) if use_resume else "",
            "stage_b_gdino_adapter_train_mode": "confidence_only",
            "stage_b_gdino_tn_scope": "image_global_topk_verified",
            "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
            "stage_b_gdino_gate_pool_temperature": 0.01,
            "stage_b_gdino_gate_topk": 3,
            "stage_b_gdino_positive_trust_margin": 0.02,
            "stage_b_gdino_positive_trust_weight": 1.0,
            "stage_b_gdino_queue_size": 512,
            "stage_b_gdino_queue_min_count": 256,
        }
        torch.save(
            {
                "model": model,
                "criterion": criterion,
                "optimizer": {
                    "state": {},
                    "param_groups": [
                        {
                            "params": [0],
                            "lr": 3.0e-4,
                            "stage_b_gdino_branch": "confidence",
                        }
                    ],
                },
                "lr_scheduler": {},
                "scaler": {},
                "rng_state": {"torch": torch.tensor([1], dtype=torch.uint8)},
                "epoch_rng_state": {"torch": torch.tensor([2], dtype=torch.uint8)},
                "epoch": 0,
                "iteration": iteration,
                "epoch_finished": False,
                "checkpoint_reason": reason,
                "args": args,
            },
            path,
        )

    @staticmethod
    def _write_json(path, value):
        path.write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _initial_bundle(self, directory):
        bootstrap = directory / "bootstrap.pth"
        bootstrap.write_bytes(b"bootstrap")
        adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
        initial_model = {
            "base.weight": torch.randn(2, 3),
            **_adapter_state(adapter),
        }
        initial_model[ADAPTER_PREFIX + "rank_output.bias"].add_(0.2)
        initial_checkpoint = directory / "rank-initial.pth"
        self._checkpoint(
            initial_checkpoint,
            source=bootstrap,
            model=initial_model,
        )
        initial_record = checkpoint_record(initial_checkpoint)
        initial_audit_path = directory / "rank-initial.audit.json"
        self._write_json(
            initial_audit_path,
            {
                "schema": TWO_PHASE_SCHEMA,
                "kind": "milestone_checkpoint",
                "phase": "rank",
                "iteration": 50,
                "checkpoint": initial_record,
            },
        )
        preflight = make_preflight(
            source_kind="rank",
            source_checkpoint=initial_checkpoint,
            source_audit=initial_audit_path,
            world_size=2,
            per_gpu_batch=4,
        )
        preflight_path = directory / "semantic-preflight.json"
        self._write_json(preflight_path, preflight)
        return {
            "initial_checkpoint": initial_checkpoint,
            "initial_audit": initial_audit_path,
            "initial_model": initial_model,
            "initial_record": initial_record,
            "preflight": preflight,
            "preflight_path": preflight_path,
        }

    def _milestone_payload(
        self,
        *,
        checkpoint,
        preflight,
        preflight_path,
        iteration,
        source,
        segment_path,
        previous_audit_path=None,
    ):
        result = validate_checkpoint(
            checkpoint=checkpoint,
            preflight=preflight,
            expected_target=iteration,
            source_checkpoint=source,
            exact=True,
        )
        static = preflight["static"]
        return {
            "schema": SCHEMA,
            "kind": "milestone_checkpoint",
            "phase": "confidence",
            "confidence_protocol": "semantic_verified_topk_v1",
            "tn_scope": "image_global_topk_verified",
            "iteration": iteration,
            "global_batch": 8,
            "initialization_mode": result["initialization_mode"],
            "queue": result["queue"],
            "segment_lineage": file_record(segment_path),
            "source_checkpoint": file_record(source),
            "preflight": file_record(preflight_path),
            "previous_audit": (
                file_record(previous_audit_path) if previous_audit_path else None
            ),
            "objective_contract": static["objective_contract"],
            "config": static["config"],
            "config_import_chain": static["config_import_chain"],
            "datasets": static["datasets"],
            "data_audit": static["data_audit"],
            "annotation": static["annotation"],
            "code": static["code"],
            "orchestration": static["orchestration"],
            "checkpoint": result["record"],
        }

    def _first_milestone_bundle(self, directory):
        bundle = self._initial_bundle(directory)
        segment_path = directory / "segment-50.json"
        segment = make_segment_lineage(
            preflight_path=bundle["preflight_path"],
            expected_target=50,
            source_checkpoint=bundle["initial_checkpoint"],
            initialization_mode="pretrain",
            previous_audit_path=None,
            recovery_inspection_path=None,
        )
        self._write_json(segment_path, segment)
        trained_model = {
            key: value.clone() for key, value in bundle["initial_model"].items()
        }
        trained_model[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.3)
        checkpoint = directory / "semantic-50.pth"
        self._checkpoint(
            checkpoint,
            source=bundle["initial_checkpoint"],
            model=trained_model,
        )
        audit = self._milestone_payload(
            checkpoint=checkpoint,
            preflight=bundle["preflight"],
            preflight_path=bundle["preflight_path"],
            iteration=50,
            source=bundle["initial_checkpoint"],
            segment_path=segment_path,
        )
        audit_path = directory / "semantic-50.audit.json"
        self._write_json(audit_path, audit)
        bundle.update(
            {
                "segment": segment,
                "segment_path": segment_path,
                "trained_model": trained_model,
                "checkpoint": checkpoint,
                "audit": audit,
                "audit_path": audit_path,
            }
        )
        return bundle

    def test_static_protocol_audit(self):
        result = validate_static()
        self.assertEqual(result["tn_scope"], "image_global_topk_verified")
        self.assertEqual(result["annotation"]["rows"], 17_829)
        self.assertEqual(result["resolved_contract"]["data_aug_hflip_prob"], 0.0)
        self.assertEqual(
            result["resolved_contract"]["stage_b_gdino_confidence_objective"],
            "detached_recent_q05_trust",
        )
        self.assertEqual(result["resolved_contract"]["stage_b_gdino_queue_size"], 512)
        self.assertEqual(result["objective_contract"]["mode_code"], 2)
        self.assertEqual(
            result["objective_contract"]["threshold_gradient"],
            "zero_value_global_positive_gate_mean_translation_proxy",
        )
        self.assertEqual(
            result["objective_contract"]["positive_trust_loss"],
            "mean_relu(-margin-positive_gate)",
        )
        root = Path(__file__).resolve().parents[1]
        dataset = json.loads(
            (
                root
                / "config/datasets_stageb_gdino_adapter_semantic_verified_pairs.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(dataset["train"][0]["neg_episode_prob"], 0.0)

    def test_semantic_source_accepts_r5000_but_not_an_extended_c_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bootstrap = directory / "bootstrap.pth"
            bootstrap.write_bytes(b"bootstrap")
            adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
            model = {"base.weight": torch.randn(2, 3), **_adapter_state(adapter)}
            model[ADAPTER_PREFIX + "rank_output.bias"].add_(0.2)
            model[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.1)
            checkpoint = directory / "r5000.pth"
            self._checkpoint(
                checkpoint,
                source=bootstrap,
                model=model,
                iteration=5000,
                max_target=5000,
            )
            record = checkpoint_record(checkpoint)
            audit_path = directory / "source.audit.json"
            self._write_json(
                audit_path,
                {
                    "schema": TWO_PHASE_SCHEMA,
                    "kind": "milestone_checkpoint",
                    "phase": "rank",
                    "iteration": 5000,
                    "checkpoint": record,
                },
            )
            self.assertEqual(
                _validate_source("rank", checkpoint, audit_path), record
            )

            self._write_json(
                audit_path,
                {
                    "schema": TWO_PHASE_SCHEMA,
                    "kind": "milestone_checkpoint",
                    "phase": "confidence",
                    "iteration": 1000,
                    "checkpoint": record,
                },
            )
            with self.assertRaisesRegex(
                SemanticProbeError, "audited confidence milestone"
            ):
                _validate_source("dataft-confidence", checkpoint, audit_path)

    def test_cross_scope_requires_pretrain_and_q05_queue_is_warm(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "rank.pth"
            source.write_bytes(b"source")
            adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
            initial_model = {
                "base.weight": torch.randn(2, 3),
                **_adapter_state(adapter),
            }
            initial = {"path": str(source), **model_hash_record(initial_model)}
            trained = {key: value.clone() for key, value in initial_model.items()}
            trained[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.25)
            checkpoint = directory / "semantic.pth"
            self._checkpoint(checkpoint, source=source, model=trained)
            static = validate_static()
            preflight = {
                "initial_checkpoint": initial,
                "static": static,
            }
            result = validate_checkpoint(
                checkpoint=checkpoint,
                preflight=preflight,
                expected_target=50,
                source_checkpoint=source,
                exact=True,
            )
            self.assertEqual(result["initialization_mode"], "pretrain_model_path")
            self.assertGreaterEqual(result["queue"]["count"], 256)

            self._checkpoint(
                checkpoint, source=source, model=trained, use_resume=True
            )
            with self.assertRaisesRegex(SemanticProbeError, "cross-scope"):
                validate_checkpoint(
                    checkpoint=checkpoint,
                    preflight=preflight,
                    expected_target=50,
                    source_checkpoint=source,
                    exact=True,
                )

    def test_checkpoint_rejects_a_non_p3_objective_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "rank.pth"
            source.write_bytes(b"source")
            adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
            initial_model = {
                "base.weight": torch.randn(2, 3),
                **_adapter_state(adapter),
            }
            initial = {"path": str(source), **model_hash_record(initial_model)}
            trained = {key: value.clone() for key, value in initial_model.items()}
            trained[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.25)
            checkpoint = directory / "semantic.pth"
            self._checkpoint(checkpoint, source=source, model=trained)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            payload["criterion"]["criterion_confidence_objective_code"] = torch.tensor(1)
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(SemanticProbeError, "P3 objective"):
                validate_checkpoint(
                    checkpoint=checkpoint,
                    preflight={
                        "initial_checkpoint": initial,
                        "static": validate_static(),
                    },
                    expected_target=50,
                    source_checkpoint=source,
                    exact=True,
                )

    def test_formal_evaluation_recomputes_all_lineage_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = self._first_milestone_bundle(directory)
            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=bundle["initial_record"],
            ):
                result = verify_evaluation_checkpoint(
                    checkpoint_path=bundle["checkpoint"],
                    audit_path=bundle["audit_path"],
                )
            self.assertTrue(result["verified"])
            self.assertEqual(result["tn_scope"], "image_global_topk_verified")
            self.assertEqual(result["lineage_replay"]["milestones"], 1)
            self.assertEqual(result["lineage_replay"]["segments"], 1)

            bundle["audit"]["checkpoint"]["confidence_sha256"] = "0" * 64
            self._write_json(bundle["audit_path"], bundle["audit"])
            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=bundle["initial_record"],
            ), self.assertRaisesRegex(SemanticProbeError, "checkpoint content"):
                verify_evaluation_checkpoint(
                    checkpoint_path=bundle["checkpoint"],
                    audit_path=bundle["audit_path"],
                )

    def test_formal_rejects_an_arbitrary_first_segment_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = self._first_milestone_bundle(directory)
            fork = directory / "unrelated-source.pth"
            fork.write_bytes(bundle["initial_checkpoint"].read_bytes())
            bundle["segment"]["source_checkpoint"] = file_record(fork)
            self._write_json(bundle["segment_path"], bundle["segment"])
            bundle["audit"]["segment_lineage"] = file_record(
                bundle["segment_path"]
            )
            bundle["audit"]["source_checkpoint"] = file_record(fork)
            self._write_json(bundle["audit_path"], bundle["audit"])
            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=bundle["initial_record"],
            ), self.assertRaisesRegex(
                SemanticProbeError, "first semantic segment did not pretrain"
            ):
                verify_evaluation_checkpoint(
                    checkpoint_path=bundle["checkpoint"],
                    audit_path=bundle["audit_path"],
                )

    def test_formal_rejects_forged_recovery_inspection_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = self._initial_bundle(directory)
            prior_segment_path = directory / "segment-before-recovery.json"
            prior_segment = make_segment_lineage(
                preflight_path=bundle["preflight_path"],
                expected_target=50,
                source_checkpoint=bundle["initial_checkpoint"],
                initialization_mode="pretrain",
                previous_audit_path=None,
                recovery_inspection_path=None,
            )
            self._write_json(prior_segment_path, prior_segment)
            partial_model = {
                key: value.clone() for key, value in bundle["initial_model"].items()
            }
            partial_model[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.15)
            live = directory / "checkpoint_iter.pth"
            self._checkpoint(
                live,
                source=bundle["initial_checkpoint"],
                model=partial_model,
                iteration=25,
                max_target=50,
                reason="interval",
            )
            partial = validate_checkpoint(
                checkpoint=live,
                preflight=bundle["preflight"],
                expected_target=50,
                source_checkpoint=bundle["initial_checkpoint"],
                exact=False,
            )
            inspection = {
                "schema": SCHEMA,
                "kind": "live_checkpoint_inspection",
                "phase": "confidence",
                "confidence_protocol": "semantic_verified_topk_v1",
                "tn_scope": "image_global_topk_verified",
                "iteration": 25,
                "expected_target": 50,
                "initialization_mode": partial["initialization_mode"],
                "queue": partial["queue"],
                "segment_lineage": file_record(prior_segment_path),
                "checkpoint": partial["record"],
            }
            inspection_path = directory / "live.audit.json"
            self._write_json(inspection_path, inspection)
            recovery = directory / "recovery.pth"
            recovery.write_bytes(live.read_bytes())
            recovery_segment_path = directory / "segment-recovery.json"
            recovery_segment = make_segment_lineage(
                preflight_path=bundle["preflight_path"],
                expected_target=50,
                source_checkpoint=recovery,
                initialization_mode="resume",
                previous_audit_path=None,
                recovery_inspection_path=inspection_path,
            )
            self._write_json(recovery_segment_path, recovery_segment)
            # The live path is ephemeral and is normally overwritten by the
            # resumed segment. Formal replay must use the sealed recovery copy.
            live.write_bytes(b"overwritten-live-path")
            final_model = {key: value.clone() for key, value in partial_model.items()}
            final_model[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.1)
            checkpoint = directory / "semantic-recovered-50.pth"
            self._checkpoint(
                checkpoint,
                source=recovery,
                model=final_model,
                use_resume=True,
            )
            audit = self._milestone_payload(
                checkpoint=checkpoint,
                preflight=bundle["preflight"],
                preflight_path=bundle["preflight_path"],
                iteration=50,
                source=recovery,
                segment_path=recovery_segment_path,
            )
            audit_path = directory / "semantic-recovered-50.audit.json"
            self._write_json(audit_path, audit)
            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=bundle["initial_record"],
            ):
                valid = verify_evaluation_checkpoint(
                    checkpoint_path=checkpoint, audit_path=audit_path
                )
            self.assertEqual(valid["lineage_replay"]["recovery_inspections"], 1)

            inspection["checkpoint"]["checkpoint_args"]["pretrain_model_path"] = (
                str(directory / "forged-parent.pth")
            )
            self._write_json(inspection_path, inspection)
            recovery_segment["recovery_inspection"] = file_record(inspection_path)
            self._write_json(recovery_segment_path, recovery_segment)
            audit["segment_lineage"] = file_record(recovery_segment_path)
            self._write_json(audit_path, audit)
            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=bundle["initial_record"],
            ), self.assertRaisesRegex(
                SemanticProbeError, "recovery checkpoint content"
            ):
                verify_evaluation_checkpoint(
                    checkpoint_path=checkpoint, audit_path=audit_path
                )

    def test_formal_rejects_disconnected_previous_milestone(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = self._first_milestone_bundle(directory)
            disconnected = directory / "copied-but-disconnected-50.pth"
            disconnected.write_bytes(first["checkpoint"].read_bytes())
            segment_path = directory / "segment-100-disconnected.json"
            segment = {
                "schema": SCHEMA,
                "kind": "segment_lineage",
                "phase": "confidence",
                "confidence_protocol": "semantic_verified_topk_v1",
                "tn_scope": "image_global_topk_verified",
                "expected_target": 100,
                "initialization_mode": "resume",
                "ancestry": "previous_milestone",
                "source_checkpoint": file_record(disconnected),
                "preflight": file_record(first["preflight_path"]),
                "previous_audit": file_record(first["audit_path"]),
                "recovery_inspection": None,
            }
            self._write_json(segment_path, segment)
            model100 = {
                key: value.clone() for key, value in first["trained_model"].items()
            }
            model100[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.1)
            checkpoint100 = directory / "semantic-100.pth"
            self._checkpoint(
                checkpoint100,
                source=disconnected,
                model=model100,
                use_resume=True,
                iteration=100,
                max_target=100,
            )
            audit100 = self._milestone_payload(
                checkpoint=checkpoint100,
                preflight=first["preflight"],
                preflight_path=first["preflight_path"],
                iteration=100,
                source=disconnected,
                segment_path=segment_path,
                previous_audit_path=first["audit_path"],
            )
            audit100_path = directory / "semantic-100.audit.json"
            self._write_json(audit100_path, audit100)
            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=first["initial_record"],
            ), self.assertRaisesRegex(
                SemanticProbeError, "did not resume from the previous milestone"
            ):
                verify_evaluation_checkpoint(
                    checkpoint_path=checkpoint100, audit_path=audit100_path
                )

    def test_formal_rejects_config_parent_chain_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = self._first_milestone_bundle(directory)
            root = Path(__file__).resolve().parents[1]
            parent = (
                root
                / "config/ablations/"
                "cfg_stageb_gdino_score_adapter_dataft.py"
            )
            self.assertIn(
                str(parent.resolve()),
                [row["path"] for row in bundle["preflight"]["static"]["config_import_chain"]],
            )

            def drifted_record(path):
                record = file_record(path)
                if Path(record["path"]) == parent.resolve():
                    record["sha256"] = "0" * 64
                return record

            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=bundle["initial_record"],
            ), patch(
                "tools.stageb_gdino_semantic_probe_audit.file_record",
                side_effect=drifted_record,
            ), self.assertRaisesRegex(SemanticProbeError, "lineage drifted"):
                verify_evaluation_checkpoint(
                    checkpoint_path=bundle["checkpoint"],
                    audit_path=bundle["audit_path"],
                )

    def test_formal_rejects_local_python_dependency_closure_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = self._first_milestone_bundle(directory)
            root = Path(__file__).resolve().parents[1]
            dependency = root / "models/GroundingDINO/utils.py"
            self.assertIn(
                str(dependency.resolve()),
                [row["path"] for row in bundle["preflight"]["static"]["code"]],
            )

            def drifted_record(path):
                record = file_record(path)
                if Path(record["path"]) == dependency.resolve():
                    record["sha256"] = "f" * 64
                return record

            with patch(
                "tools.stageb_gdino_semantic_probe_audit."
                "_verify_two_phase_source_milestone",
                return_value=bundle["initial_record"],
            ), patch(
                "tools.stageb_gdino_semantic_probe_audit.file_record",
                side_effect=drifted_record,
            ), self.assertRaisesRegex(SemanticProbeError, "lineage drifted"):
                verify_evaluation_checkpoint(
                    checkpoint_path=bundle["checkpoint"],
                    audit_path=bundle["audit_path"],
                )

    def test_recovery_lineage_rejects_a_fork_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial = directory / "rank.pth"
            initial.write_bytes(b"audited-rank")
            preflight_path = directory / "preflight.json"
            preflight_path.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "phase_preflight",
                        "phase": "semantic-confidence",
                        "initial_checkpoint": file_record(initial),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            first_segment_path = directory / "segment-01.json"
            first_segment = make_segment_lineage(
                preflight_path=preflight_path,
                expected_target=50,
                source_checkpoint=initial,
                initialization_mode="pretrain",
                previous_audit_path=None,
                recovery_inspection_path=None,
            )
            first_segment_path.write_text(
                json.dumps(first_segment, sort_keys=True) + "\n", encoding="utf-8"
            )

            adapter = StageBGDINOScoreAdapter(8, adapter_dim=4, gate_hidden_dim=4)
            model = {"base.weight": torch.randn(2, 3), **_adapter_state(adapter)}
            model[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.2)
            live = directory / "checkpoint_iter.pth"
            self._checkpoint(live, source=initial, model=model)
            from tools.stageb_gdino_adapter_probe_audit import checkpoint_record

            inspection_path = directory / "live.audit.json"
            inspection_path.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "kind": "live_checkpoint_inspection",
                        "phase": "confidence",
                        "confidence_protocol": "semantic_verified_topk_v1",
                        "tn_scope": "image_global_topk_verified",
                        "expected_target": 50,
                        "segment_lineage": file_record(first_segment_path),
                        "checkpoint": checkpoint_record(live),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            recovery = directory / "recovery.pth"
            recovery.write_bytes(live.read_bytes())
            valid = make_segment_lineage(
                preflight_path=preflight_path,
                expected_target=50,
                source_checkpoint=recovery,
                initialization_mode="resume",
                previous_audit_path=None,
                recovery_inspection_path=inspection_path,
            )
            self.assertEqual(valid["ancestry"], "audited_live_recovery")

            fork = directory / "fork.pth"
            fork_model = {key: value.clone() for key, value in model.items()}
            fork_model[ADAPTER_PREFIX + "confidence_gate.4.bias"].add_(0.1)
            self._checkpoint(fork, source=initial, model=fork_model)
            with self.assertRaisesRegex(
                SemanticProbeError, "does not match the audited live"
            ):
                make_segment_lineage(
                    preflight_path=preflight_path,
                    expected_target=50,
                    source_checkpoint=fork,
                    initialization_mode="resume",
                    previous_audit_path=None,
                    recovery_inspection_path=inspection_path,
                )


if __name__ == "__main__":
    unittest.main()
