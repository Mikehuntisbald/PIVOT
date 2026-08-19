import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.responsibility_isolation import (
    RESPONSIBILITY_OWNERSHIP_ISOLATED,
    RESPONSIBILITY_OWNERSHIP_SHARED,
    FrozenCandidateResponsibilityHeads,
)
from tools.responsibility_isolation_cache import (
    CACHE_FEATURE_DIM,
    CACHE_ROW_SCHEMA,
    CACHE_SHARD_SCHEMA,
    CACHE_SOURCE_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    CachedCandidateContractError,
    build_synthetic_cached_candidate_shard,
    cached_candidate_content_sha256,
    file_sha256,
    load_cached_candidate_shard,
    validate_cached_candidate_shard,
)
from tools.train_responsibility_isolation_pilot import (
    CONFIDENCE_LOSS_CONTRACT,
    PILOT_CHECKPOINT_SCHEMA,
    PILOT_RECEIPT_SCHEMA,
    PILOT_SCHEDULE_SCHEMA,
    RANK_LOSS_CONTRACT,
    PilotConfig,
    PilotContractError,
    build_interleaved_exposure_schedule,
    confidence_pair_sample_max_bce_loss,
    rank_listwise_preserve_loss,
    run_cached_feature_pilot,
)


def _nested_equal(testcase, left, right):
    if torch.is_tensor(left):
        testcase.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right))
        for key in left:
            _nested_equal(testcase, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        testcase.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _nested_equal(testcase, left_item, right_item)
    else:
        testcase.assertEqual(left, right)


class CachedCandidateShardContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / "synthetic_cache.pt"
        self.hashes = build_synthetic_cached_candidate_shard(self.cache)
        self.shard = load_cached_candidate_shard(self.cache)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixture_binds_versioned_rows_tensor_contract_and_hashes(self):
        self.assertEqual(self.shard["schema"], CACHE_SHARD_SCHEMA)
        self.assertEqual(self.shard["feature_dim"], CACHE_FEATURE_DIM)
        self.assertEqual(self.shard["source"]["schema"], CACHE_SOURCE_SCHEMA)
        self.assertEqual(
            self.shard["source"]["model_id"],
            "synthetic-frozen-candidate-fixture",
        )
        self.assertEqual(len(self.shard["source"]["checkpoint_sha256"]), 64)
        self.assertEqual(len(self.shard["rows"]), 6)
        self.assertEqual(self.hashes["file_sha256"], file_sha256(self.cache))
        self.assertEqual(
            self.hashes["content_sha256"],
            cached_candidate_content_sha256(self.shard),
        )
        tasks = [row["task"] for row in self.shard["rows"]]
        self.assertEqual(tasks.count(CACHE_TASK_RANK), 2)
        self.assertEqual(tasks.count(CACHE_TASK_CONFIDENCE_PAIR), 4)
        for row in self.shard["rows"]:
            self.assertEqual(row["schema"], CACHE_ROW_SCHEMA)
            self.assertIn(row["query_features"].dtype, (torch.float16, torch.float32))
            self.assertEqual(tuple(row["query_features"].shape), (4, 256))
            self.assertEqual(tuple(row["native_score"].shape), (4,))
            self.assertEqual(tuple(row["boxes"].shape), (4, 4))
            self.assertEqual(row["candidate_mask"].dtype, torch.bool)

    def test_rank_oracle_pair_closure_and_unknown_fields_fail_closed(self):
        rank_broken = copy.deepcopy(self.shard)
        rank_broken["rows"][0]["gt_boxes"] = torch.empty(0, 4)
        with self.assertRaisesRegex(CachedCandidateContractError, "rank row"):
            validate_cached_candidate_shard(rank_broken)

        pair_broken = copy.deepcopy(self.shard)
        pair_broken["rows"] = tuple(
            row
            for row in pair_broken["rows"]
            if row["sample_id"] != "confidence-0-negative"
        )
        with self.assertRaisesRegex(CachedCandidateContractError, "one positive"):
            validate_cached_candidate_shard(pair_broken)

        field_broken = copy.deepcopy(self.shard)
        field_broken["rows"][0]["unversioned_metadata"] = "forbidden"
        with self.assertRaisesRegex(CachedCandidateContractError, "extra"):
            validate_cached_candidate_shard(field_broken)

        source_broken = copy.deepcopy(self.shard)
        source_broken["source"]["checkpoint_sha256"] = "not-a-checkpoint-hash"
        with self.assertRaisesRegex(CachedCandidateContractError, "SHA-256"):
            validate_cached_candidate_shard(source_broken)

    def test_schedule_is_deterministic_prefix_stable_and_seed_closed(self):
        u1 = build_interleaved_exposure_schedule(self.shard, seed=17, updates=1)
        u5 = build_interleaved_exposure_schedule(self.shard, seed=17, updates=5)
        self.assertEqual(u1, u5[:1])
        self.assertEqual(
            [entry["task"] for entry in u5],
            [
                CACHE_TASK_RANK,
                CACHE_TASK_CONFIDENCE_PAIR,
                CACHE_TASK_RANK,
                CACHE_TASK_CONFIDENCE_PAIR,
                CACHE_TASK_RANK,
            ],
        )
        self.assertEqual(
            u5,
            build_interleaved_exposure_schedule(self.shard, seed=17, updates=5),
        )
        with self.assertRaisesRegex(PilotContractError, "seed"):
            build_interleaved_exposure_schedule(self.shard, seed=1, updates=5)


class PilotLossContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name) / "cache.pt"
        build_synthetic_cached_candidate_shard(self.cache)
        self.shard = load_cached_candidate_shard(self.cache)

    def tearDown(self):
        self.temporary.cleanup()

    def test_zero_init_rank_listwise_preserve_and_confidence_bce_are_finite(self):
        module = FrozenCandidateResponsibilityHeads(feature_dim=256, hidden_dim=16)
        rank_row = next(
            row for row in self.shard["rows"] if row["task"] == CACHE_TASK_RANK
        )
        rank_output = module(
            rank_row["query_features"].unsqueeze(0),
            rank_row["native_score"].unsqueeze(0),
            rank_row["candidate_mask"].unsqueeze(0),
        )
        rank_loss, rank_metrics = rank_listwise_preserve_loss(
            rank_output,
            rank_row,
            temperature=0.05,
            margin=0.05,
            hard_negative_count=32,
            preserve_tolerance=0.01,
            preserve_floor=0.005,
            preserve_weight=2.0,
        )
        self.assertTrue(torch.isfinite(rank_loss))
        self.assertEqual(rank_metrics["preserve"], 0.0)
        self.assertGreater(rank_metrics["positive_count"], 0)
        self.assertGreater(rank_metrics["negative_count"], 0)

        pair = {
            row["pair_role"]: row
            for row in self.shard["rows"]
            if row.get("pair_id") == "pair-0"
        }
        outputs = {}
        for role, row in pair.items():
            outputs[role] = module(
                row["query_features"].unsqueeze(0),
                row["native_score"].unsqueeze(0),
                row["candidate_mask"].unsqueeze(0),
            )
        confidence_loss, confidence_metrics = confidence_pair_sample_max_bce_loss(
            outputs["positive"], outputs["negative"]
        )
        self.assertAlmostEqual(
            float(confidence_loss.detach()), torch.log(torch.tensor(2.0)).item()
        )
        self.assertEqual(confidence_metrics["positive_sample_max_logit"], 0.0)
        self.assertEqual(confidence_metrics["negative_sample_max_logit"], 0.0)


class CachedFeaturePilotRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache.pt"
        build_synthetic_cached_candidate_shard(self.cache)

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, name, mode, updates, *, seed=17, resume_from=None):
        return run_cached_feature_pilot(
            cache_path=self.cache,
            output_dir=self.root / name,
            config=PilotConfig(
                ownership=mode,
                seed=seed,
                updates=updates,
                hidden_dim=16,
                device="cpu",
                amp=False,
            ),
            resume_from=resume_from,
        )

    def test_u1_shared_and_isolated_have_identical_exposure_and_complete_receipts(self):
        shared = self._run("u1-shared", RESPONSIBILITY_OWNERSHIP_SHARED, 1)
        isolated = self._run("u1-isolated", RESPONSIBILITY_OWNERSHIP_ISOLATED, 1)
        shared_receipt = shared["receipt"]
        isolated_receipt = isolated["receipt"]
        for receipt in (shared_receipt, isolated_receipt):
            self.assertEqual(receipt["schema"], PILOT_RECEIPT_SCHEMA)
            self.assertTrue(receipt["cache"]["unchanged"])
            self.assertEqual(receipt["cache"]["requires_grad_tensor_count"], 0)
            self.assertEqual(receipt["updates"]["attempted"], 1)
            self.assertEqual(receipt["updates"]["applied"], 1)
            self.assertEqual(receipt["updates"]["rank"], 1)
            self.assertEqual(receipt["updates"]["confidence"], 0)
            self.assertEqual(receipt["updates"]["nonfinite"], 0)
            self.assertFalse(receipt["amp"]["enabled"])
            self.assertEqual(receipt["amp"]["skip_count"], 0)
            self.assertEqual(receipt["config"]["rank_loss_contract"], RANK_LOSS_CONTRACT)
            self.assertEqual(
                receipt["config"]["confidence_loss_contract"],
                CONFIDENCE_LOSS_CONTRACT,
            )
            self.assertEqual(receipt["schedule"]["schema"], PILOT_SCHEDULE_SCHEMA)
            self.assertTrue(Path(receipt["checkpoint"]["path"]).is_file())
            on_disk = json.loads(Path(receipt["checkpoint"]["path"]).with_name(
                "run_receipt.json"
            ).read_text())
            self.assertEqual(on_disk["schema"], PILOT_RECEIPT_SCHEMA)
        self.assertEqual(
            shared_receipt["schedule"]["sha256"],
            isolated_receipt["schedule"]["sha256"],
        )
        self.assertEqual(
            shared_receipt["schedule"]["exposures"],
            isolated_receipt["schedule"]["exposures"],
        )
        self.assertGreater(
            shared_receipt["ownership"]["shared_tensor_count"], 0
        )
        self.assertEqual(
            isolated_receipt["ownership"]["shared_tensor_count"], 0
        )
        self.assertTrue(
            isolated_receipt["gradient_audit"]["structurally_isolated"]
        )

    def test_u5_shared_records_gradient_cosine_and_isolated_has_no_joint_path(self):
        shared = self._run("u5-shared", RESPONSIBILITY_OWNERSHIP_SHARED, 5)
        isolated = self._run("u5-isolated", RESPONSIBILITY_OWNERSHIP_ISOLATED, 5)
        shared_audit = shared["receipt"]["gradient_audit"]
        isolated_audit = isolated["receipt"]["gradient_audit"]
        self.assertTrue(shared_audit["joint_gradient_cosine_defined"])
        self.assertIsInstance(shared_audit["joint_gradient_cosine"], float)
        self.assertGreater(len(shared_audit["jointly_connected_parameter_names"]), 0)
        self.assertFalse(isolated_audit["joint_gradient_cosine_defined"])
        self.assertIsNone(isolated_audit["joint_gradient_cosine"])
        self.assertEqual(isolated_audit["jointly_connected_parameter_names"], ())
        self.assertTrue(isolated_audit["structurally_isolated"])
        self.assertEqual(shared["receipt"]["updates"]["rank"], 3)
        self.assertEqual(shared["receipt"]["updates"]["confidence"], 2)

    def test_u1_to_u5_resume_is_bitwise_deterministic_for_both_ownerships(self):
        for mode in (
            RESPONSIBILITY_OWNERSHIP_SHARED,
            RESPONSIBILITY_OWNERSHIP_ISOLATED,
        ):
            with self.subTest(mode=mode):
                full = self._run(f"{mode}-full", mode, 5, seed=42)
                u1 = self._run(f"{mode}-u1", mode, 1, seed=42)
                resumed = self._run(
                    f"{mode}-resumed",
                    mode,
                    5,
                    seed=42,
                    resume_from=u1["checkpoint_path"],
                )
                self.assertEqual(
                    full["receipt"]["checkpoint"]["model_state_sha256"],
                    resumed["receipt"]["checkpoint"]["model_state_sha256"],
                )
                self.assertEqual(
                    full["receipt"]["schedule"]["exposures"],
                    resumed["receipt"]["schedule"]["exposures"],
                )
                self.assertEqual(
                    full["receipt"]["loss_history"],
                    resumed["receipt"]["loss_history"],
                )
                full_checkpoint = torch.load(
                    full["checkpoint_path"], weights_only=True
                )
                resumed_checkpoint = torch.load(
                    resumed["checkpoint_path"], weights_only=True
                )
                self.assertEqual(
                    full_checkpoint["schema"], PILOT_CHECKPOINT_SCHEMA
                )
                _nested_equal(
                    self,
                    full_checkpoint["model_state_dict"],
                    resumed_checkpoint["model_state_dict"],
                )
                _nested_equal(
                    self,
                    full_checkpoint["optimizer_state_dict"],
                    resumed_checkpoint["optimizer_state_dict"],
                )

    def test_resume_rejects_training_contract_drift(self):
        u1 = self._run("drift-u1", RESPONSIBILITY_OWNERSHIP_ISOLATED, 1)
        with self.assertRaisesRegex(PilotContractError, "contract mismatch"):
            run_cached_feature_pilot(
                cache_path=self.cache,
                output_dir=self.root / "drift-resume",
                config=PilotConfig(
                    ownership=RESPONSIBILITY_OWNERSHIP_ISOLATED,
                    seed=17,
                    updates=5,
                    hidden_dim=32,
                ),
                resume_from=u1["checkpoint_path"],
            )


if __name__ == "__main__":
    unittest.main()
