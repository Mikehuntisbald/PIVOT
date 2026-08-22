import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_128,
)
from tools.responsibility_isolation_cache import (
    CACHE_BOX_FORMAT,
    CACHE_FEATURE_DIM,
    CACHE_ROW_SCHEMA,
    CACHE_SHARD_SCHEMA,
    CACHE_SOURCE_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    save_cached_candidate_shard,
)
from tools.train_mmgdino_e5_ownership import (
    D3QueueState,
    FORMAL_CONFIDENCE_BATCH_SIZE,
    FORMAL_CONFIDENCE_UPDATES,
    FORMAL_RANK_BATCH_SIZE,
    FORMAL_RANK_UPDATES,
    FormalConfig,
    SCHEDULE_SCHEMA,
    _rank_loss,
    run_formal_training,
    validate_schedule,
)


ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64
TWO_SHA = "2" * 64


def _rank_row(index: int):
    generator = torch.Generator().manual_seed(1000 + index)
    return {
        "schema": CACHE_ROW_SCHEMA,
        "sample_id": f"rank:{index:05d}",
        "image_id": str(index // 2),
        "task": CACHE_TASK_RANK,
        "query_features": torch.randn(4, 256, generator=generator),
        "native_score": torch.tensor([0.2, 0.8, 0.1, 0.3]),
        "boxes": torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1],
                [0.8, 0.8, 0.1, 0.1],
                [0.3, 0.7, 0.1, 0.1],
            ]
        ),
        "candidate_mask": torch.ones(4, dtype=torch.bool),
        "gt_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }


def _pair_rows(index: int):
    pair_id = f"pair:{index:04d}"
    rows = []
    for role, offset in (("positive", 2000), ("negative", 3000)):
        generator = torch.Generator().manual_seed(offset + index)
        rows.append(
            {
                "schema": CACHE_ROW_SCHEMA,
                "sample_id": f"{pair_id}:{role}",
                "image_id": str(10000 + index),
                "task": CACHE_TASK_CONFIDENCE_PAIR,
                "query_features": torch.randn(4, 256, generator=generator),
                "native_score": torch.tensor([0.3, 0.6, 0.2, 0.1]),
                "boxes": torch.tensor(
                    [
                        [0.5, 0.5, 0.2, 0.2],
                        [0.1, 0.1, 0.1, 0.1],
                        [0.8, 0.8, 0.1, 0.1],
                        [0.3, 0.7, 0.1, 0.1],
                    ]
                ),
                "candidate_mask": torch.ones(4, dtype=torch.bool),
                "gt_boxes": (
                    torch.tensor([[0.5, 0.5, 0.2, 0.2]])
                    if role == "positive"
                    else torch.empty(0, 4)
                ),
                "pair_id": pair_id,
                "pair_role": role,
            }
        )
    return rows


def _schedule(seed: int = 17):
    updates = []
    rank_cursor = 0
    pair_cursor = 0
    for update in range(1, 151):
        if (update - 1) % 3 == 1:
            identities = [
                f"pair:{index:04d}"
                for index in range(
                    pair_cursor, pair_cursor + FORMAL_CONFIDENCE_BATCH_SIZE
                )
            ]
            pair_cursor += FORMAL_CONFIDENCE_BATCH_SIZE
            task = CACHE_TASK_CONFIDENCE_PAIR
        else:
            identities = [
                f"rank:{index:05d}"
                for index in range(rank_cursor, rank_cursor + FORMAL_RANK_BATCH_SIZE)
            ]
            rank_cursor += FORMAL_RANK_BATCH_SIZE
            task = CACHE_TASK_RANK
        updates.append({"update": update, "task": task, "identities": identities})
    assert rank_cursor == FORMAL_RANK_UPDATES * FORMAL_RANK_BATCH_SIZE
    assert pair_cursor == FORMAL_CONFIDENCE_UPDATES * FORMAL_CONFIDENCE_BATCH_SIZE
    return {
        "schema": SCHEDULE_SCHEMA,
        "seed": seed,
        "source": {
            "rank_jsonl_sha256": ZERO_SHA,
            "d3_jsonl_sha256": ONE_SHA,
        },
        "rank_batch_size": FORMAL_RANK_BATCH_SIZE,
        "confidence_batch_size": FORMAL_CONFIDENCE_BATCH_SIZE,
        "updates": updates,
    }


class FormalOwnershipTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        schedule = validate_schedule(_schedule())
        cls.schedule_path = cls.root / "schedule.json"
        cls.schedule_path.write_text(
            json.dumps(schedule, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows = [_rank_row(index) for index in range(3200)]
        for index in range(400):
            rows.extend(_pair_rows(index))
        shard = {
            "schema": CACHE_SHARD_SCHEMA,
            "shard_id": "synthetic-formal-seed17",
            "source": {
                "schema": CACHE_SOURCE_SCHEMA,
                "model_id": "synthetic-e5",
                "checkpoint_sha256": ZERO_SHA,
                "config_sha256": ONE_SHA,
                "extractor_code_sha256": TWO_SHA,
                "query_feature_name": "synthetic",
            },
            "feature_dim": CACHE_FEATURE_DIM,
            "box_format": CACHE_BOX_FORMAT,
            "rows": tuple(rows),
        }
        cls.cache_path = cls.root / "cache.pt"
        save_cached_candidate_shard(shard, cls.cache_path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_queue_fifo_and_warmup(self):
        queue = D3QueueState.empty(size=4)
        self.assertIsNone(queue.history(min_count=3, device=torch.device("cpu")))
        queue.append(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(
            torch.equal(
                queue.history(min_count=3, device=torch.device("cpu")),
                torch.tensor([1.0, 2.0, 3.0]),
            )
        )
        queue.append(torch.tensor([4.0, 5.0]))
        self.assertTrue(
            torch.equal(
                queue.history(min_count=3, device=torch.device("cpu")),
                torch.tensor([2.0, 3.0, 4.0, 5.0]),
            )
        )

    def test_no_eligible_positive_preserves_row_with_zero_rank_margin(self):
        row = _rank_row(0)
        row["candidate_mask"] = torch.tensor([False, True, True, True])
        module = MMGDinoE5ResponsibilityOwners(
            ownership=OWNERSHIP_SHARED_128
        )
        loss, metrics = _rank_loss(
            module, [row], device=torch.device("cpu")
        )
        self.assertEqual(metrics["valid_rows"], 0.0)
        self.assertEqual(metrics["rows_no_positive"], 1.0)
        self.assertEqual(metrics["fix_loss"], 0.0)
        self.assertEqual(metrics["preserve_loss"], 0.0)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_shared_run_has_two_optimizer_states_and_zero_decay(self):
        output = self.root / "shared"
        receipt = run_formal_training(
            cache_path=self.cache_path,
            schedule_path=self.schedule_path,
            output_dir=output,
            config=FormalConfig(
                ownership=OWNERSHIP_SHARED_128, seed=17, device="cpu"
            ),
        )
        self.assertEqual(receipt["updates"]["rank"], 100)
        self.assertEqual(receipt["updates"]["confidence"], 50)
        self.assertEqual(receipt["optimizers"]["weight_decay"], 0.0)
        self.assertGreater(receipt["optimizers"]["rank_state_tensor_count"], 0)
        self.assertGreater(
            receipt["optimizers"]["confidence_state_tensor_count"], 0
        )
        self.assertEqual(receipt["d3_queue"]["count"], 400)
        self.assertFalse(receipt["ownership"]["structurally_isolated"])
        self.assertEqual(set(receipt["gradient_probes"]), {"25", "50", "100", "150"})

    def test_isolated_resume_matches_uninterrupted_model_state(self):
        direct_dir = self.root / "isolated-direct"
        direct = run_formal_training(
            cache_path=self.cache_path,
            schedule_path=self.schedule_path,
            output_dir=direct_dir,
            config=FormalConfig(
                ownership=OWNERSHIP_ISOLATED_128, seed=17, device="cpu"
            ),
        )
        resumed = run_formal_training(
            cache_path=self.cache_path,
            schedule_path=self.schedule_path,
            output_dir=self.root / "isolated-resumed",
            config=FormalConfig(
                ownership=OWNERSHIP_ISOLATED_128, seed=17, device="cpu"
            ),
            resume_from=direct_dir / "checkpoint_u050.pt",
        )
        self.assertEqual(
            direct["checkpoint"]["model_state_sha256"],
            resumed["checkpoint"]["model_state_sha256"],
        )
        self.assertTrue(resumed["ownership"]["structurally_isolated"])
        for probe in resumed["gradient_probes"].values():
            self.assertTrue(probe["structurally_isolated"])
            self.assertEqual(probe["cross_task_parameter_count"], 0)


if __name__ == "__main__":
    unittest.main()
