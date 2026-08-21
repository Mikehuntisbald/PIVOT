import unittest

import torch

from tools.build_mmgdino_e5_cross_dataset_probe_manifests import (
    PROBE_BATCHES,
    PROBE_BATCH_SIZE,
    ROWS_PER_SEED,
    SEEDS,
    sample_id,
    select_schedule,
)
from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_SHARED_128,
)
from tools.probe_mmgdino_e5_cross_dataset_gradients import (
    _gradient_metrics,
    _rank_diagnostics,
)
from tools.responsibility_isolation_cache import CACHE_ROW_SCHEMA, CACHE_TASK_RANK


class CrossDatasetProbeTests(unittest.TestCase):
    def _manifest_rows(self, count=400):
        return [
            {
                "source": "refcoco_unc_val",
                "image_id": index,
                "ann_id": 1000 + index,
                "ref_id": 2000 + index,
                "sent_id": 3000 + index,
            }
            for index in range(count)
        ]

    def test_selection_is_deterministic_seed_closed_and_batched(self):
        first = select_schedule(self._manifest_rows())
        second = select_schedule(self._manifest_rows())
        self.assertEqual(first, second)
        self.assertEqual(first["seeds"], list(SEEDS))
        for seed in SEEDS:
            batches = first["batches"][str(seed)]
            self.assertEqual(len(batches), PROBE_BATCHES)
            self.assertTrue(all(len(batch) == PROBE_BATCH_SIZE for batch in batches))
            self.assertEqual(len({item for batch in batches for item in batch}), ROWS_PER_SEED)
        self.assertLessEqual(len(first["union_identities"]), len(SEEDS) * ROWS_PER_SEED)

    def test_identity_rejects_boolean_integer(self):
        row = self._manifest_rows(1)[0]
        expected = "refcoco:refcoco_unc_val:0:1000:2000:3000"
        self.assertEqual(sample_id(row), expected)
        row["image_id"] = True
        with self.assertRaisesRegex(RuntimeError, "image_id"):
            sample_id(row)

    def test_gradient_metrics_report_norm_cosine_and_sign_conflict(self):
        result = _gradient_metrics(
            torch.tensor([1.0, -2.0, 3.0]),
            torch.tensor([-1.0, -2.0, 3.0]),
        )
        self.assertAlmostEqual(result["rank_gradient_l2"], 14.0 ** 0.5, places=6)
        self.assertAlmostEqual(result["confidence_gradient_l2"], 14.0 ** 0.5, places=6)
        self.assertAlmostEqual(result["cosine"], 12.0 / 14.0, places=6)
        self.assertAlmostEqual(result["sign_conflict_fraction"], 1.0 / 3.0)

    def test_rank_diagnostics_define_native_top1_runnerup_margin(self):
        module = MMGDinoE5ResponsibilityOwners(ownership=OWNERSHIP_SHARED_128)
        row = {
            "schema": CACHE_ROW_SCHEMA,
            "sample_id": "rank:0",
            "image_id": "0",
            "task": CACHE_TASK_RANK,
            "query_features": torch.zeros(3, 256),
            "native_score": torch.tensor([0.9, 0.4, 0.2]),
            "boxes": torch.tensor(
                [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]
            ),
            "candidate_mask": torch.ones(3, dtype=torch.bool),
            "gt_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        }
        loss, metrics = _rank_diagnostics(module, [row], device=torch.device("cpu"))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["native_p1"], 1.0)
        self.assertAlmostEqual(metrics["native_top1_runnerup_margin"], 0.5)
        self.assertAlmostEqual(metrics["native_oracle_positive_negative_gap"], 0.5)


if __name__ == "__main__":
    unittest.main()
