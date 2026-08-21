import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.eval_mmgdino_e5_ownership_cache import (
    binary_metrics,
    evaluate_cache,
    exact_q05,
)
from tools.extract_mmgdino_e5_eval_cache import EVAL_CACHE_SCHEMA
from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
)
from tools.responsibility_isolation_cache import (
    CACHE_BOX_FORMAT,
    CACHE_FEATURE_DIM,
    CACHE_ROW_SCHEMA,
    CACHE_SOURCE_SCHEMA,
    CACHE_TASK_CONFIDENCE_PAIR,
    CACHE_TASK_RANK,
    CachedCandidateContractError,
    validate_cached_candidate_row,
)
from tools.train_mmgdino_e5_ownership import CHECKPOINT_SCHEMA


class OwnershipCacheEvalTests(unittest.TestCase):
    def _source(self):
        return {
            "schema": CACHE_SOURCE_SCHEMA,
            "model_id": "synthetic",
            "checkpoint_sha256": "0" * 64,
            "config_sha256": "1" * 64,
            "extractor_code_sha256": "2" * 64,
            "query_feature_name": "synthetic",
        }

    def _base(self, sample_id, image_id, task):
        return {
            "schema": CACHE_ROW_SCHEMA,
            "sample_id": sample_id,
            "image_id": image_id,
            "task": task,
            "query_features": torch.zeros(3, 256),
            "native_score": torch.tensor([0.2, 0.9, 0.1]),
            "boxes": torch.tensor(
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.1, 0.1, 0.1, 0.1],
                    [0.8, 0.8, 0.1, 0.1],
                ]
            ),
            "candidate_mask": torch.ones(3, dtype=torch.bool),
        }

    def _save_eval_cache(self, path, task, rows):
        torch.save(
            {
                "schema": EVAL_CACHE_SCHEMA,
                "surface": "synthetic",
                "task": task,
                "source": self._source(),
                "feature_dim": CACHE_FEATURE_DIM,
                "box_format": CACHE_BOX_FORMAT,
                "rows": tuple(rows),
            },
            path,
        )

    def test_q05_and_fixed_threshold_use_greater_equal(self):
        positive = np.arange(20, dtype=np.float64)
        negative = np.asarray([0.0, 1.0, 2.0])
        self.assertEqual(exact_q05(positive), 1.0)
        metrics = binary_metrics(positive, negative, fixed_threshold=1.0)
        self.assertEqual(metrics["fixed_tpr"], 19 / 20)
        self.assertEqual(metrics["fixed_fpr"], 2 / 3)

    def test_zero_initialized_isolated_rank_matches_native_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._base("rank:0", "10", CACHE_TASK_RANK)
            row["gt_boxes"] = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
            cache = root / "ref.pt"
            self._save_eval_cache(cache, CACHE_TASK_RANK, (row,))
            module = MMGDinoE5ResponsibilityOwners(
                ownership=OWNERSHIP_ISOLATED_128
            )
            checkpoint = root / "owner.pt"
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "config": {"ownership": OWNERSHIP_ISOLATED_128, "seed": 17},
                    "model_state_dict": module.state_dict(),
                },
                checkpoint,
            )
            native = evaluate_cache(
                cache_path=cache,
                route="native",
                surface="ref",
                output_dir=root / "native",
                device="cpu",
            )
            learned = evaluate_cache(
                cache_path=cache,
                route=OWNERSHIP_ISOLATED_128,
                surface="ref",
                output_dir=root / "learned",
                checkpoint_path=checkpoint,
                device="cpu",
            )
            self.assertEqual(native["metrics"], learned["metrics"])
            self.assertEqual(native["metrics"]["p1_iou50"], 1.0)

    def test_eval_counts_missing_oracle_as_failure_without_weakening_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._base("rank:no-oracle", "11", CACHE_TASK_RANK)
            row["gt_boxes"] = torch.tensor([[0.45, 0.45, 0.05, 0.05]])
            with self.assertRaisesRegex(
                CachedCandidateContractError, "IoU>=0.5 positives"
            ):
                validate_cached_candidate_row(row)
            validate_cached_candidate_row(
                row, require_trainable_rank_pair=False
            )
            cache = root / "ref-no-oracle.pt"
            self._save_eval_cache(cache, CACHE_TASK_RANK, (row,))
            summary = evaluate_cache(
                cache_path=cache,
                route="native",
                surface="ref",
                output_dir=root / "native-no-oracle",
                device="cpu",
            )
            self.assertEqual(summary["metrics"]["oracle_iou50"], 0.0)
            self.assertEqual(summary["metrics"]["p1_iou50"], 0.0)

    def test_native_tn_metrics_use_sample_max(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, (positive_max, negative_max) in enumerate(
                ((0.9, 0.1), (0.8, 0.2))
            ):
                pair_id = f"pair:{index}"
                for role, score in (("positive", positive_max), ("negative", negative_max)):
                    row = self._base(
                        f"{pair_id}:{role}", str(index), CACHE_TASK_CONFIDENCE_PAIR
                    )
                    row["native_score"] = torch.tensor([score, 0.0, 0.0])
                    row["gt_boxes"] = (
                        torch.tensor([[0.5, 0.5, 0.2, 0.2]])
                        if role == "positive"
                        else torch.empty(0, 4)
                    )
                    row["pair_id"] = pair_id
                    row["pair_role"] = role
                    rows.append(row)
            cache = root / "tn.pt"
            self._save_eval_cache(cache, CACHE_TASK_CONFIDENCE_PAIR, rows)
            summary = evaluate_cache(
                cache_path=cache,
                route="native",
                surface="tn",
                output_dir=root / "eval",
                device="cpu",
            )
            self.assertEqual(summary["metrics"]["auroc"], 1.0)
            self.assertEqual(summary["metrics"]["fpr95"], 0.0)


if __name__ == "__main__":
    unittest.main()
