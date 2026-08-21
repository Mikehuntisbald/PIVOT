import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import aggregate_mmgdino_e5_ownership as aggregate
from tools.mmgdino_e5_ownership import (
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
)


class OwnershipAggregateTests(unittest.TestCase):
    def _write_records(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_paired_bootstrap_and_claim_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluation"
            routes = (
                "native",
                OWNERSHIP_SHARED_128,
                OWNERSHIP_SHARED_WIDE,
                OWNERSHIP_ISOLATED_128,
            )
            for surface in ("refcoco_testA", "refcoco_testB"):
                for route in routes:
                    seeds = (None,) if route == "native" else (17, 42, 73)
                    for seed in seeds:
                        suffix = "native" if seed is None else f"{route}/seed{seed}"
                        rows = [
                            {
                                "sample_id": f"{surface}:{index}",
                                "image_id": str(index // 2),
                                "correct_iou50": True,
                            }
                            for index in range(8)
                        ]
                        self._write_records(
                            evaluation / surface / suffix / "records.jsonl", rows
                        )
            for route in routes:
                seeds = (None,) if route == "native" else (17, 42, 73)
                for seed in seeds:
                    suffix = "native" if seed is None else f"{route}/seed{seed}"
                    negative = 1.0 if route != OWNERSHIP_ISOLATED_128 else 0.1
                    rows = [
                        {
                            "pair_id": f"pair:{index}",
                            "image_id": str(index // 2),
                            "positive_score": 0.9,
                            "negative_score": negative,
                        }
                        for index in range(20)
                    ]
                    self._write_records(
                        evaluation / "strict2031" / suffix / "records.jsonl", rows
                    )
            for route in (
                OWNERSHIP_SHARED_128,
                OWNERSHIP_SHARED_WIDE,
                OWNERSHIP_ISOLATED_128,
            ):
                for seed in (17, 42, 73):
                    path = (
                        root
                        / f"outputs/mmgdino_e5_ownership_transfer_20260821/formal/{route}/seed{seed}/training_receipt.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    probe = {
                        "cosine_mean": (
                            None if route == OWNERSHIP_ISOLATED_128 else -0.1
                        ),
                        "structurally_isolated": route == OWNERSHIP_ISOLATED_128,
                    }
                    path.write_text(
                        json.dumps({"gradient_probes": {"150": probe}}),
                        encoding="utf-8",
                    )
            output = root / "aggregate.json"
            with patch.object(aggregate, "ROOT", root), patch.object(
                aggregate, "BOOTSTRAP_REPLICATES", 100
            ):
                result = aggregate.aggregate(
                    evaluation_root=evaluation, output=output
                )
            self.assertTrue(
                result["claim_gate"]["full_conflict_and_endpoint_claim_supported"]
            )
            self.assertEqual(
                result["point_metrics"][OWNERSHIP_ISOLATED_128][
                    "strict2031_fpr95"
                ],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
