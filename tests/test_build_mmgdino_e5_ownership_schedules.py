import json
import tempfile
import unittest
from pathlib import Path

from tools.build_mmgdino_e5_ownership_schedules import build_schedules
from tools.responsibility_isolation_cache import file_sha256
from tools.train_mmgdino_e5_ownership import load_schedule


class OwnershipScheduleBuilderTests(unittest.TestCase):
    def test_three_seed_schedules_are_closed_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank_path = root / "rank.jsonl"
            d3_path = root / "d3.jsonl"
            with rank_path.open("w", encoding="utf-8") as handle:
                for index in range(3300):
                    row = {
                        "filename": f"COCO_train2014_{index:012d}.jpg",
                        "source": "refcoco_unc_train",
                        "image_id": index,
                        "ann_id": index,
                        "ref_id": index,
                        "sent_id": index,
                        "instances": [
                            {
                                "positive_phrase": f"object {index}",
                                "text_is_negative": False,
                            }
                        ],
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            with d3_path.open("w", encoding="utf-8") as handle:
                for index in range(450):
                    row = {
                        "sample_id": f"sample:{index}",
                        "sent": f"positive {index}",
                        "try_tn": f"negative {index}",
                        "proposal_covered_verified": True,
                        "visual_verified_negative": True,
                        "traceable_counterfactual_edit": True,
                        "table_b_id": "D3",
                        "tn_scope": "proposal_covered_verified",
                        "split": "train",
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            output = root / "formal"
            receipt = build_schedules(
                rank_jsonl=rank_path,
                rank_jsonl_sha256=file_sha256(rank_path),
                d3_jsonl=d3_path,
                d3_jsonl_sha256=file_sha256(d3_path),
                output_dir=output,
            )
            self.assertEqual(set(receipt["outputs"]), {"17", "42", "73"})
            semantic = []
            for seed in (17, 42, 73):
                schedule, _ = load_schedule(output / f"schedule_seed{seed}.json")
                self.assertEqual(schedule["seed"], seed)
                self.assertEqual(len(schedule["updates"]), 150)
                self.assertEqual(
                    sum(update["task"] == "rank" for update in schedule["updates"]),
                    100,
                )
                semantic.append(
                    json.dumps(schedule["updates"], sort_keys=True)
                )
            self.assertEqual(len(set(semantic)), 3)


if __name__ == "__main__":
    unittest.main()
