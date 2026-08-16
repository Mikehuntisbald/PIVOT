import json
import tempfile
import unittest
from pathlib import Path

from tools.select_stageb_u2v2_milestone import SPLITS, select


class SelectStageBU2V2MilestoneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_json(self, name, payload):
        path = self.root / name
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    def _records(self, name, *, both_correct, mask_hash="a" * 64):
        path = self.root / name
        rows = []
        for index, correct in enumerate((True, both_correct)):
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "correct50": correct,
                    "stage_b_u2v2_eligible_mask_sha256": mask_hash,
                }
            )
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_selects_earliest_equal_candidate_and_rejects_mask_drift(self):
        baseline = {"refcoco": []}
        c0 = {"refcoco": []}
        milestones = {"refcoco": []}
        checkpoints = {}
        for update in (25, 50, 100):
            checkpoint = self.root / f"checkpoint_iter_{update:06d}.pth"
            checkpoint.write_bytes(str(update).encode("ascii"))
            checkpoints[update] = checkpoint
        for split in SPLITS:
            baseline["refcoco"].append(
                {"dataset": split, "acc50": 0.5, "num_expressions": 2}
            )
            c0_records = self._records(f"c0-{split}.jsonl", both_correct=False)
            c0["refcoco"].append(
                {
                    "checkpoint": str(self.root / "c0.pth"),
                    "dataset": split,
                    "acc50": 0.5,
                    "num_expressions": 2,
                    "records_jsonl": str(c0_records),
                }
            )
            for update in (25, 50, 100):
                records = self._records(
                    f"u{update}-{split}.jsonl",
                    both_correct=True,
                    mask_hash=("b" * 64 if update == 100 else "a" * 64),
                )
                milestones["refcoco"].append(
                    {
                        "checkpoint": str(checkpoints[update]),
                        "dataset": split,
                        "acc50": 1.0,
                        "num_expressions": 2,
                        "records_jsonl": str(records),
                    }
                )
        (self.root / "c0.pth").write_bytes(b"c0")
        result = select(
            milestone_summary=self._write_json("milestones.json", milestones),
            c0_summary=self._write_json("c0.json", c0),
            c0_gate_receipt=self._write_json(
                "gate.json",
                {
                    "selected": {
                        "gap": 5.0,
                        "aggregate_acc50": 0.5,
                        "raw_r100_aggregate_acc50": 0.5,
                    }
                },
            ),
            baseline_summary=self._write_json("baseline.json", baseline),
        )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected"]["update"], 25)
        by_update = {row["update"]: row for row in result["candidates"]}
        self.assertTrue(by_update[25]["admitted"])
        self.assertTrue(by_update[50]["admitted"])
        self.assertFalse(by_update[100]["admitted"])
        self.assertFalse(by_update[100]["patch_eligibility_bitwise_equal_to_c0"])


if __name__ == "__main__":
    unittest.main()
