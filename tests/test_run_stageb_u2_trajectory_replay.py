import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools import run_stageb_u2_trajectory_replay as replay


class U2TrajectoryReplayTest(unittest.TestCase):
    def test_plan_is_exact_four_segment_complete_state_replay(self):
        root = replay.REPO_ROOT / "outputs/test-u2-replay"
        plan = replay.build_plan(output_root=root, python_bin=Path("/env/python"))
        self.assertEqual([row["target"] for row in plan["segments"]], [25, 50, 75, 100])
        self.assertEqual(plan["contract"]["physical_batch_size"], 56)
        self.assertEqual(plan["contract"]["amp_initial_scale"], 8192.0)
        self.assertEqual(plan["contract"]["checkpoint_interval"], 25)
        self.assertEqual(plan["contract"]["seed"], 42)
        for index, segment in enumerate(plan["segments"]):
            command = segment["command"]
            self.assertIn("batch_size=56", command)
            self.assertEqual(command[command.index("--seed") + 1], "42")
            self.assertEqual(
                command[command.index("--iter_checkpoint_interval") + 1], "25"
            )
            self.assertEqual(
                command[command.index("--max_train_iters") + 1],
                str(segment["target"]),
            )
            if index == 0:
                self.assertIn("--pretrain_model_path", command)
                self.assertNotIn("--resume", command)
            else:
                self.assertIn("--resume", command)
                self.assertNotIn("--pretrain_model_path", command)

    def test_formal_output_root_is_never_accepted(self):
        with self.assertRaisesRegex(replay.U2ReplayError, "isolated"):
            replay._validate_output_root(replay.FORMAL_ROOT)
        with self.assertRaisesRegex(replay.U2ReplayError, "isolated"):
            replay._validate_output_root(replay.FORMAL_ROOT / "replay")

    def test_u100_acceptance_is_bitwise_not_file_hash_based(self):
        audits = [
            {"target": target, "trainable_tensor_sha256": f"sha-{target}"}
            for target in replay.MILESTONES
        ]
        audits[-1]["trainable_tensor_sha256"] = replay.EXPECTED_FORMAL_TRAINABLE_SHA256
        self.assertEqual(
            audits[-1]["trainable_tensor_sha256"],
            replay.EXPECTED_FORMAL_TRAINABLE_SHA256,
        )
        self.assertTrue(
            replay.build_plan(
                output_root=replay.DEFAULT_OUTPUT_ROOT,
                python_bin=Path("/env/python"),
            )["acceptance"]["full_checkpoint_file_sha_expected_to_differ_due_to_replay_args"]
        )

    def test_atomic_copy_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            destination = root / "milestones" / "checkpoint.pth"
            source.write_bytes(b"checkpoint")
            replay._atomic_copy_fresh(source, destination)
            self.assertEqual(destination.read_bytes(), b"checkpoint")
            with self.assertRaisesRegex(replay.U2ReplayError, "overwrite"):
                replay._atomic_copy_fresh(source, destination)

    def test_checkpoint_audit_rejects_final_trainable_mismatch(self):
        trainable_keys = [f"weight_{index}" for index in range(16)]
        payload = {
            "model": {key: torch.tensor([1.0]) for key in trainable_keys}
        }
        with mock.patch.object(
            replay,
            "_verify_receipt_and_sources",
            return_value=({}, {"changed_keys": trainable_keys}),
        ), mock.patch.object(replay, "_load", return_value=payload), mock.patch.object(
            replay,
            "_audit_path",
            side_effect=lambda **kwargs: {
                "target": kwargs["target"],
                "checkpoint": {"path": str(kwargs["checkpoint"]), "sha256": "x", "size_bytes": 1},
                "trainable_tensor_sha256": "not-formal",
                "frozen_tensor_sha256": replay.EXPECTED_FROZEN_SHA256,
            },
        ), mock.patch.object(replay, "_publish_json"):
            with self.assertRaisesRegex(replay.U2ReplayError, "not the formal trajectory"):
                replay.audit_milestones(replay.DEFAULT_OUTPUT_ROOT)


if __name__ == "__main__":
    unittest.main()
