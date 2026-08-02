import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tools import build_stageb_legacy_replay_receipt as replay
from tools.stageb_gdino_adapter_probe_audit import (
    ADAPTER_PREFIX,
    file_record,
    model_hash_record,
    tensor_state_sha256,
)


def _adapter_state(*, rank_trained: bool, confidence_trained: bool):
    return {
        ADAPTER_PREFIX + "rank_norm.weight": torch.tensor([1.0, 2.0]),
        ADAPTER_PREFIX + "rank_output.weight": torch.tensor(
            [[0.25, -0.5]] if rank_trained else [[0.0, 0.0]]
        ),
        ADAPTER_PREFIX + "rank_output.bias": torch.tensor(
            [0.125] if rank_trained else [0.0]
        ),
        ADAPTER_PREFIX + "confidence_norm.weight": torch.tensor([3.0, 4.0]),
        ADAPTER_PREFIX + "confidence_gate.4.weight": torch.tensor(
            [[0.75, -0.25]] if confidence_trained else [[0.0, 0.0]]
        ),
        ADAPTER_PREFIX + "confidence_gate.4.bias": torch.tensor(
            [-0.375] if confidence_trained else [0.0]
        ),
    }


class LegacyReplayFixture:
    def __init__(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.b58 = self.root / "b58.pth"
        self.rank = self.root / "rank-r100.pth"
        self.confidence = self.root / "confidence-p50.pth"
        self.merged = self.root / "merged.pth"
        self.receipt = self.root / "receipt.json"
        self.base = {
            "backbone.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "decoder.bias": torch.tensor([0.5, -0.5], dtype=torch.float32),
        }
        torch.save({"model": self._base_copy(), "epoch": 1}, self.b58)
        self._write_training_source("rank")
        self._write_training_source("confidence")
        self._write_merged()

    def close(self):
        self.context.cleanup()

    def _base_copy(self):
        return {key: value.clone() for key, value in self.base.items()}

    def _write_training_source(self, role: str):
        is_rank = role == "rank"
        path = self.rank if is_rank else self.confidence
        target = 100 if is_rank else 50
        state = self._base_copy()
        state.update(
            _adapter_state(
                rank_trained=is_rank,
                confidence_trained=not is_rank,
            )
        )
        torch.save(
            {
                "model": state,
                "epoch": 0,
                "iteration": target,
                "optimizer_updates": target,
                "epoch_finished": False,
                "checkpoint_reason": "max_train_iters",
                "optimizer": {"state": {}, "param_groups": []},
                "criterion": {},
                "args": SimpleNamespace(
                    config_file=f"cfg_{role}.py",
                    datasets=f"datasets_{role}.json",
                    seed=42,
                    batch_size=32,
                    world_size=1,
                    distributed=False,
                    amp=True,
                    max_train_iters=target,
                    stage_b_gdino_adapter_train_mode=(
                        "rank_only" if is_rank else "confidence_only"
                    ),
                ),
            },
            path,
        )

    def _write_merged(self, *, corrupt_contract: bool = False):
        rank_state = torch.load(self.rank, map_location="cpu", weights_only=False)[
            "model"
        ]
        confidence_state = torch.load(
            self.confidence, map_location="cpu", weights_only=False
        )["model"]
        state = {}
        for key, value in rank_state.items():
            state[key] = (
                confidence_state[key].clone()
                if key.startswith(ADAPTER_PREFIX + "confidence_")
                else value.clone()
            )
        hashes = model_hash_record(state)
        contract = {
            "schema": "stageb-gdino-adapter-merged-eval-contract-v1",
            "eval_only": True,
            "resumable": False,
            "model_state_keys": hashes["model_state_keys"],
            "base_state_keys": hashes["base_state_keys"],
            "adapter_state_keys": hashes["adapter_state_keys"],
            "base_tensor_sha256": hashes["base_model_sha256"],
            "adapter_tensor_sha256": hashes["adapter_sha256"],
            "rank_tensor_sha256": hashes["rank_sha256"],
            "confidence_tensor_sha256": hashes["confidence_sha256"],
            "full_model_tensor_sha256": tensor_state_sha256(state, state.keys()),
        }
        if corrupt_contract:
            contract["rank_tensor_sha256"] = "0" * 64
        torch.save({"model": state, "contract": contract}, self.merged)

    def expected_hashes(self):
        rank_state = torch.load(self.rank, map_location="cpu", weights_only=False)[
            "model"
        ]
        confidence_state = torch.load(
            self.confidence, map_location="cpu", weights_only=False
        )["model"]
        merged_state = torch.load(
            self.merged, map_location="cpu", weights_only=False
        )["model"]
        return {
            "b58": {"file_sha256": file_record(self.b58)["sha256"]},
            "rank_r100": {
                "file_sha256": file_record(self.rank)["sha256"],
                "rank_tensor_sha256": model_hash_record(rank_state)["rank_sha256"],
            },
            "confidence_p50": {
                "file_sha256": file_record(self.confidence)["sha256"],
                "confidence_tensor_sha256": model_hash_record(confidence_state)[
                    "confidence_sha256"
                ],
            },
            "merged": {
                "file_sha256": file_record(self.merged)["sha256"],
                "full_model_tensor_sha256": tensor_state_sha256(
                    merged_state, merged_state.keys()
                ),
            },
        }


class BuildStageBLegacyReplayReceiptTest(unittest.TestCase):
    def setUp(self):
        self.fixture = LegacyReplayFixture()

    def tearDown(self):
        self.fixture.close()

    def _build(self, **overrides):
        kwargs = {
            "output": self.fixture.receipt,
            "b58_checkpoint": self.fixture.b58,
            "rank_r100_checkpoint": self.fixture.rank,
            "confidence_p50_checkpoint": self.fixture.confidence,
            "merged_checkpoint": self.fixture.merged,
            "expected_hashes": self.fixture.expected_hashes(),
        }
        kwargs.update(overrides)
        return replay.build_receipt(**kwargs)

    def test_build_and_verify_bind_files_tensors_metadata_and_args(self):
        receipt = self._build()
        self.assertEqual(receipt["schema"], replay.SCHEMA)
        self.assertEqual(
            receipt["receipt_sha256"],
            replay.canonical_json_sha256(
                {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            ),
        )
        self.assertEqual(
            receipt["checkpoints"]["rank_r100"]["training"]["optimizer_updates"],
            100,
        )
        self.assertEqual(
            receipt["checkpoints"]["confidence_p50"]["training"]["args_summary"]
            ["batch_size"],
            32,
        )
        self.assertTrue(
            receipt["invariants"]["merged_rank_matches_rank_r100_bitwise"]
        )
        verified = replay.verify_receipt(self.fixture.receipt)
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["receipt_sha256"], receipt["receipt_sha256"])

    def test_optional_merged_can_be_omitted(self):
        expected = self.fixture.expected_hashes()
        del expected["merged"]
        receipt = self._build(merged_checkpoint=None, expected_hashes=expected)
        self.assertNotIn("merged", receipt["checkpoints"])
        self.assertFalse(receipt["invariants"]["merged_present"])

    def test_wrong_expected_file_hash_fails_before_publication(self):
        expected = self.fixture.expected_hashes()
        expected["rank_r100"]["file_sha256"] = "0" * 64
        with self.assertRaisesRegex(replay.LegacyReplayReceiptError, "mismatch"):
            self._build(expected_hashes=expected)
        self.assertFalse(self.fixture.receipt.exists())

    def test_wrong_expected_tensor_hash_fails_before_publication(self):
        expected = self.fixture.expected_hashes()
        expected["confidence_p50"]["confidence_tensor_sha256"] = "f" * 64
        with self.assertRaisesRegex(replay.LegacyReplayReceiptError, "mismatch"):
            self._build(expected_hashes=expected)
        self.assertFalse(self.fixture.receipt.exists())

    def test_role_branch_contract_is_strict(self):
        payload = torch.load(
            self.fixture.rank, map_location="cpu", weights_only=False
        )
        payload["model"][ADAPTER_PREFIX + "confidence_gate.4.bias"].fill_(1.0)
        torch.save(payload, self.fixture.rank)
        with self.assertRaisesRegex(
            replay.LegacyReplayReceiptError, "confidence output is not untouched"
        ):
            self._build(expected_hashes={})

    def test_merged_embedded_contract_is_recomputed(self):
        self.fixture._write_merged(corrupt_contract=True)
        with self.assertRaisesRegex(
            replay.LegacyReplayReceiptError, "merged contract hash mismatch"
        ):
            self._build(expected_hashes={})

    def test_merged_must_remain_eval_only(self):
        payload = torch.load(
            self.fixture.merged, map_location="cpu", weights_only=False
        )
        payload["contract"]["resumable"] = True
        torch.save(payload, self.fixture.merged)
        with self.assertRaisesRegex(
            replay.LegacyReplayReceiptError, "eval-only and non-resumable"
        ):
            self._build(expected_hashes={})

    def test_parse_expectations_rejects_duplicates_and_bad_selectors(self):
        with self.assertRaisesRegex(
            replay.LegacyReplayReceiptError, "duplicate"
        ):
            replay.parse_expectations(
                ["b58.file_sha256=" + "a" * 64, "b58.file_sha256=" + "b" * 64]
            )
        with self.assertRaisesRegex(replay.LegacyReplayReceiptError, "unknown"):
            replay.parse_expectations(["rank_r100.unknown=" + "a" * 64])

    def test_verify_rejects_tamper_before_reloading_checkpoints(self):
        self._build()
        value = json.loads(self.fixture.receipt.read_text(encoding="ascii"))
        value["invariants"]["merged_present"] = False
        self.fixture.receipt.write_text(json.dumps(value), encoding="ascii")
        with mock.patch.object(replay, "inspect_checkpoint") as inspect:
            with self.assertRaisesRegex(
                replay.LegacyReplayReceiptError, "self-hash mismatch"
            ):
                replay.verify_receipt(self.fixture.receipt)
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
