import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.build_stageb_u2v4_legacy_admission_replay import (
    SURFACE_KEYS,
    U2V4ReplayError,
    build_replay_payload,
    validate_replay_payload,
)
from tools.stageb_gdino_adapter_probe_audit import file_record


def _state(surface_value: float, frozen_value: float):
    state = {
        key: torch.full((1,), surface_value + index)
        for index, key in enumerate(SURFACE_KEYS)
    }
    for index in range(1165 - len(SURFACE_KEYS)):
        state[f"frozen.tensor.{index:04d}"] = torch.full(
            (1,), frozen_value + index
        )
    return state


class U2V4LegacyAdmissionReplayTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.c0_path = root / "c0.pth"
        self.legacy_path = root / "legacy.pth"
        self.c0_state = _state(10.0, 100.0)
        self.legacy_state = _state(20.0, 200.0)
        torch.save(
            {"model": self.c0_state, "u2v2_initializer": {}}, self.c0_path
        )
        torch.save(
            {
                "model": self.legacy_state,
                "args": {
                    "stage_b_u2_category_complete_supervision": True,
                    "stage_b_u0_patch_rank": True,
                    "batch_size": 56,
                    "seed": 42,
                    "max_train_iters": 100,
                },
                "optimizer_updates": 100,
                "checkpoint_reason": "max_train_iters",
            },
            self.legacy_path,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    @mock.patch(
        "tools.build_stageb_u2v4_legacy_admission_replay.validate_initializer_payload",
        return_value={"schema": "pivot.stageb.u2v2_initializer/v1"},
    )
    def test_transplants_only_nine_surface_tensors(self, _validate):
        payload = build_replay_payload(
            c0_checkpoint=self.c0_path,
            c0_sha256=file_record(self.c0_path)["sha256"],
            legacy_u2_checkpoint=self.legacy_path,
            legacy_u2_sha256=file_record(self.legacy_path)["sha256"],
        )
        result = payload["model"]
        for key in result:
            expected = self.legacy_state[key] if key in SURFACE_KEYS else self.c0_state[key]
            self.assertTrue(torch.equal(result[key], expected), key)
        contract = validate_replay_payload(payload, verify_sources=False)
        self.assertEqual(contract["surface_tensor_count"], 9)
        self.assertEqual(contract["frozen_tensor_count"], 1156)

    @mock.patch(
        "tools.build_stageb_u2v4_legacy_admission_replay.validate_initializer_payload",
        return_value={"schema": "pivot.stageb.u2v2_initializer/v1"},
    )
    def test_source_hash_mismatch_fails_closed(self, _validate):
        with self.assertRaisesRegex(U2V4ReplayError, "SHA256 mismatch"):
            build_replay_payload(
                c0_checkpoint=self.c0_path,
                c0_sha256="0" * 64,
                legacy_u2_checkpoint=self.legacy_path,
                legacy_u2_sha256=file_record(self.legacy_path)["sha256"],
            )

    @mock.patch(
        "tools.build_stageb_u2v4_legacy_admission_replay.validate_initializer_payload",
        return_value={"schema": "pivot.stageb.u2v2_initializer/v1"},
    )
    def test_tensor_mutation_fails_contract_hash(self, _validate):
        payload = build_replay_payload(
            c0_checkpoint=self.c0_path,
            c0_sha256=file_record(self.c0_path)["sha256"],
            legacy_u2_checkpoint=self.legacy_path,
            legacy_u2_sha256=file_record(self.legacy_path)["sha256"],
        )
        payload["model"][SURFACE_KEYS[0]] += 1
        with self.assertRaisesRegex(U2V4ReplayError, "hash drifted"):
            validate_replay_payload(payload, verify_sources=False)


if __name__ == "__main__":
    unittest.main()
