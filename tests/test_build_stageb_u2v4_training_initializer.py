import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.build_stageb_u2v4_training_initializer import (
    U0_PREFIX,
    U2V4TrainingInitializerError,
    build_training_initializer_payload,
    validate_training_initializer_payload,
)
from tools.stageb_gdino_adapter_probe_audit import file_record


U0_KEYS = (
    U0_PREFIX + "_contract_query_count",
    U0_PREFIX + "_contract_score_clip",
    U0_PREFIX + "_contract_version",
    U0_PREFIX + "output.bias",
    U0_PREFIX + "output.weight",
    U0_PREFIX + "trunk.0.bias",
    U0_PREFIX + "trunk.0.weight",
    U0_PREFIX + "trunk.1.bias",
    U0_PREFIX + "trunk.1.weight",
    U0_PREFIX + "trunk.3.bias",
    U0_PREFIX + "trunk.3.weight",
)


def _state(*, legacy: bool):
    state = {f"frozen.{index:04d}": torch.tensor([float(index)]) for index in range(1154)}
    for key in U0_KEYS:
        if key.endswith("weight") and "output" not in key and legacy:
            value = torch.ones(2)
        elif key.startswith(U0_PREFIX + "_contract"):
            value = torch.ones((), dtype=torch.int64)
        else:
            value = torch.zeros(2)
        state[key] = value
    return state


class U2V4TrainingInitializerTest(unittest.TestCase):
    def _build(self, directory: Path):
        c0_path = directory / "c0.pth"
        u0_path = directory / "u0.pth"
        torch.save({"model": _state(legacy=False), "u2v2_initializer": {}}, c0_path)
        torch.save({"model": _state(legacy=True), "u0_initializer": {}}, u0_path)

        def verify_u0(path):
            return {
                "schema": "pivot.stageb.u0_initializer/v2",
                "checkpoint": file_record(path),
            }

        with mock.patch(
            "tools.build_stageb_u2v4_training_initializer.validate_initializer_payload",
            return_value={"schema": "pivot.stageb.u2v2_initializer/v1"},
        ), mock.patch(
            "tools.build_stageb_u2v4_training_initializer.verify_u0",
            side_effect=verify_u0,
        ):
            payload = build_training_initializer_payload(
                c0_checkpoint=c0_path,
                c0_sha256=file_record(c0_path)["sha256"],
                legacy_u0_initializer=u0_path,
                legacy_u0_sha256=file_record(u0_path)["sha256"],
            )
        return payload, c0_path, u0_path

    def test_restores_exact_u0_shell_and_preserves_every_other_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, c0_path, u0_path = self._build(Path(directory))
            c0 = torch.load(c0_path, map_location="cpu", weights_only=False)["model"]
            u0 = torch.load(u0_path, map_location="cpu", weights_only=False)["model"]
            for key, value in payload["model"].items():
                self.assertTrue(torch.equal(value, u0[key] if key in U0_KEYS else c0[key]))
            contract = validate_training_initializer_payload(
                payload, verify_sources=False
            )
            self.assertEqual(contract["frozen_c0_tensor_count"], 1154)
            self.assertTrue(contract["invariants"]["legacy_u0_output_zero"])

    def test_rejects_source_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            c0_path = Path(directory) / "c0.pth"
            u0_path = Path(directory) / "u0.pth"
            torch.save({"model": _state(legacy=False)}, c0_path)
            torch.save({"model": _state(legacy=True)}, u0_path)
            with self.assertRaisesRegex(U2V4TrainingInitializerError, "SHA256"):
                build_training_initializer_payload(
                    c0_checkpoint=c0_path,
                    c0_sha256="0" * 64,
                    legacy_u0_initializer=u0_path,
                    legacy_u0_sha256=file_record(u0_path)["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
