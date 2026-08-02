import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools import extract_stageb_fixed_gdino_top1_vlm_manifest as extraction
from tools.smoke_stageb_fixed_gdino_top1_runtime import (
    DEPLOY_ROW_INDICES,
    DEPLOY_WORST_BATCH_INDEX,
    MIN_TOTAL_HEADROOM_BYTES,
    RUNTIME_CONTRACT,
    TRAIN_ROW_INDICES,
    RuntimeSmokeError,
    _AmpObservedModel,
    _run_deploy_forwards,
    _run_train_forwards,
    _runtime_code_provenance,
    _validate_batch_tensor,
    _validate_forward_observations,
    _validate_memory_capacity,
    _write_report,
)


class _Nested:
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor) -> None:
        self.tensors = tensors
        self.mask = mask


class _CpuModel:
    def __init__(self) -> None:
        self.captions = []

    def __call__(self, samples, *, captions):
        captions = list(captions)
        self.captions.append(captions)
        batch = len(captions)
        logits = torch.full(
            (batch, extraction.EXPECTED_QUERIES, 3),
            -2.0,
            dtype=torch.float32,
            device=samples.tensors.device,
        )
        for index in range(batch):
            logits[index, index % extraction.EXPECTED_QUERIES, :2] = 2.0
        boxes = torch.full(
            (batch, extraction.EXPECTED_QUERIES, 4),
            0.5,
            dtype=torch.float32,
            device=samples.tensors.device,
        )
        boxes[..., 2:] = 0.2
        phrase_mask = torch.zeros(
            (batch, 1, 3), dtype=torch.bool, device=samples.tensors.device
        )
        phrase_mask[..., :2] = True
        return {
            "pred_logits_text": logits,
            "pred_boxes": boxes,
            "phrase_to_token_mask": phrase_mask,
        }


def _samples(batch: int, height: int = 8, width: int = 9) -> _Nested:
    return _Nested(
        torch.zeros(batch, 3, height, width, dtype=torch.float32),
        torch.zeros(batch, height, width, dtype=torch.bool),
    )


class StageBFixedGdinoTop1RuntimeSmokeTest(unittest.TestCase):
    def test_frozen_batch_contract_targets_exact_rows(self):
        self.assertEqual(TRAIN_ROW_INDICES, (0, 1, 2, 3))
        self.assertEqual(DEPLOY_WORST_BATCH_INDEX, 48)
        self.assertEqual(DEPLOY_ROW_INDICES, tuple(range(768, 784)))
        self.assertEqual(RUNTIME_CONTRACT["queries"], 900)
        self.assertEqual(
            RUNTIME_CONTRACT["deploy"]["forward_order"],
            ["separate_negative", "separate_positive"],
        )

    def test_cpu_mock_replays_train_and_deploy_forward_order(self):
        model = _CpuModel()
        observed = _AmpObservedModel(model)
        train_positive = [f"train positive {index} ." for index in range(4)]
        train_negative = [f"train negative {index} ." for index in range(4)]
        train = _run_train_forwards(
            observed,
            _samples(4),
            train_positive,
            train_negative,
            require_cuda_amp=False,
        )
        deploy_positive = [f"deploy positive {index} ." for index in range(16)]
        deploy_negative = [f"deploy negative {index} ." for index in range(16)]
        deploy = _run_deploy_forwards(
            observed,
            _samples(16),
            deploy_positive,
            deploy_negative,
            require_cuda_amp=False,
        )

        self.assertEqual(
            model.captions,
            [
                train_positive + train_negative,
                train_negative,
                train_positive,
                deploy_negative,
                deploy_positive,
            ],
        )
        self.assertEqual(
            [row["label"] for row in train["calls"]],
            [
                "paired_positive_then_negative",
                "separate_negative",
                "separate_positive",
            ],
        )
        self.assertEqual(
            [row["label"] for row in deploy["calls"]],
            ["separate_negative", "separate_positive"],
        )
        for stage in (train, deploy):
            for output in stage["outputs"].values():
                self.assertEqual(output["queries"], 900)
                self.assertTrue(output["scores"]["finite"])
                self.assertTrue(output["boxes"]["finite"])
                self.assertEqual(output["scores"]["dtype"], "torch.float32")
                self.assertEqual(len(output["scores"]["float32_c_order_sha256"]), 64)
        self.assertTrue(
            all(
                row["autocast_enabled_at_model_call"] is False
                for row in observed.observations
            )
        )

    def test_production_amp_validation_fails_closed_on_cpu_observation(self):
        captions = ["negative ."]
        observation = {
            "device_type": "cpu",
            "autocast_enabled_at_model_call": False,
            "autocast_dtype": "torch.bfloat16",
            "captions": {
                "count": 1,
                "canonical_sha256": extraction.canonical_sha256(captions),
            },
            "outputs": {
                "token_logits": {"device_type": "cpu"},
                "boxes": {"device_type": "cpu"},
                "phrase_mask": {"device_type": "cpu"},
            },
        }
        with self.assertRaisesRegex(RuntimeSmokeError, "did not execute on CUDA"):
            _validate_forward_observations(
                [observation],
                [("separate_negative", captions)],
                require_cuda_amp=True,
            )

    def test_exact_input_shape_and_finite_contract(self):
        report = _validate_batch_tensor(
            _samples(4, 8, 9),
            label="mock",
            expected_shape=(4, 3, 8, 9),
        )
        self.assertTrue(report["finite"])
        self.assertEqual(report["mask_shape"], [4, 8, 9])
        with self.assertRaisesRegex(RuntimeSmokeError, "tensor shape drifted"):
            _validate_batch_tensor(
                _samples(4, 8, 9),
                label="mock",
                expected_shape=(4, 3, 8, 10),
            )
        nonfinite = _samples(4, 8, 9)
        nonfinite.tensors[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeSmokeError, "non-finite"):
            _validate_batch_tensor(
                nonfinite,
                label="mock",
                expected_shape=(4, 3, 8, 9),
            )

    def test_memory_capacity_accepts_boundary_and_rejects_either_shortfall(self):
        gib = MIN_TOTAL_HEADROOM_BYTES
        report = _validate_memory_capacity(
            total_bytes=4 * gib,
            peak_allocated_bytes=2 * gib,
            peak_reserved_bytes=3 * gib,
            minimum_system_free_bytes=gib,
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["total_headroom_bytes"], gib)

        with self.assertRaisesRegex(RuntimeSmokeError, "less than the required"):
            _validate_memory_capacity(
                total_bytes=4 * gib,
                peak_allocated_bytes=2 * gib,
                peak_reserved_bytes=3 * gib + 1,
                minimum_system_free_bytes=gib,
            )
        with self.assertRaisesRegex(RuntimeSmokeError, "less than the required"):
            _validate_memory_capacity(
                total_bytes=4 * gib,
                peak_allocated_bytes=2 * gib,
                peak_reserved_bytes=3 * gib,
                minimum_system_free_bytes=gib - 1,
            )

    def test_atomic_report_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "runtime_smoke.json"
            payload = {"schema": "test", "pass": True}
            record = _write_report(output, payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertEqual(record["sha256"], extraction.sha256_file(output))
            with self.assertRaisesRegex(RuntimeSmokeError, "refusing to overwrite"):
                _write_report(output, payload)

    def test_runtime_code_provenance_binds_new_script_and_locked_extractor(self):
        provenance = _runtime_code_provenance()
        self.assertEqual(
            [row["role"] for row in provenance["files"]],
            ["runtime_smoke", "locked_extractor_core"],
        )
        for row in provenance["files"]:
            self.assertEqual(row["sha256"], extraction.sha256_file(Path(row["path"])))
        self.assertEqual(
            provenance["sha256"], extraction.canonical_sha256(provenance["files"])
        )


if __name__ == "__main__":
    unittest.main()
