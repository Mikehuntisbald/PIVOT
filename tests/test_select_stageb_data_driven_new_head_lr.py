import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import select_stageb_data_driven_new_head_lr as selector


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path.read_bytes()),
    }


def _seal(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload.pop("canonical_payload_sha256", None)
    payload["canonical_payload_sha256"] = _sha256(
        selector._canonical_bytes(payload)
    )
    return payload


class NewHeadLRSelectorTest(unittest.TestCase):
    def setUp(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name)
        self.output = self.root / "selection" / "selection_receipt.json"
        self.preregistration = self.root / "preregistration.json"
        preregistration = _seal(
            {
                "schema": selector.PREREGISTRATION_SCHEMA,
                "status": "preregistered",
                "candidate_rank_lrs": list(selector.CANDIDATE_RANK_LRS),
                "optimizer_updates_per_candidate": (
                    selector.OPTIMIZER_UPDATES_PER_CANDIDATE
                ),
                "selection_partition": selector.SELECTION_PARTITION,
                "selection_metric": selector.SELECTION_METRIC,
                "secondary_selection_metric": (
                    selector.SECONDARY_SELECTION_METRIC
                ),
                "tie_break_rule": list(selector.TIE_BREAK_RULE),
                "invariants": {
                    "candidate_set_was_fixed_before_observing_metrics": True,
                    "selection_rule_was_fixed_before_observing_metrics": True,
                },
            }
        )
        self.preregistration.write_text(
            json.dumps(preregistration, sort_keys=True) + "\n", encoding="ascii"
        )
        self.metrics = {
            3e-5: (0.50, 2.0),
            1e-4: (0.60, 3.0),
            3e-4: (0.55, 1.0),
        }
        self.summaries = {
            rank_lr: self._write_summary(rank_lr, *self.metrics[rank_lr])
            for rank_lr in selector.CANDIDATE_RANK_LRS
        }

    def tearDown(self):
        self.context.cleanup()

    def _write_summary(self, rank_lr: float, acc50: float, nll: float) -> Path:
        label = f"{rank_lr:.0e}".replace("-0", "-")
        checkpoint_file = self.root / f"checkpoint_{label}.pth"
        config_file = self.root / f"config_{label}.py"
        checkpoint_file.write_bytes(f"checkpoint-{rank_lr}".encode("ascii"))
        config_file.write_text(f"rank_lr = {rank_lr!r}\n", encoding="ascii")
        paired = {
            "stage_b_data_driven_rank_lr": rank_lr,
            "stage_b_data_driven_patch_lr": 3e-4,
            "stage_b_data_driven_execution_scope": selector.EXECUTION_SCOPE,
            "stage_b_data_driven_formal_fresh_start": False,
            "stage_b_data_driven_formal_expected_optimizer_updates": 1000,
            "stage_b_data_driven_new_head_formal_contract": (
                selector.EXECUTION_CONTRACT
            ),
            "seed": 42,
        }
        evaluation_config = {
            "stage_b_data_driven_rank_lr": rank_lr,
            "stage_b_data_driven_patch_lr": 3e-4,
            "stage_b_data_driven_execution_scope": selector.EXECUTION_SCOPE,
            "stage_b_data_driven_formal_fresh_start": False,
            "stage_b_data_driven_formal_expected_optimizer_updates": 1000,
            "stage_b_data_driven_new_head_formal_contract": (
                selector.EXECUTION_CONTRACT
            ),
        }
        summary = _seal(
            {
                "schema": selector.EVALUATION_SUMMARY_SCHEMA,
                "evaluation_scope": selector.EVALUATION_SCOPE,
                "scope_limits": {"common_base": "frozen_b58"},
                "variant": selector.EVALUATION_VARIANT,
                "evaluation_manifest_variant": (
                    selector.EVALUATION_MANIFEST_VARIANT
                ),
                "partition": selector.SELECTION_PARTITION,
                "partition_canonical_payload_sha256": "a" * 64,
                "evaluation_inputs": {
                    "support_patch_pool_content": {
                        "ordered_content_sha256": "b" * 64
                    },
                    "query_image_content": {
                        "ordered_content_sha256": "c" * 64
                    },
                },
                "checkpoint_contract": {
                    "checkpoint": _file_record(checkpoint_file),
                    "config": _file_record(config_file),
                    "optimizer_updates": 1000,
                    "experiment_id": "DD1",
                    "rank_lr": rank_lr,
                    "paired_training_contract": paired,
                    "evaluation_config_training_contract": evaluation_config,
                    "shared_training_provenance": {
                        "schema": "training-provenance/v1",
                        "code_files": [{"sha256": "d" * 64}],
                    },
                    "initializer_provenance": {
                        "common_tensor_sha256": "e" * 64,
                        "base_initializer": {"sha256": "f" * 64},
                    },
                    "formal_execution_status": {
                        "formal": False,
                        "reason": "diagnostic_new_head_execution_scope",
                        "execution_scope": selector.EXECUTION_SCOPE,
                        "formal_fresh_start": False,
                        "declared_optimizer_updates": 1000,
                        "observed_optimizer_updates": 1000,
                        "formal_contract": selector.EXECUTION_CONTRACT,
                    },
                    "training_partition_status": {
                        "formal": True,
                        "reason": "new_head_train_partition_bound",
                    },
                    "formal_new_head_partition_evaluation": False,
                },
                "protocol": {
                    "evaluation_scope": selector.EVALUATION_SCOPE,
                    "evaluation_manifest_variant": (
                        selector.EVALUATION_MANIFEST_VARIANT
                    ),
                    "rank_only": True,
                    "category_gate": False,
                    "query_count": 900,
                    "seed": 20260723,
                },
                "metrics": {
                    selector.SELECTION_METRIC: float(acc50),
                    selector.SECONDARY_SELECTION_METRIC: float(nll),
                },
            }
        )
        path = self.root / f"summary_{label}.json"
        path.write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="ascii"
        )
        return path

    def _rewrite_summary(self, rank_lr: float, mutation) -> None:
        path = self.summaries[rank_lr]
        value = json.loads(path.read_text(encoding="ascii"))
        mutation(value)
        value = _seal(value)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")

    def _build(self):
        return selector.build_selection_receipt(
            self.summaries,
            preregistration=self.preregistration,
            output=self.output,
        )

    def test_selects_highest_macro_acc50_and_emits_bound_receipt(self):
        receipt = self._build()
        self.assertEqual(receipt["selected_rank_lr"], 1e-4)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(
            receipt["candidate_rank_lrs"], list(selector.CANDIDATE_RANK_LRS)
        )
        self.assertEqual(len(receipt["candidates"]), 3)
        self.assertTrue(all(receipt["invariants"].values()))
        on_disk = json.loads(self.output.read_text(encoding="ascii"))
        self.assertEqual(on_disk, receipt)
        canonical = dict(receipt)
        digest = canonical.pop("canonical_payload_sha256")
        self.assertEqual(digest, _sha256(selector._canonical_bytes(canonical)))

    def test_rejects_summary_tampering_without_resealed_hash(self):
        path = self.summaries[3e-5]
        value = json.loads(path.read_text(encoding="ascii"))
        value["metrics"][selector.SELECTION_METRIC] = 0.99
        path.write_text(json.dumps(value), encoding="ascii")
        with self.assertRaisesRegex(
            selector.NewHeadLRSelectionError, "canonical hash drifted"
        ):
            self._build()

    def test_rejects_resealed_shared_contract_drift(self):
        self._rewrite_summary(
            3e-4,
            lambda value: value["protocol"].update(seed=99),
        )
        with self.assertRaisesRegex(
            selector.NewHeadLRSelectionError, "shared contract: protocol"
        ):
            self._build()

    def test_tie_breaks_by_nll_then_smaller_lr(self):
        for rank_lr in selector.CANDIDATE_RANK_LRS:
            self._rewrite_summary(
                rank_lr,
                lambda value: value["metrics"].update(
                    {
                        selector.SELECTION_METRIC: 0.7,
                        selector.SECONDARY_SELECTION_METRIC: (
                            1.0 if rank_lr in (3e-5, 1e-4) else 2.0
                        ),
                    }
                ),
            )
        receipt = self._build()
        self.assertEqual(receipt["selected_rank_lr"], 3e-5)

    def test_never_overwrites_existing_output(self):
        self._build()
        original = self.output.read_bytes()
        with self.assertRaisesRegex(
            selector.NewHeadLRSelectionError, "refusing to replace output"
        ):
            self._build()
        self.assertEqual(self.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
