import copy
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path

from datasets.patch_episode import _validate_data_driven_ref_dataset_binding


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_CONFIG = (
    REPO_ROOT
    / "config/datasets_stageb_data_driven_role_routed_clean_train_20260727.json"
)
OLD_FULL_CONFIG = (
    REPO_ROOT
    / "config/datasets_stageb_data_driven_dd1_official_assignment_three_ref.json"
)
NEW_HEAD_D1_CONFIG = (
    REPO_ROOT / "config/datasets_stageb_data_driven_dd1_new_head_train_20260723.json"
)
ROLE_ROUTED_V2 = (
    "role_routed_official_assignment_all_exclusive_nonowned_v2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(supervision: str):
    return types.SimpleNamespace(
        stage_b_data_driven_score=True,
        stage_b_data_driven_train_mode="rank_patch_only",
        stage_b_data_driven_category_complete=True,
        stage_b_data_driven_rank_supervision=supervision,
    )


class RoleRoutedDatasetContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = json.loads(CLEAN_CONFIG.read_text(encoding="ascii"))
        cls.old_full = json.loads(OLD_FULL_CONFIG.read_text(encoding="ascii"))
        cls.new_head = json.loads(NEW_HEAD_D1_CONFIG.read_text(encoding="ascii"))

    def test_clean_role_routed_config_replays_all_three_sealed_lineages(self):
        self.assertEqual(self.clean["val"], [])
        self.assertEqual(len(self.clean["train"]), 3)
        self.assertTrue(
            all(row.get("lazy_jsonl") is True for row in self.clean["train"])
        )
        results = [
            _validate_data_driven_ref_dataset_binding(
                _args(ROLE_ROUTED_V2),
                row,
                image_set="train",
            )
            for row in self.clean["train"]
        ]
        self.assertEqual(results, ["dd1_official_assignment_pair"] * 3)

    def test_lazy_loading_is_exclusive_and_required_for_clean_role_routing(self):
        missing = copy.deepcopy(self.clean["train"][0])
        missing.pop("lazy_jsonl")
        with self.assertRaisesRegex(ValueError, "requires lazy_jsonl=true"):
            _validate_data_driven_ref_dataset_binding(
                _args(ROLE_ROUTED_V2),
                missing,
                image_set="train",
            )

        non_clean = copy.deepcopy(self.old_full["train"][0])
        non_clean["lazy_jsonl"] = True
        with self.assertRaisesRegex(ValueError, "sealed role-routed clean"):
            _validate_data_driven_ref_dataset_binding(
                _args(ROLE_ROUTED_V2),
                non_clean,
                image_set="train",
            )

    def test_role_routed_mode_rejects_old_full_and_plain_clean_d1_receipts(self):
        for label, row in (
            ("old full assignment", self.old_full["train"][0]),
            ("plain clean D1", self.new_head["train"][0]),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                _validate_data_driven_ref_dataset_binding(
                    _args(ROLE_ROUTED_V2),
                    row,
                    image_set="train",
                )

    def test_legacy_assignment_mode_rejects_clean_role_routed_receipt(self):
        with self.assertRaisesRegex(ValueError, "sealed role-routed clean"):
            _validate_data_driven_ref_dataset_binding(
                _args("official_same_image_same_category_assignment_v1"),
                self.clean["train"][0],
                image_set="train",
            )

    def test_clean_contract_requires_train_partition_and_clean_support(self):
        row = self.clean["train"][0]
        bad_partition = dict(row)
        bad_partition["stage_b_data_driven_partition"] = "dev_full"
        with self.assertRaisesRegex(ValueError, "partition='train'"):
            _validate_data_driven_ref_dataset_binding(
                _args(ROLE_ROUTED_V2),
                bad_partition,
                image_set="train",
            )

        no_support = dict(row)
        no_support.pop("stage_b_data_driven_support_receipt")
        with self.assertRaisesRegex(ValueError, "support receipt/hash binding"):
            _validate_data_driven_ref_dataset_binding(
                _args(ROLE_ROUTED_V2),
                no_support,
                image_set="train",
            )

    def test_resealed_receipt_cannot_relax_model_score_free_selection(self):
        row = copy.deepcopy(self.clean["train"][0])
        source_receipt = Path(row["stage_b_data_driven_receipt"])
        receipt = json.loads(source_receipt.read_text(encoding="ascii"))
        receipt["selection_contract"]["model_score_free"] = False
        receipt.pop("canonical_payload_sha256")
        canonical = json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        receipt["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(
                json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n",
                encoding="ascii",
            )
            row["stage_b_data_driven_receipt"] = str(path)
            row["stage_b_data_driven_receipt_sha256"] = _sha256(path)
            with self.assertRaisesRegex(ValueError, "receipt contract drifted"):
                _validate_data_driven_ref_dataset_binding(
                    _args(ROLE_ROUTED_V2),
                    row,
                    image_set="train",
                )


if __name__ == "__main__":
    unittest.main()
