import copy
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from datasets.patch_episode import PatchEpisodeJsonlDataset
from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)
from util.stage_b_table_b_contract import (
    TABLE_B_SCOPE_BY_ID,
    TableBContractError,
    build_confidence_ablation_eligible,
    table_b_contract_from_args,
    validate_table_b_dataset_binding,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "config" / "ablations"
AUDIT_PATH = (
    REPO_ROOT
    / "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
)
AUDIT_SHA256 = "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
MATCHED_AUDIT_PATH = (
    REPO_ROOT / "data/ablations/stageb_tn_c2_parent_matched_20260717/audit.json"
)
MATCHED_AUDIT_SHA256 = (
    "ca1c9c581fd78f1fe026397cc127d9b7448c60227b31c5e83148c91e9c61861e"
)
LEAF_NAMES = {
    "D0": "cfg_stageb_v23_table_b_d0_no_tn.py",
    "D1": "cfg_stageb_v23_table_b_d1_unverified_allneg.py",
    "D2": "cfg_stageb_v23_table_b_d2_traceable_edits.py",
    "D3": "cfg_stageb_v23_table_b_d3_proposal_covered.py",
    "D2m": "cfg_stageb_v24_table_b_d2m_matched.py",
    "D3m": "cfg_stageb_v24_table_b_d3m_matched.py",
}
DATASET_NAMES = {
    "D0": "datasets_stageb_table_b_d0_no_tn.json",
    "D1": "datasets_stageb_table_b_d1_unverified_allneg.json",
    "D2": "datasets_stageb_table_b_d2_traceable_edits.json",
    "D3": "datasets_stageb_table_b_d3_proposal_covered.json",
    "D2m": "datasets_stageb_table_b_d2m_matched_traceable.json",
    "D3m": "datasets_stageb_table_b_d3m_matched_proposal_covered.json",
}
AUDIT_PATH_BY_ID = {
    **{key: AUDIT_PATH for key in ("D1", "D2", "D3")},
    **{key: MATCHED_AUDIT_PATH for key in ("D2m", "D3m")},
}
AUDIT_SHA_BY_ID = {
    **{key: AUDIT_SHA256 for key in ("D1", "D2", "D3")},
    **{key: MATCHED_AUDIT_SHA256 for key in ("D2m", "D3m")},
}


def _leaf(table_b_id):
    return runpy.run_path(str(CONFIG_ROOT / LEAF_NAMES[table_b_id]))


def _args(table_b_id):
    config = _leaf(table_b_id)
    return SimpleNamespace(
        **{
            key: value
            for key, value in config.items()
            if not key.startswith("__")
        }
    )


def _datasets(table_b_id):
    return json.loads(
        (REPO_ROOT / "config" / DATASET_NAMES[table_b_id]).read_text(
            encoding="utf-8"
        )
    )


def _target(*, table_b_id=None, scope=None, audit_sha=AUDIT_SHA256, global_flag=False):
    target = {"global_tn_verified": torch.tensor([global_flag], dtype=torch.bool)}
    if table_b_id is not None:
        target.update(
            table_b_id=table_b_id,
            tn_scope=scope,
            table_b_audit_sha256=audit_sha,
        )
    return target


class TableBV23ConfigContractTests(unittest.TestCase):
    def test_leaf_configs_fix_l4_and_exact_scope_switches(self):
        for table_b_id, filename in LEAF_NAMES.items():
            with self.subTest(table_b_id=table_b_id, filename=filename):
                config = _leaf(table_b_id)
                self.assertEqual(config["stage_b_v23_ablation_table"], "B")
                self.assertEqual(config["stage_b_v23_table_id"], table_b_id)
                self.assertEqual(
                    config["stage_b_v23_objective_contract"],
                    "v19_base_plus_gate_acc50_hardneg_v21_l4",
                )
                self.assertEqual(config["stage_b_v16_confidence_output_mode"], "base_plus_gate")
                self.assertTrue(config["stage_b_v19_explicit_confidence_output_contract"])
                self.assertEqual(config["stage_b_v11_negative_iou_threshold"], 0.499)
                self.assertEqual(config["stage_b_v21_token_objective"], "edit_bce")
                self.assertEqual(config["stage_b_v11_predicate_tn_rank_weight"], 1.0)
                expected_audit_sha = (
                    AUDIT_SHA256
                    if table_b_id == "D0"
                    else AUDIT_SHA_BY_ID[table_b_id]
                )
                self.assertEqual(
                    config["stage_b_v19_table_b_audit_sha256"], expected_audit_sha
                )
                if table_b_id == "D0":
                    self.assertFalse(
                        config["stage_b_v19_allow_scope_labeled_tn_ablation"]
                    )
                    self.assertEqual(
                        config["stage_b_v19_table_b_scope_allowlist"], []
                    )
                else:
                    self.assertTrue(
                        config["stage_b_v19_allow_scope_labeled_tn_ablation"]
                    )
                    self.assertEqual(
                        config["stage_b_v19_table_b_scope_allowlist"],
                        [TABLE_B_SCOPE_BY_ID[table_b_id]],
                    )

    def test_dataset_configs_keep_uniform_tn_token_provenance_off(self):
        positives = _datasets("D0")["train"]
        self.assertEqual(len(positives), 3)
        for table_b_id in ("D1", "D2", "D3", "D2m", "D3m"):
            train = _datasets(table_b_id)["train"]
            self.assertEqual(train[:3], positives)
            tn = train[3]
            self.assertIs(tn["require_global_tn_verified"], False)
            self.assertIs(tn["require_single_edit_token_provenance"], False)
            self.assertEqual(tn["paper_table_b_id"], table_b_id)
            self.assertEqual(tn["paper_tn_scope"], TABLE_B_SCOPE_BY_ID[table_b_id])


class TableBFailClosedContractTests(unittest.TestCase):
    def test_default_false_does_not_read_or_require_contract_fields(self):
        args = SimpleNamespace(
            stage_b_v19_allow_scope_labeled_tn_ablation=False,
            stage_b_v19_table_b_audit="/missing/audit.json",
            stage_b_v19_table_b_audit_sha256="not-a-hash",
        )
        self.assertIsNone(table_b_contract_from_args(args))
        self.assertIsNone(
            build_confidence_ablation_eligible(
                args,
                [_target(global_flag=True)],
                torch.tensor([True], dtype=torch.bool),
                device=torch.device("cpu"),
            )
        )

    def test_d0_no_tn_is_valid_and_switch_remains_off(self):
        args = _args("D0")
        self.assertIsNone(table_b_contract_from_args(args))
        self.assertIsNone(
            build_confidence_ablation_eligible(
                args,
                [_target()],
                torch.tensor([False], dtype=torch.bool),
                device=torch.device("cpu"),
            )
        )

    def test_exact_audit_and_dataset_output_binding_pass(self):
        for table_b_id in ("D1", "D2", "D3", "D2m", "D3m"):
            with self.subTest(table_b_id=table_b_id):
                args = _args(table_b_id)
                contract = table_b_contract_from_args(args)
                self.assertEqual(contract.table_b_id, table_b_id)
                self.assertEqual(contract.audit_path, AUDIT_PATH_BY_ID[table_b_id])
                self.assertEqual(
                    contract.audit_sha256, AUDIT_SHA_BY_ID[table_b_id]
                )
                datasets = _datasets(table_b_id)["train"]
                self.assertIsNone(
                    validate_table_b_dataset_binding(args, datasets[0])
                )
                bound = validate_table_b_dataset_binding(args, datasets[3])
                self.assertEqual(bound, contract)

    def test_config_id_scope_audit_and_dataset_mismatches_fail(self):
        args = _args("D1")
        args.stage_b_v19_table_b_id = "D4"
        with self.assertRaisesRegex(TableBContractError, "exact.*table_b_id"):
            table_b_contract_from_args(args)

        args = _args("D1")
        args.stage_b_v19_table_b_scope_allowlist = [
            "unverified_all_negative",
            "traceable_counterfactual_edit",
        ]
        with self.assertRaisesRegex(TableBContractError, "scope allowlist"):
            table_b_contract_from_args(args)

        args = _args("D1")
        args.stage_b_v19_table_b_audit_sha256 = "0" * 64
        with self.assertRaisesRegex(TableBContractError, "SHA-256 mismatch"):
            table_b_contract_from_args(args)

        args = _args("D1")
        d2_tn = _datasets("D2")["train"][3]
        with self.assertRaisesRegex(TableBContractError, "does not match config"):
            validate_table_b_dataset_binding(args, d2_tn)

        d1_tn = copy.deepcopy(_datasets("D1")["train"][3])
        d1_tn["require_global_tn_verified"] = True
        with self.assertRaisesRegex(TableBContractError, "must keep.*false"):
            validate_table_b_dataset_binding(_args("D1"), d1_tn)

    def test_unknown_mixed_scope_hash_and_global_upgrade_fail_per_batch(self):
        args = _args("D1")
        good = _target(table_b_id="D1", scope=TABLE_B_SCOPE_BY_ID["D1"])
        eligible = build_confidence_ablation_eligible(
            args,
            [_target(), good],
            torch.tensor([False, True], dtype=torch.bool),
            device=torch.device("cpu"),
        )
        self.assertEqual(eligible.tolist(), [False, True])

        cases = (
            (
                _target(table_b_id="D2", scope=TABLE_B_SCOPE_BY_ID["D2"]),
                "unknown/mixed Table-B ID",
            ),
            (_target(table_b_id="D1", scope="unknown"), "not in the exact allowlist"),
            (
                _target(
                    table_b_id="D1",
                    scope=TABLE_B_SCOPE_BY_ID["D1"],
                    audit_sha="0" * 64,
                ),
                "audit SHA-256 binding mismatch",
            ),
            (
                _target(
                    table_b_id="D1",
                    scope=TABLE_B_SCOPE_BY_ID["D1"],
                    global_flag=True,
                ),
                "must retain global_tn_verified=false",
            ),
        )
        for target, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TableBContractError, message):
                    build_confidence_ablation_eligible(
                        args,
                        [target],
                        torch.tensor([True], dtype=torch.bool),
                        device=torch.device("cpu"),
                    )

    def test_dataset_propagates_distinct_contract_without_global_upgrade(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(image_path)
            row = {
                "table_b_pair_schema": "stage-b-paper-table-b-scope-preserving-pair-v1",
                "table_b_id": "D1",
                "sample_id": "unit-d1",
                "image_path": str(image_path),
                "image_id": 1,
                "class_id": 1,
                "target_bbox_used": [4, 4, 16, 16],
                "sent": "red car",
                "try_tn": "blue car",
                "class_norm_name": "car",
                "replace_from": "red car",
                "replace_to": "blue car",
                "replace_category": "color",
                "tn_scope": "unverified_all_negative",
                "global_tn_verified": False,
            }
            annotation = root / "pairs.jsonl"
            annotation.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = PatchEpisodeJsonlDataset(
                root=str(root),
                anno=str(annotation),
                source="sam3_tn_pair",
                box_format="xywh",
                neg_episode_prob=0.0,
                support_min_count=1,
                support_num_patches_min=1,
                support_num_patches_max=1,
                build_text_token_masks=True,
                text_encoder_type="bert-base-uncased",
                text_mask_warn_limit=0,
                tn_balance_sampling=False,
                table_b_id="D1",
                table_b_scope="unverified_all_negative",
                table_b_audit_sha256=AUDIT_SHA256,
            )
            _image, target = dataset[0]

        self.assertEqual(target["table_b_id"], "D1")
        self.assertEqual(target["tn_scope"], "unverified_all_negative")
        self.assertEqual(target["table_b_audit_sha256"], AUDIT_SHA256)
        self.assertEqual(target["global_tn_verified"].tolist(), [False])


class TableBCriterionEligibilityTests(unittest.TestCase):
    def _criterion(self):
        return StageBFixedTextCriterion(
            listwise_weight=0.0,
            local_tn_rank_weight=0.0,
            predicate_tn_rank_weight=0.0,
            local_anchor_weight=0.0,
            global_tn_negative_weight=1.0,
            global_tn_tail_weight=0.0,
        )

    def _inputs(self):
        return {
            "candidate_logits": torch.tensor([[0.1, -0.2]], requires_grad=True),
            "candidate_ious": torch.tensor([[0.8, 0.1]]),
            "global_tn_logits": torch.tensor([[0.7, 1.1]], requires_grad=True),
            "global_tn_verified": torch.tensor([False]),
        }

    def test_separate_ablation_mask_enables_confidence_loss_without_global_alias(self):
        inputs = self._inputs()
        global_before = inputs["global_tn_verified"].clone()
        rejected = self._criterion()(**inputs)
        accepted = self._criterion()(
            **inputs, confidence_ablation_eligible=torch.tensor([True])
        )
        self.assertEqual(
            float(rejected["loss_fixed_text_global_tn_negative"].detach()), 0.0
        )
        self.assertGreater(
            float(accepted["loss_fixed_text_global_tn_negative"].detach()), 0.0
        )
        self.assertEqual(
            float(accepted["fixed_text_confidence_ablation_eligible_count"]), 1.0
        )
        self.assertTrue(torch.equal(inputs["global_tn_verified"], global_before))
        self.assertFalse(bool(inputs["global_tn_verified"].item()))

    def test_global_and_ablation_masks_must_be_disjoint(self):
        inputs = self._inputs()
        inputs["global_tn_verified"] = torch.tensor([True])
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            self._criterion()(
                **inputs, confidence_ablation_eligible=torch.tensor([True])
            )

    def test_omitted_and_explicit_none_preserve_legacy_outputs(self):
        inputs = self._inputs()
        first = self._criterion()(**inputs)
        second = self._criterion()(
            **inputs, confidence_ablation_eligible=None
        )
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertNotIn("fixed_text_confidence_ablation_eligible_count", first)


if __name__ == "__main__":
    unittest.main()
