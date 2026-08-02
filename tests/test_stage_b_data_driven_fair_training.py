import hashlib
import json
import os
import random
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset

import main as main_module
from datasets.patch_episode import _validate_data_driven_ref_dataset_binding
from datasets.transforms import crop
from engine import _clip_stage_b_data_driven_optimizer_grad_norms
from main import (
    DeterministicEpochSampler,
    _prepare_stage_b_data_driven_epoch_sampling,
    _stage_b_data_driven_epoch_checkpoint_due,
    _stage_b_data_driven_dataset_asset_paths,
    _stage_b_data_driven_support_pool_content_records,
    _validate_stage_b_data_driven_assignment_training_contract,
    _validate_stage_b_data_driven_dd1h_fresh_training_contract,
    _validate_stage_b_data_driven_eval_update_gate,
    _validate_stage_b_data_driven_formal_evidence_payloads,
    _validate_stage_b_data_driven_new_head_formal_training_contract,
    _validate_stage_b_data_driven_sampling_resume_state,
)
from models.GroundingDINO.stage_b_data_driven_score import (
    StageBDataDrivenCriterion,
    StageBDataDrivenScoreHeads,
    validate_data_driven_trained_checkpoint_payload,
)
from util.slconfig import SLConfig


class _ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_b_data_driven_score_heads = StageBDataDrivenScoreHeads(
            4,
            rank_dim=3,
            confidence_dim=3,
            gate_hidden_dim=4,
        )


class _RandomizedDataset(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, index):
        return torch.tensor(
            [
                float(index),
                random.random(),
                float(np.random.rand()),
                float(torch.rand(()).item()),
            ],
            dtype=torch.float64,
        )


def _trained_payload(model, *, updates=5020, reason="max_train_iters"):
    criterion = StageBDataDrivenCriterion(
        train_mode="rank_patch_only",
        category_complete=True,
    )
    return {
        "model": {
            key: value.detach().clone() for key, value in model.state_dict().items()
        },
        "criterion": criterion.state_dict(),
        "args": {
            "stage_b_data_driven_score": True,
            "stage_b_data_driven_experiment_id": "DD1",
            "stage_b_data_driven_train_mode": "rank_patch_only",
            "stage_b_data_driven_category_complete": True,
            "stage_b_data_driven_confidence_trained": False,
            "stage_b_data_driven_rank_architecture": "absolute_token",
            "stage_b_data_driven_rank_supervision": (
                "all_nonpositive_negative_v1"
            ),
            "stage_b_data_driven_rank_negative_iou_threshold": 0.3,
            "seed": 42,
            "max_train_iters": 5020,
        },
        "optimizer_updates": updates,
        "checkpoint_reason": reason,
    }


class DeterministicEpochSamplerTest(unittest.TestCase):
    def test_sampler_contract_rejects_implicit_integer_coercion(self):
        with self.assertRaisesRegex(ValueError, "sampling seeds"):
            DeterministicEpochSampler(4, seed=1.5)
        with self.assertRaisesRegex(ValueError, "num_samples"):
            DeterministicEpochSampler(4, seed=1, num_samples=2.5)

    def test_uniform_ledger_is_independent_of_global_model_rng(self):
        torch.manual_seed(3)
        _ = torch.rand(1000)
        first = DeterministicEpochSampler(101, seed=42017)
        first.set_epoch(7)
        first_indices = list(first)
        first_state = first.ledger_state()

        torch.manual_seed(9999)
        _ = torch.rand(2000)
        second = DeterministicEpochSampler(101, seed=42017)
        second.set_epoch(7)
        self.assertEqual(first_indices, list(second))
        self.assertEqual(first_state, second.ledger_state())

        second.set_epoch(8)
        self.assertNotEqual(first_indices, list(second))
        self.assertNotEqual(
            first_state["ledger_sha256"],
            second.ledger_state()["ledger_sha256"],
        )

    def test_weighted_mid_epoch_resume_replays_exact_suffix(self):
        weights = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.double)
        original = DeterministicEpochSampler(
            4,
            seed=91,
            weights=weights,
            num_samples=40,
            replacement=True,
        )
        original.set_epoch(2)
        ledger = list(original)

        resumed = DeterministicEpochSampler(
            4,
            seed=91,
            weights=weights,
            num_samples=40,
            replacement=True,
        )
        resumed.set_epoch(2)
        resumed_iterator = iter(resumed)
        for _ in range(13):
            next(resumed_iterator)
        self.assertEqual(ledger[13:], list(resumed_iterator))
        self.assertEqual(original.ledger_state(), resumed.ledger_state())

    def test_loader_and_sampling_epoch_state_are_reconstructed(self):
        first_sampler = DeterministicEpochSampler(31, seed=101)
        first_generator = torch.Generator()
        first_state = _prepare_stage_b_data_driven_epoch_sampling(
            first_sampler,
            first_generator,
            epoch=4,
            sampler_seed=101,
            loader_seed=202,
        )
        first_rng_values = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(()).item()),
        )

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        second_sampler = DeterministicEpochSampler(31, seed=101)
        second_generator = torch.Generator()
        second_state = _prepare_stage_b_data_driven_epoch_sampling(
            second_sampler,
            second_generator,
            epoch=4,
            sampler_seed=101,
            loader_seed=202,
        )
        second_rng_values = (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(()).item()),
        )
        self.assertEqual(first_state, second_state)
        self.assertEqual(first_rng_values, second_rng_values)

        checkpoint = {
            "epoch": 4,
            "stage_b_data_driven_sampling_state": first_state,
        }
        self.assertEqual(
            _validate_stage_b_data_driven_sampling_resume_state(
                second_sampler, checkpoint, loader_seed=202
            ),
            first_state,
        )
        checkpoint["stage_b_data_driven_sampling_state"] = dict(first_state)
        checkpoint["stage_b_data_driven_sampling_state"]["ledger_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "ledger drifted"):
            _validate_stage_b_data_driven_sampling_resume_state(
                second_sampler, checkpoint, loader_seed=202
            )

    def test_multiworker_mid_epoch_replay_matches_samples_and_transforms(self):
        def build_epoch():
            sampler = DeterministicEpochSampler(24, seed=501)
            generator = torch.Generator()
            _prepare_stage_b_data_driven_epoch_sampling(
                sampler,
                generator,
                epoch=3,
                sampler_seed=501,
                loader_seed=601,
            )
            loader = DataLoader(
                _RandomizedDataset(),
                batch_sampler=BatchSampler(sampler, batch_size=4, drop_last=True),
                generator=generator,
                num_workers=2,
                persistent_workers=False,
                prefetch_factor=1,
            )
            return loader

        uninterrupted = [batch.tolist() for batch in build_epoch()]
        resumed_iterator = iter(build_epoch())
        for _ in range(2):
            next(resumed_iterator)
        resumed_suffix = [batch.tolist() for batch in resumed_iterator]
        self.assertEqual(uninterrupted[2:], resumed_suffix)


class OfficialAssignmentTransformTest(unittest.TestCase):
    def test_crop_keeps_assignment_roles_aligned_with_boxes(self):
        image = Image.new("RGB", (10, 10))
        target = {
            "boxes": torch.tensor(
                [
                    [1.0, 1.0, 3.0, 3.0],
                    [5.0, 1.0, 7.0, 3.0],
                    [20.0, 20.0, 22.0, 22.0],
                ]
            ),
            "labels": torch.tensor([4, 4, 4]),
            "area": torch.tensor([4.0, 4.0, 4.0]),
            "primary_instance_mask": torch.tensor([True, False, False]),
            "stage_b_data_driven_assignment_role": torch.tensor([0, 1, -1]),
        }
        _image, cropped = crop(image, target, (0, 0, 10, 10))
        self.assertEqual(tuple(cropped["boxes"].shape), (2, 4))
        self.assertEqual(
            cropped["stage_b_data_driven_assignment_role"].tolist(), [0, 1]
        )


class IndependentGradientClipTest(unittest.TestCase):
    def test_rank_and_patch_are_clipped_without_cross_branch_scaling(self):
        rank = nn.Parameter(torch.zeros(2))
        patch = nn.Parameter(torch.zeros(2))
        rank.grad = torch.tensor([3.0, 4.0])
        patch.grad = torch.tensor([0.0, 12.0])
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [rank],
                    "stage_b_data_driven_branch": "rank",
                },
                {
                    "params": [patch],
                    "stage_b_data_driven_branch": "patch",
                },
            ],
            lr=0.1,
        )
        stats = _clip_stage_b_data_driven_optimizer_grad_norms(
            optimizer, 1.0, train_mode="rank_patch_only"
        )
        self.assertAlmostEqual(stats["grad_norm_data_driven_rank_preclip"], 5.0)
        self.assertAlmostEqual(stats["grad_norm_data_driven_patch_preclip"], 12.0)
        self.assertAlmostEqual(float(rank.grad.norm().item()), 1.0, places=5)
        self.assertAlmostEqual(float(patch.grad.norm().item()), 1.0, places=5)

    def test_confidence_mode_and_optimizer_labels_fail_closed(self):
        confidence = nn.Parameter(torch.zeros(1))
        confidence.grad = torch.tensor([9.0])
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [confidence],
                    "stage_b_data_driven_branch": "confidence",
                }
            ],
            lr=0.1,
        )
        _clip_stage_b_data_driven_optimizer_grad_norms(
            optimizer, 2.0, train_mode="confidence_pair"
        )
        self.assertAlmostEqual(float(confidence.grad.item()), 2.0, places=5)

        unlabeled = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
        with self.assertRaisesRegex(RuntimeError, "unlabeled or inactive"):
            _clip_stage_b_data_driven_optimizer_grad_norms(
                unlabeled, 1.0, train_mode="confidence_pair"
            )


class ExactUpdateEvaluationGateTest(unittest.TestCase):
    def test_main_and_shared_eval_gates_require_exact_terminal_update(self):
        model = _ScoreModel()
        payload = _trained_payload(model)
        args = types.SimpleNamespace(
            eval=True,
            stage_b_data_driven_eval_expected_optimizer_updates=5020,
        )
        self.assertEqual(
            _validate_stage_b_data_driven_eval_update_gate(
                args, payload, checkpoint_label="formal"
            ),
            5020,
        )
        validate_data_driven_trained_checkpoint_payload(
            model,
            payload,
            checkpoint_label="formal",
            expected_experiment_id="DD1",
            expected_confidence_trained=False,
            expected_optimizer_updates=5020,
        )

        wrong_update = _trained_payload(model, updates=5019)
        with self.assertRaisesRegex(ValueError, "expected exactly 5020"):
            validate_data_driven_trained_checkpoint_payload(
                model,
                wrong_update,
                checkpoint_label="short",
                expected_experiment_id="DD1",
                expected_confidence_trained=False,
                expected_optimizer_updates=5020,
            )
        wrong_reason = _trained_payload(model, reason="signal")
        with self.assertRaisesRegex(ValueError, "max_train_iters terminal"):
            validate_data_driven_trained_checkpoint_payload(
                model,
                wrong_reason,
                checkpoint_label="interrupted",
                expected_experiment_id="DD1",
                expected_confidence_trained=False,
                expected_optimizer_updates=5020,
            )

        args.stage_b_data_driven_eval_expected_optimizer_updates = -1
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            _validate_stage_b_data_driven_eval_update_gate(
                args, payload, checkpoint_label="negative"
            )

    def test_trained_checkpoint_binds_dd1h_rank_supervision(self):
        model = _ScoreModel()
        payload = _trained_payload(model)
        payload["args"]["stage_b_data_driven_variant_id"] = "DD1-H"
        payload["args"]["stage_b_data_driven_rank_supervision"] = (
            "primary_vs_same_category_aux_v1"
        )
        payload["criterion"] = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision="primary_vs_same_category_aux_v1",
        ).state_dict()
        validate_data_driven_trained_checkpoint_payload(
            model,
            payload,
            checkpoint_label="dd1h",
            expected_experiment_id="DD1",
            expected_confidence_trained=False,
            expected_variant_id="DD1-H",
            expected_rank_supervision="primary_vs_same_category_aux_v1",
            expected_rank_negative_iou_threshold=0.3,
        )
        with self.assertRaisesRegex(ValueError, "supervision attribution"):
            validate_data_driven_trained_checkpoint_payload(
                model,
                payload,
                checkpoint_label="dd1h-as-legacy",
                expected_experiment_id="DD1",
                expected_confidence_trained=False,
                expected_rank_supervision="all_nonpositive_negative_v1",
            )
        with self.assertRaisesRegex(ValueError, "variant attribution"):
            validate_data_driven_trained_checkpoint_payload(
                model,
                payload,
                checkpoint_label="wrong-variant",
                expected_experiment_id="DD1",
                expected_confidence_trained=False,
                expected_variant_id="DD1-PairTop1",
            )

        payload["args"]["stage_b_data_driven_variant_id"] = "DD1-HC"
        payload["args"]["stage_b_data_driven_rank_supervision"] = (
            "primary_vs_same_category_aux_plus_gap3_coverage_v1"
        )
        payload["criterion"] = StageBDataDrivenCriterion(
            train_mode="rank_patch_only",
            category_complete=True,
            rank_supervision=(
                "primary_vs_same_category_aux_plus_gap3_coverage_v1"
            ),
        ).state_dict()
        validate_data_driven_trained_checkpoint_payload(
            model,
            payload,
            checkpoint_label="dd1hc",
            expected_experiment_id="DD1",
            expected_confidence_trained=False,
            expected_variant_id="DD1-HC",
            expected_rank_supervision=(
                "primary_vs_same_category_aux_plus_gap3_coverage_v1"
            ),
            expected_rank_negative_iou_threshold=0.3,
        )


class DataDrivenDatasetBindingScopeTest(unittest.TestCase):
    source_manifests = (
        "refcoco_stageb_phrase_v1.jsonl",
        "refcocoplus_stageb_phrase_v1.jsonl",
        "refcocog_stageb_phrase_v1.jsonl",
    )

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _seal_receipt(path, receipt, *, canonical=True):
        receipt = json.loads(json.dumps(receipt))
        if canonical:
            receipt.pop("canonical_payload_sha256", None)
            payload = json.dumps(
                receipt,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            receipt["canonical_payload_sha256"] = hashlib.sha256(
                payload
            ).hexdigest()
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return receipt

    def _partition_fixture(self, root):
        empty_sha = hashlib.sha256(b"").hexdigest()
        variants = ("d0_ordinary_primary", "d1_category_complete")
        partitions = ("train", "dev_full", "dev_screen", "quarantine")
        outputs = {variant: {} for variant in variants}
        source_records = {}
        for name in self.source_manifests:
            raw = (json.dumps({"source_manifest": name}) + "\n").encode("ascii")
            digest = hashlib.sha256(raw).hexdigest()
            identity_sha = hashlib.sha256(name.encode("ascii")).hexdigest()
            source_records[name] = {
                "rows": 1,
                "d0_ordinary_primary": {
                    "path": str(root / "upstream" / "d0_ordinary_primary" / name),
                    "size_bytes": len(raw),
                    "sha256": digest,
                },
                "d1_category_complete": {
                    "path": str(root / "upstream" / "d1_category_complete" / name),
                    "size_bytes": len(raw),
                    "sha256": digest,
                },
            }
            for variant in variants:
                for partition in partitions:
                    outputs[variant].setdefault(partition, {})
                    path = root / variant / partition / name
                    rows = 1 if partition == "train" else 0
                    if partition == "train":
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(raw)
                    outputs[variant][partition][name] = {
                        "path": str(path),
                        "rows": rows,
                        "unique_identities": rows,
                        "unique_image_keys": rows,
                        "ordered_identity_stream_sha256": (
                            identity_sha if rows else empty_sha
                        ),
                        "size_bytes": len(raw) if rows else 0,
                        "sha256": digest if rows else empty_sha,
                    }
        partition_summary = {}
        for partition in partitions:
            rows = 3 if partition == "train" else 0
            partition_summary[partition] = {
                "unique_image_keys": rows,
                "rows": rows,
                "rows_by_manifest": {
                    name: 1 if partition == "train" else 0
                    for name in self.source_manifests
                },
                "ordered_image_key_stream_sha256": empty_sha,
            }
        receipt = {
            "schema": "pivot.stageb.data_driven.new_head_partition_receipt/v1",
            "source_manifest_order": list(self.source_manifests),
            "source_manifests": source_records,
            "output_layout": "<variant>/<partition>/<source_manifest>",
            "output_stream_encoding": (
                "raw_input_record_including_original_line_ending_v1"
            ),
            "outputs": outputs,
            "partition_summary": partition_summary,
            "invariants": {"fixture_contract_holds": True},
        }
        receipt_path = root / "receipt.json"
        receipt = self._seal_receipt(receipt_path, receipt)
        return receipt_path, receipt

    def _support_fixture(self, root, partition_receipt_path, partition_receipt):
        support_root = root / "support"
        support_root.mkdir(parents=True, exist_ok=True)
        support_image = support_root / "support.jpg"
        support_image.write_bytes(b"support-image")
        runtime_tsv = support_root / "filtered_support.tsv"
        runtime_tsv.write_text(
            f"path\tclass_id\n{support_image}\t1\n",
            encoding="ascii",
        )
        audit_tsv = support_root / "raw_filtered_clean.tsv"
        audit_tsv.write_text(
            f"path\tclass_id\n{support_image}\t1\n",
            encoding="ascii",
        )

        def file_record(path, *, rows=None):
            record = {
                "path": str(path),
                "sha256": self._sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if rows is not None:
                record["rows"] = rows
            return record

        partition_record = file_record(partition_receipt_path)
        receipt = {
            "schema": "pivot.stageb.data_driven.support_partition_receipt/v1",
            "inputs": {"partition_receipt": partition_record},
            "partition": {
                "schema": (
                    "pivot.stageb.data_driven.new_head_partition_receipt/v1"
                ),
                "receipt": partition_record,
                "canonical_payload_sha256": partition_receipt[
                    "canonical_payload_sha256"
                ],
            },
            "filter_contract": {
                "D0_and_D1_share_identical_runtime_bank": True,
                "bank_consumers": ["D0", "D1"],
                "required_dataset_settings": {
                    "patch_bank_cache": False,
                    "patch_bank_cache_write": False,
                    "support_patch_use_embedding": False,
                    "support_patch_max_per_class": 200,
                },
            },
            "runtime_bank": {
                "candidate_rows": 1,
                "class_count": 1,
                "class_counts": {"1": 1},
            },
            "training_class_coverage": {
                "required_class_count": 1,
                "required_class_ids": [1],
                "covered_class_count": 1,
                "covered_class_ids": [1],
                "missing_class_ids": [],
                "support_counts": {"1": 1},
            },
            "outputs": {
                "runtime_support_tsv": file_record(runtime_tsv, rows=1),
                "audit_raw_tsv": file_record(audit_tsv, rows=1),
            },
            "invariants": {"fixture_support_contract_holds": True},
        }
        receipt_path = support_root / "receipt.json"
        receipt = self._seal_receipt(receipt_path, receipt)
        return receipt_path, receipt

    @staticmethod
    def _args(*, category_complete):
        return types.SimpleNamespace(
            stage_b_data_driven_score=True,
            stage_b_data_driven_train_mode="rank_patch_only",
            stage_b_data_driven_category_complete=category_complete,
            stage_b_data_driven_rank_supervision="all_nonpositive_negative_v1",
        )

    def _datasetinfo(self, receipt_path, receipt, *, category_complete=False):
        receipt_variant = (
            "d1_category_complete"
            if category_complete
            else "d0_ordinary_primary"
        )
        dataset_variant = (
            "dd1_category_complete"
            if category_complete
            else "dd0_ordinary_primary"
        )
        anno = Path(
            receipt["outputs"][receipt_variant]["train"][
                self.source_manifests[0]
            ]["path"]
        )
        support_receipt_path, support_receipt = self._support_fixture(
            receipt_path.parent,
            receipt_path,
            receipt,
        )
        return {
            "anno": str(anno),
            "stage_b_data_driven_variant": dataset_variant,
            "stage_b_data_driven_partition": "train",
            "stage_b_data_driven_manifest_sha256": self._sha256(anno),
            "stage_b_data_driven_receipt": str(receipt_path),
            "stage_b_data_driven_receipt_sha256": self._sha256(receipt_path),
            "stage_b_data_driven_support_receipt": str(support_receipt_path),
            "stage_b_data_driven_support_receipt_sha256": self._sha256(
                support_receipt_path
            ),
            "support_patch_tsv": support_receipt["outputs"][
                "runtime_support_tsv"
            ]["path"],
            "patch_bank_cache": False,
            "patch_bank_cache_write": False,
            "support_patch_use_embedding": False,
            "support_patch_max_per_class": 200,
        }

    def test_training_manifest_binding_does_not_apply_to_eval_rows(self):
        for rank_supervision in (
            "primary_vs_same_category_aux_v1",
            "primary_vs_same_category_aux_plus_gap3_coverage_v1",
        ):
            args = types.SimpleNamespace(
                stage_b_data_driven_score=True,
                stage_b_data_driven_train_mode="rank_patch_only",
                stage_b_data_driven_category_complete=True,
                stage_b_data_driven_rank_supervision=rank_supervision,
            )
            self.assertIsNone(
                _validate_data_driven_ref_dataset_binding(
                    args, {}, image_set="val"
                )
            )
            with self.assertRaisesRegex(ValueError, "dataset variant drifted"):
                _validate_data_driven_ref_dataset_binding(
                    args, {}, image_set="train"
                )

    def test_new_head_partition_accepts_only_explicit_train_d0_or_d1_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt_path, receipt = self._partition_fixture(root)
            for category_complete in (False, True):
                with self.subTest(category_complete=category_complete):
                    datasetinfo = self._datasetinfo(
                        receipt_path,
                        receipt,
                        category_complete=category_complete,
                    )
                    expected = datasetinfo["stage_b_data_driven_variant"]
                    self.assertEqual(
                        _validate_data_driven_ref_dataset_binding(
                            self._args(category_complete=category_complete),
                            datasetinfo,
                            image_set="train",
                        ),
                        expected,
                    )
                    for invalid_partition in (None, "dev_full", "quarantine"):
                        invalid = dict(datasetinfo)
                        if invalid_partition is None:
                            del invalid["stage_b_data_driven_partition"]
                        else:
                            invalid["stage_b_data_driven_partition"] = (
                                invalid_partition
                            )
                        with self.assertRaisesRegex(
                            ValueError, "requires explicit.*partition='train'"
                        ):
                            _validate_data_driven_ref_dataset_binding(
                                self._args(
                                    category_complete=category_complete
                                ),
                                invalid,
                                image_set="train",
                            )

    def test_new_head_partition_receipt_fails_closed_on_contract_drift(self):
        mutations = {
            "canonical hash": lambda receipt, root: receipt.update(
                {"unsealed_field": True}
            ),
            "invariants": lambda receipt, root: receipt["invariants"].update(
                {"fixture_contract_holds": False}
            ),
            "source manifest order": lambda receipt, root: receipt[
                "source_manifest_order"
            ].reverse(),
            "partition summary": lambda receipt, root: receipt[
                "partition_summary"
            ]["train"].update({"rows": 4}),
            "identity count": lambda receipt, root: receipt["outputs"][
                "d0_ordinary_primary"
            ]["train"][self.source_manifests[0]].update(
                {"unique_identities": 2}
            ),
            "manifest size": lambda receipt, root: receipt["outputs"][
                "d0_ordinary_primary"
            ]["train"][self.source_manifests[0]].update({"size_bytes": 1}),
            "manifest path": lambda receipt, root: receipt["outputs"][
                "d0_ordinary_primary"
            ]["train"][self.source_manifests[0]].update(
                {
                    "path": str(
                        root
                        / "d0_ordinary_primary"
                        / "train"
                        / self.source_manifests[1]
                    )
                }
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                receipt_path, receipt = self._partition_fixture(root)
                datasetinfo = self._datasetinfo(receipt_path, receipt)
                mutate(receipt, root)
                self._seal_receipt(
                    receipt_path,
                    receipt,
                    canonical=(label != "canonical hash"),
                )
                datasetinfo["stage_b_data_driven_receipt_sha256"] = self._sha256(
                    receipt_path
                )
                with self.assertRaises(ValueError):
                    _validate_data_driven_ref_dataset_binding(
                        self._args(category_complete=False),
                        datasetinfo,
                        image_set="train",
                    )

    def test_new_head_support_receipt_is_required_and_fails_closed(self):
        mutations = {
            "canonical hash": lambda support, root: support.update(
                {"unsealed_field": True}
            ),
            "invariants": lambda support, root: support["invariants"].update(
                {"fixture_support_contract_holds": False}
            ),
            "input lineage": lambda support, root: support["inputs"][
                "partition_receipt"
            ].update({"sha256": "0" * 64}),
            "summary lineage": lambda support, root: support["partition"].update(
                {"canonical_payload_sha256": "0" * 64}
            ),
            "runtime path": lambda support, root: support["outputs"][
                "runtime_support_tsv"
            ].update({"path": str(root / "support" / "raw_filtered_clean.tsv")}),
            "runtime sha": lambda support, root: support["outputs"][
                "runtime_support_tsv"
            ].update({"sha256": "0" * 64}),
            "required settings": lambda support, root: support[
                "filter_contract"
            ]["required_dataset_settings"].update(
                {"support_patch_max_per_class": 199}
            ),
            "runtime-bank count": lambda support, root: support[
                "runtime_bank"
            ].update({"candidate_rows": 2}),
            "training coverage": lambda support, root: support[
                "training_class_coverage"
            ].update({"missing_class_ids": [1]}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                receipt_path, receipt = self._partition_fixture(root)
                datasetinfo = self._datasetinfo(receipt_path, receipt)
                support_receipt_path = Path(
                    datasetinfo["stage_b_data_driven_support_receipt"]
                )
                support = json.loads(
                    support_receipt_path.read_text(encoding="ascii")
                )
                mutate(support, root)
                self._seal_receipt(
                    support_receipt_path,
                    support,
                    canonical=(label != "canonical hash"),
                )
                datasetinfo[
                    "stage_b_data_driven_support_receipt_sha256"
                ] = self._sha256(support_receipt_path)
                with self.assertRaises(ValueError):
                    _validate_data_driven_ref_dataset_binding(
                        self._args(category_complete=False),
                        datasetinfo,
                        image_set="train",
                    )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt_path, receipt = self._partition_fixture(root)
            datasetinfo = self._datasetinfo(receipt_path, receipt)
            for key in (
                "stage_b_data_driven_support_receipt",
                "stage_b_data_driven_support_receipt_sha256",
            ):
                with self.subTest(missing=key):
                    incomplete = dict(datasetinfo)
                    del incomplete[key]
                    with self.assertRaisesRegex(ValueError, "binding is incomplete"):
                        _validate_data_driven_ref_dataset_binding(
                            self._args(category_complete=False),
                            incomplete,
                            image_set="train",
                        )
            for key, value in (
                ("patch_bank_cache", True),
                ("patch_bank_cache_write", True),
                ("support_patch_use_embedding", True),
                ("support_patch_max_per_class", 199),
            ):
                with self.subTest(setting=key):
                    drifted = dict(datasetinfo)
                    drifted[key] = value
                    with self.assertRaisesRegex(
                        ValueError, "dataset setting drifted"
                    ):
                        _validate_data_driven_ref_dataset_binding(
                            self._args(category_complete=False),
                            drifted,
                            image_set="train",
                        )
            stale_cache_path = dict(datasetinfo)
            stale_cache_path["patch_bank_cache_path"] = str(root / "stale.pkl")
            with self.assertRaisesRegex(
                ValueError, "must not retain a cache path"
            ):
                _validate_data_driven_ref_dataset_binding(
                    self._args(category_complete=False),
                    stale_cache_path,
                    image_set="train",
                )
            bad_receipt_sha = dict(datasetinfo)
            bad_receipt_sha[
                "stage_b_data_driven_support_receipt_sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(ValueError, "support receipt SHA drifted"):
                _validate_data_driven_ref_dataset_binding(
                    self._args(category_complete=False),
                    bad_receipt_sha,
                    image_set="train",
                )
            runtime_path = Path(datasetinfo["support_patch_tsv"])
            runtime_path.write_bytes(runtime_path.read_bytes() + b"drift")
            with self.assertRaisesRegex(
                ValueError, "runtime support TSV binding drifted"
            ):
                _validate_data_driven_ref_dataset_binding(
                    self._args(category_complete=False),
                    datasetinfo,
                    image_set="train",
                )

    def test_support_receipt_is_provenance_asset_but_audit_tsv_is_not_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt_path, receipt = self._partition_fixture(root)
            datasetinfo = self._datasetinfo(receipt_path, receipt)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps({"train": [datasetinfo]}),
                encoding="ascii",
            )
            assets = _stage_b_data_driven_dataset_asset_paths(dataset_path)
            support_receipt_path = Path(
                datasetinfo["stage_b_data_driven_support_receipt"]
            ).resolve(strict=True)
            support = json.loads(
                support_receipt_path.read_text(encoding="ascii")
            )
            runtime_path = Path(datasetinfo["support_patch_tsv"]).resolve(
                strict=True
            )
            audit_path = Path(
                support["outputs"]["audit_raw_tsv"]["path"]
            ).resolve(strict=True)
            self.assertIn(support_receipt_path, assets)
            self.assertIn(runtime_path, assets)
            self.assertNotIn(audit_path, assets)
            records = _stage_b_data_driven_support_pool_content_records(assets)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["support_tsv_path"], str(runtime_path))
            self.assertEqual(records[0]["class_count"], 1)
            self.assertEqual(records[0]["file_count"], 1)

            other_receipt_path, other_receipt = self._support_fixture(
                root / "other",
                receipt_path,
                receipt,
            )
            other_row = dict(datasetinfo)
            other_row["stage_b_data_driven_support_receipt"] = str(
                other_receipt_path
            )
            other_row[
                "stage_b_data_driven_support_receipt_sha256"
            ] = self._sha256(other_receipt_path)
            other_row["support_patch_tsv"] = other_receipt["outputs"][
                "runtime_support_tsv"
            ]["path"]
            dataset_path.write_text(
                json.dumps({"train": [datasetinfo, other_row]}),
                encoding="ascii",
            )
            with self.assertRaisesRegex(RuntimeError, "do not share one support"):
                _stage_b_data_driven_dataset_asset_paths(dataset_path)

            stale_cache_row = dict(datasetinfo)
            stale_cache_row["patch_bank_cache_path"] = str(root / "stale.pkl")
            dataset_path.write_text(
                json.dumps({"train": [stale_cache_row]}),
                encoding="ascii",
            )
            with self.assertRaisesRegex(RuntimeError, "must not retain"):
                _stage_b_data_driven_dataset_asset_paths(dataset_path)

    def test_direct_support_image_replacement_during_hashing_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            support_image = root / "support.jpg"
            support_image.write_bytes(b"support-before")
            support_tsv = root / "support.tsv"
            support_tsv.write_text(
                f"path\tclass_id\n{support_image}\t1\n",
                encoding="ascii",
            )
            original_sha256_file = main_module._sha256_file

            def replace_after_hash(path):
                digest = original_sha256_file(path)
                if Path(path).resolve() == support_image:
                    replacement = root / "replacement.jpg"
                    replacement.write_bytes(b"support-after")
                    replacement.replace(support_image)
                return digest

            with mock.patch.object(
                main_module,
                "_sha256_file",
                side_effect=replace_after_hash,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "direct support image changed while hashing",
                ):
                    _stage_b_data_driven_support_pool_content_records(
                        [support_tsv]
                    )

    def test_legacy_full_data_receipt_remains_valid_without_partition_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            anno = root / self.source_manifests[0]
            anno.write_text('{"row": 1}\n', encoding="ascii")
            manifest_sha = self._sha256(anno)
            receipt = {
                "schema": "pivot.stageb.data_driven_ref_pair_receipt/v1",
                "rows": 321327,
                "unique_identities": 321327,
                "invariants": {"legacy_contract_holds": True},
                "manifests": {
                    anno.name: {
                        "rows": 1,
                        "ordinary_primary": {"sha256": manifest_sha},
                    }
                },
            }
            receipt_path = root / "legacy_receipt.json"
            receipt_path.write_text(json.dumps(receipt), encoding="ascii")
            datasetinfo = {
                "anno": str(anno),
                "stage_b_data_driven_variant": "dd0_ordinary_primary",
                "stage_b_data_driven_manifest_sha256": manifest_sha,
                "stage_b_data_driven_receipt": str(receipt_path),
                "stage_b_data_driven_receipt_sha256": self._sha256(receipt_path),
            }
            self.assertEqual(
                _validate_data_driven_ref_dataset_binding(
                    self._args(category_complete=False),
                    datasetinfo,
                    image_set="train",
                ),
                "dd0_ordinary_primary",
            )


class NewHeadFormalTrainingContractTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    cases = {
        "DD0": (
            root
            / "config/ablations/"
            "cfg_stageb_data_driven_dd0_new_head_formal_20260723.py",
            root
            / "config/"
            "datasets_stageb_data_driven_dd0_new_head_train_20260723.json",
        ),
        "DD1": (
            root
            / "config/ablations/"
            "cfg_stageb_data_driven_dd1_new_head_formal_20260723.py",
            root
            / "config/"
            "datasets_stageb_data_driven_dd1_new_head_train_20260723.json",
        ),
    }

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _args(self, experiment_id="DD0"):
        config_path, dataset_path = self.cases[experiment_id]
        values = SLConfig.fromfile(str(config_path))._cfg_dict.to_dict()
        values.update(
            config_file=str(config_path),
            datasets=str(dataset_path),
            output_dir=values["stage_b_data_driven_formal_output_dir"],
            eval=False,
            resume="",
            pretrain_model_path=values[
                "stage_b_data_driven_base_initializer_path"
            ],
            seed=42,
            max_train_iters=0,
            iter_checkpoint_interval=0,
            gradient_accumulation_steps=1,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=True,
            persistent_workers=False,
            amp=True,
            save_log=True,
            world_size=1,
            distributed=False,
        )
        return types.SimpleNamespace(**values)

    def _validate(
        self,
        args,
        *,
        observed_base_sha=None,
        selection_overrides=None,
        allocator_conf="expandable_segments:True",
    ):
        config_path = Path(args.config_file).resolve(strict=True)
        dataset_path = Path(args.datasets).resolve(strict=True)
        base_path = Path(
            args.stage_b_data_driven_base_initializer_path
        ).resolve(strict=True)
        pair_path = Path(
            args.stage_b_data_driven_initializer_pair_receipt_path
        ).resolve(strict=True)
        selection_receipt = {
            "schema": (
                "pivot.stageb.data_driven.new_head_lr_selection_receipt/v1"
            ),
            "status": "passed",
            "candidate_rank_lrs": [3e-5, 1e-4, 3e-4],
            "optimizer_updates_per_candidate": 1000,
            "selection_partition": "dev_screen",
            "selection_metric": "macro_ref3_acc50",
            "secondary_selection_metric": "macro_ref3_mean_listwise_nll",
            "selected_rank_lr": args.stage_b_data_driven_rank_lr,
        }
        if selection_overrides:
            selection_receipt.update(selection_overrides)
        with tempfile.TemporaryDirectory() as tmp_dir:
            selection_path = Path(tmp_dir) / "selection_receipt.json"
            selection_path.write_text(
                json.dumps(selection_receipt, sort_keys=True) + "\n",
                encoding="ascii",
            )
            args.stage_b_data_driven_new_head_lr_selection_receipt_path = str(
                selection_path
            )
            args.stage_b_data_driven_new_head_lr_selection_receipt_sha256 = (
                self._sha256(selection_path)
            )
            with mock.patch.object(
                main_module,
                "_STAGE_B_DATA_DRIVEN_NEW_HEAD_LR_SELECTION_RECEIPT_PATH",
                str(selection_path),
            ), mock.patch.dict(
                os.environ,
                (
                    {"PYTORCH_CUDA_ALLOC_CONF": allocator_conf}
                    if allocator_conf is not None
                    else {}
                ),
                clear=allocator_conf is None,
            ):
                return _validate_stage_b_data_driven_new_head_formal_training_contract(
                    args,
                    config_path=config_path,
                    dataset_path=dataset_path,
                    base_path=base_path,
                    observed_base_sha=(
                        observed_base_sha
                        or args.stage_b_data_driven_base_initializer_sha256
                    ),
                    pair_path=pair_path,
                    observed_pair_sha=(
                        args.stage_b_data_driven_initializer_pair_receipt_sha256
                    ),
                )

    def test_d0_and_d1_formal_contracts_bind_the_exact_three_epoch_budget(self):
        for experiment_id in self.cases:
            with self.subTest(experiment_id=experiment_id):
                args = self._args(experiment_id)
                binding = self._validate(args)
                self.assertIs(
                    args.stage_b_data_driven_new_head_formal_binding,
                    binding,
                )
                self.assertEqual(binding["experiment_id"], experiment_id)
                self.assertEqual(len(binding["manifests"]), 3)
                self.assertEqual(
                    binding["training_budget"],
                    {
                        "train_rows_per_epoch": 263661,
                        "batch_size": 64,
                        "drop_last": True,
                        "steps_per_epoch": 4119,
                        "dropped_rows_per_epoch": 45,
                        "epochs": 3,
                        "expected_optimizer_updates": 12357,
                        "max_train_iters": 0,
                    },
                )
                self.assertEqual(binding["runtime_support_tsv"]["rows"], 158599)
                self.assertEqual(
                    binding["runtime"]["allocator"]["value"],
                    "expandable_segments:True",
                )

    def test_rank_lr_must_match_a_passed_selection_receipt(self):
        args = self._args("DD0")
        args.stage_b_data_driven_rank_lr = 1e-4
        binding = self._validate(
            args,
            selection_overrides={"selected_rank_lr": 1e-4},
        )
        self.assertEqual(binding["optimizer_contract"]["rank_lr"], 1e-4)
        self.assertEqual(
            binding["optimizer_contract"]["selected_rank_lr"],
            1e-4,
        )

        args = self._args("DD0")
        with self.assertRaisesRegex(
            RuntimeError,
            "LR selection receipt semantics drifted",
        ):
            self._validate(
                args,
                selection_overrides={"selected_rank_lr": 1e-4},
            )

        args = self._args("DD0")
        with self.assertRaisesRegex(
            RuntimeError,
            "LR selection receipt semantics drifted",
        ):
            self._validate(
                args,
                selection_overrides={
                    "secondary_selection_metric": "unexpected_metric"
                },
            )

    def test_probe_scope_does_not_trigger_the_formal_contract(self):
        args = types.SimpleNamespace(
            stage_b_data_driven_execution_scope=(
                "fresh_a0_new_head_lr_probe_u1000_v1"
            )
        )
        result = _validate_stage_b_data_driven_new_head_formal_training_contract(
            args,
            config_path=Path("missing-probe-config.py"),
            dataset_path=Path("missing-probe-dataset.json"),
            base_path=Path("missing-probe-initializer.pth"),
            observed_base_sha="",
            pair_path=None,
            observed_pair_sha=None,
        )
        self.assertIsNone(result)
        self.assertFalse(
            hasattr(args, "stage_b_data_driven_new_head_formal_binding")
        )

    def test_formal_runtime_and_teacher_routes_fail_closed(self):
        mutations = (
            (
                "stage_b_data_driven_new_head_formal_contract",
                "diagnostic_lr_probe_u1000_v1",
            ),
            ("stage_b_data_driven_variant_id", "DD1-PairTop1"),
            ("stage_b_data_driven_category_complete", True),
            ("stage_b_data_driven_rank_architecture", "relational_v1"),
            ("stage_b_data_driven_rank_supervision", "primary_vs_aux_v1"),
            ("stage_b_gdino_score_adapter", True),
            ("stage_b_u0_patch_rank", True),
            ("stage_b_v7", True),
            ("stage_b_v11_fixed_text", True),
            ("stage_b_legacy_global_gate", True),
            ("batch_size", 32),
            ("epochs", 1),
            ("gradient_accumulation_steps", 2),
            ("seed", 43),
            ("amp", False),
            ("fix_size", False),
            ("strong_aug", True),
            ("data_aug_hflip_prob", 0.5),
            ("aux_loss", True),
            ("use_checkpoint", True),
            ("use_transformer_ckpt", True),
            ("weight_decay", 0.0),
            ("clip_max_norm", 1.0),
            ("lr_drop", 3),
            ("onecyclelr", True),
            ("stage_b_data_driven_sampler_seed", 43),
            ("stage_b_data_driven_loader_seed", 1043),
            ("num_workers", 8),
            ("prefetch_factor", 2),
            ("pin_memory", False),
            ("persistent_workers", True),
            ("max_train_iters", 12357),
            ("stage_b_data_driven_formal_expected_optimizer_updates", 5020),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                args = self._args("DD0")
                setattr(args, key, value)
                with self.assertRaisesRegex(
                    RuntimeError, "runtime/model contract drifted"
                ):
                    self._validate(args)

    def test_formal_fresh_start_and_sealed_assets_fail_closed(self):
        config_path, dataset_path = self.cases["DD0"]
        cases = (
            ("resume", str(self.root / "probe.pth"), "--resume is forbidden"),
            (
                "pretrain_model_path",
                str(
                    self.root
                    / "outputs/paper_cvpr_v1/data_driven_initializers/"
                    "fair_v2_seed42/a0_a1_v2_pair_receipt.json"
                ),
                "A0 initializer binding drifted",
            ),
            (
                "stage_b_data_driven_formal_config_path",
                str(dataset_path),
                "config path drifted",
            ),
            (
                "stage_b_data_driven_new_head_dataset_config_sha256",
                "0" * 64,
                "dataset config SHA drifted",
            ),
            (
                "stage_b_data_driven_formal_output_dir",
                str(self.root / "outputs/wrong"),
                "output directory drifted",
            ),
            (
                "stage_b_data_driven_base_initializer_sha256",
                "0" * 64,
                "A0 initializer binding drifted",
            ),
            (
                "stage_b_data_driven_initializer_pair_receipt_sha256",
                "0" * 64,
                "pair receipt binding drifted",
            ),
            (
                "stage_b_data_driven_new_head_partition_receipt_sha256",
                "0" * 64,
                "partition receipt binding",
            ),
            (
                "stage_b_data_driven_new_head_support_receipt_sha256",
                "0" * 64,
                "support receipt binding",
            ),
        )
        for key, value, pattern in cases:
            with self.subTest(key=key):
                args = self._args("DD0")
                setattr(args, key, value)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self._validate(args)

        args = self._args("DD0")
        with self.assertRaisesRegex(RuntimeError, "A0 initializer binding drifted"):
            self._validate(args, observed_base_sha="0" * 64)

    def test_matching_outer_dataset_sha_cannot_hide_row_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = json.loads(
                self.cases["DD0"][1].read_text(encoding="utf-8")
            )
            dataset["train"][0][
                "stage_b_data_driven_variant"
            ] = "dd1_category_complete"
            drifted_path = Path(tmp_dir) / "drifted.json"
            drifted_path.write_text(
                json.dumps(dataset, ensure_ascii=True), encoding="ascii"
            )
            args = self._args("DD0")
            args.datasets = str(drifted_path)
            args.stage_b_data_driven_new_head_dataset_config_path = str(
                drifted_path
            )
            args.stage_b_data_driven_new_head_dataset_config_sha256 = self._sha256(
                drifted_path
            )
            with self.assertRaisesRegex(
                RuntimeError, "dataset row 0 contract drifted"
            ):
                self._validate(args)

    def test_allocator_environment_is_part_of_the_formal_contract(self):
        args = self._args("DD0")
        with self.assertRaisesRegex(
            RuntimeError, "allocator environment drifted"
        ):
            self._validate(args, allocator_conf=None)


class DD1AssignmentTrainingContractTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "config/ablations/"
        "cfg_stageb_data_driven_dd1_pairtop1_fair_v2.py"
    )
    dataset_path = (
        root
        / "config/datasets_stageb_data_driven_dd1_official_assignment_three_ref.json"
    )

    def _args(self):
        values = SLConfig.fromfile(str(self.config_path))._cfg_dict.to_dict()
        values.update(
            eval=False,
            resume="",
            pretrain_model_path=values[
                "stage_b_data_driven_base_initializer_path"
            ],
            max_train_iters=50,
            datasets=str(self.dataset_path),
        )
        return types.SimpleNamespace(**values)

    def _validate(self, args):
        initializer = Path(args.stage_b_data_driven_base_initializer_path)
        return _validate_stage_b_data_driven_assignment_training_contract(
            args,
            base_path=initializer.resolve(strict=True),
            variant_id=args.stage_b_data_driven_variant_id,
        )

    def test_pairtop1_is_the_only_positive_weight_assignment_variant(self):
        args = self._args()
        self.assertEqual(args.stage_b_data_driven_rank_weight, 0.0)
        self.assertEqual(args.stage_b_data_driven_assignment_weight, 1.0)
        self.assertEqual(args.stage_b_data_driven_deployment_weight, 0.0)
        self.assertIsNone(self._validate(args))
        for invalid_weight in (0.0, -1.0, 2.0):
            args = self._args()
            args.stage_b_data_driven_assignment_weight = invalid_weight
            with self.assertRaisesRegex(RuntimeError, "contract drifted"):
                self._validate(args)
        args = self._args()
        args.stage_b_data_driven_rank_weight = 1.0
        with self.assertRaisesRegex(RuntimeError, "contract drifted"):
            self._validate(args)
        args = self._args()
        args.stage_b_data_driven_deployment_weight = 1.0
        with self.assertRaisesRegex(RuntimeError, "contract drifted"):
            self._validate(args)
        args = self._args()
        args.stage_b_data_driven_variant_id = "DD1-X-C0"
        with self.assertRaisesRegex(RuntimeError, "unknown official-assignment"):
            self._validate(args)

    def test_transform_teacher_and_dataset_identity_fail_closed(self):
        for key, value, pattern in (
            ("fix_size", False, "contract drifted"),
            ("strong_aug", True, "contract drifted"),
            ("data_aug_hflip_prob", 0.5, "contract drifted"),
            (
                "stage_b_gdino_score_adapter",
                True,
                "teacher/legacy score routes",
            ),
            (
                "stage_b_data_driven_no_teacher_contract",
                "teacher_anchor_v1",
                "contract drifted",
            ),
            (
                "stage_b_data_driven_assignment_dataset_config_sha256",
                "0" * 64,
                "dataset config SHA drifted",
            ),
        ):
            args = self._args()
            setattr(args, key, value)
            with self.assertRaisesRegex(RuntimeError, pattern):
                self._validate(args)

    def test_hardgap3_is_a_separate_exact_weight_variant(self):
        hard_path = (
            self.root
            / "config/ablations/"
            "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2.py"
        )
        values = SLConfig.fromfile(str(hard_path))._cfg_dict.to_dict()
        values.update(
            eval=False,
            resume="",
            pretrain_model_path=values[
                "stage_b_data_driven_base_initializer_path"
            ],
            max_train_iters=50,
            datasets=str(self.dataset_path),
        )
        args = types.SimpleNamespace(**values)
        self.assertEqual(
            args.stage_b_data_driven_variant_id,
            "DD1-PairTop1-HardGap3",
        )
        self.assertEqual(args.stage_b_data_driven_deployment_weight, 1.0)
        self.assertIsNone(self._validate(args))
        args.stage_b_data_driven_deployment_weight = 0.0
        with self.assertRaisesRegex(RuntimeError, "contract drifted"):
            self._validate(args)

    def test_hardgap3_probe_and_gap3_eval_leaves_preserve_full_contract(self):
        probe_path = (
            self.root
            / "config/ablations/"
            "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2_probe_u50.py"
        )
        probe = SLConfig.fromfile(str(probe_path))._cfg_dict.to_dict()
        self.assertEqual(
            probe["stage_b_data_driven_variant_id"],
            "DD1-PairTop1-HardGap3",
        )
        self.assertEqual(
            probe["stage_b_data_driven_assignment_dataset_scope"],
            "official_assignment_full_321327_v1",
        )
        self.assertEqual(probe["stage_b_data_driven_assignment_weight"], 1.0)
        self.assertEqual(probe["stage_b_data_driven_deployment_weight"], 1.0)
        self.assertEqual(probe["stage_b_data_driven_rank_weight"], 0.0)
        self.assertFalse(probe["stage_b_data_driven_category_gate"])
        self.assertTrue(probe["stage_b_data_driven_probe_fresh_start"])
        self.assertEqual(probe["epochs"], 1)
        self.assertEqual(probe["batch_size"], 64)
        self.assertEqual(
            probe["stage_b_data_driven_probe_expected_max_train_iters"], 50
        )
        self.assertEqual(
            probe[
                "stage_b_data_driven_probe_expected_iter_checkpoint_interval"
            ],
            50,
        )
        self.assertEqual(
            probe["stage_b_data_driven_probe_expected_num_workers"], 4
        )
        self.assertEqual(
            probe["stage_b_data_driven_probe_expected_prefetch_factor"], 1
        )

        probe_args = types.SimpleNamespace(**probe)
        probe_args.eval = False
        probe_args.resume = ""
        probe_args.pretrain_model_path = probe[
            "stage_b_data_driven_base_initializer_path"
        ]
        probe_args.max_train_iters = 50
        probe_args.datasets = str(self.dataset_path)
        self.assertIsNone(self._validate(probe_args))

        eval_path = (
            self.root
            / "config/ablations/"
            "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2_eval_gap3.py"
        )
        evaluation = SLConfig.fromfile(str(eval_path))._cfg_dict.to_dict()
        self.assertEqual(
            evaluation["stage_b_data_driven_variant_id"],
            "DD1-PairTop1-HardGap3",
        )
        self.assertEqual(
            evaluation["stage_b_data_driven_assignment_dataset_scope"],
            "official_assignment_full_321327_v1",
        )
        self.assertTrue(evaluation["stage_b_data_driven_category_gate"])
        self.assertEqual(
            evaluation["stage_b_data_driven_category_gate_max_gap"], 3.0
        )
        self.assertEqual(
            evaluation["stage_b_data_driven_patch_score_clip"], 5.0
        )
        self.assertEqual(
            evaluation["stage_b_data_driven_eval_expected_optimizer_updates"],
            5020,
        )

    def test_resume_and_unsealed_u5020_fail_closed(self):
        args = self._args()
        args.resume = "probe.pth"
        with self.assertRaisesRegex(RuntimeError, "fresh-start"):
            self._validate(args)
        args.resume = ""
        args.max_train_iters = 5020
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            self._validate(args)

    def test_hardgap3_u5020_requires_full_formal_scope(self):
        hard_path = (
            self.root
            / "config/ablations/"
            "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2.py"
        )
        values = SLConfig.fromfile(str(hard_path))._cfg_dict.to_dict()
        values.update(
            eval=False,
            resume="",
            pretrain_model_path=values[
                "stage_b_data_driven_base_initializer_path"
            ],
            max_train_iters=5020,
            datasets=str(self.dataset_path),
        )
        args = types.SimpleNamespace(**values)
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            self._validate(args)
        args.stage_b_data_driven_execution_scope = (
            "formal_fresh_a1_u5020_v1"
        )
        self.assertIsNone(self._validate(args))


class DD1PairTop1Overfit64ContractTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "config/ablations/"
        "cfg_stageb_data_driven_dd1_pairtop1_overfit64_u500.py"
    )
    dataset_path = (
        root
        / "config/datasets_stageb_data_driven_dd1_pairtop1_overfit64.json"
    )

    def _args(self):
        values = SLConfig.fromfile(str(self.config_path))._cfg_dict.to_dict()
        values.update(
            eval=False,
            resume="",
            pretrain_model_path=values[
                "stage_b_data_driven_base_initializer_path"
            ],
            datasets=str(self.dataset_path),
            seed=42,
            max_train_iters=500,
            iter_checkpoint_interval=500,
            num_workers=0,
            prefetch_factor=1,
            pin_memory=False,
            gradient_accumulation_steps=1,
            amp=True,
            save_log=True,
            world_size=1,
            distributed=False,
        )
        return types.SimpleNamespace(**values)

    def _validate(self, args):
        initializer = Path(args.stage_b_data_driven_base_initializer_path)
        return _validate_stage_b_data_driven_assignment_training_contract(
            args,
            base_path=initializer.resolve(strict=True),
            variant_id=args.stage_b_data_driven_variant_id,
        )

    def test_exact_u500_contract_and_direct_singleton_support(self):
        args = self._args()
        self.assertIsNone(self._validate(args))
        self.assertFalse(_stage_b_data_driven_epoch_checkpoint_due(args, 499))
        self.assertTrue(_stage_b_data_driven_epoch_checkpoint_due(args, 500))
        assets = _stage_b_data_driven_dataset_asset_paths(self.dataset_path)
        records = _stage_b_data_driven_support_pool_content_records(assets)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["class_count"], 25)
        self.assertEqual(records[0]["file_count"], 25)
        self.assertTrue(
            records[0]["support_tsv_path"].endswith(
                "overfit64_support_clean.tsv"
            )
        )

    def test_u500_runtime_is_fail_closed(self):
        for key, value in (
            ("epochs", 499),
            ("lr_drop", 100),
            ("stage_b_data_driven_epoch_checkpoint_interval", 1),
            ("max_train_iters", 499),
            ("save_checkpoint_interval", 1),
            ("num_workers", 1),
            ("pin_memory", True),
            ("persistent_workers", True),
            ("amp", False),
        ):
            args = self._args()
            setattr(args, key, value)
            with self.assertRaisesRegex(RuntimeError, "Overfit64/U500 runtime"):
                self._validate(args)

    def test_u500_artifact_bindings_are_fail_closed(self):
        for key, pattern in (
            (
                "stage_b_data_driven_assignment_dataset_config_sha256",
                "dataset config SHA drifted",
            ),
            (
                "stage_b_data_driven_assignment_overfit_support_tsv_sha256",
                "support contract drifted",
            ),
            (
                "stage_b_data_driven_assignment_overfit_member_stream_sha256",
                "lineage contract drifted",
            ),
            (
                "stage_b_data_driven_assignment_overfit_heldout_sha256",
                "lineage contract drifted",
            ),
        ):
            args = self._args()
            setattr(args, key, "0" * 64)
            with self.assertRaisesRegex(RuntimeError, pattern):
                self._validate(args)


class DD1HFreshTrainingContractTest(unittest.TestCase):
    def _args(self, *, root: Path):
        return types.SimpleNamespace(
            eval=False,
            resume="",
            pretrain_model_path=str(root / "initializer.pth"),
            stage_b_data_driven_execution_scope="formal_fresh_a1_u5020_v1",
            stage_b_data_driven_formal_fresh_start=True,
            stage_b_data_driven_formal_expected_optimizer_updates=5020,
            stage_b_data_driven_formal_config_path=str(root / "formal.py"),
            stage_b_data_driven_formal_output_dir=str(root / "formal-output"),
            output_dir=str(root / "formal-output"),
            seed=42,
            batch_size=64,
            epochs=1,
            max_train_iters=5020,
            iter_checkpoint_interval=500,
            num_workers=4,
            prefetch_factor=1,
            gradient_accumulation_steps=1,
            amp=True,
            save_log=True,
            world_size=1,
            distributed=False,
        )

    def test_formal_contract_accepts_only_fresh_a1_u5020(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            initializer = root / "initializer.pth"
            config = root / "formal.py"
            initializer.touch()
            config.touch()
            args = self._args(root=root)
            _validate_stage_b_data_driven_dd1h_fresh_training_contract(
                args, config_path=config.resolve(), base_path=initializer.resolve()
            )
            _validate_stage_b_data_driven_dd1h_fresh_training_contract(
                args,
                config_path=config.resolve(),
                base_path=initializer.resolve(),
                variant_id="DD1-HC",
            )
            _validate_stage_b_data_driven_dd1h_fresh_training_contract(
                args,
                config_path=config.resolve(),
                base_path=initializer.resolve(),
                variant_id="DD1-PairTop1-HardGap3",
            )

            args.resume = str(root / "probe.pth")
            with self.assertRaisesRegex(RuntimeError, "--resume is forbidden"):
                _validate_stage_b_data_driven_dd1h_fresh_training_contract(
                    args,
                    config_path=config.resolve(),
                    base_path=initializer.resolve(),
                )
            args.resume = ""
            args.stage_b_data_driven_execution_scope = ""
            with self.assertRaisesRegex(RuntimeError, "sealed fresh-start formal"):
                _validate_stage_b_data_driven_dd1h_fresh_training_contract(
                    args,
                    config_path=config.resolve(),
                    base_path=initializer.resolve(),
                )
            args.stage_b_data_driven_execution_scope = (
                "formal_fresh_a1_u5020_v1"
            )
            args.max_train_iters = 5019
            with self.assertRaisesRegex(RuntimeError, "exactly max_train_iters=5020"):
                _validate_stage_b_data_driven_dd1h_fresh_training_contract(
                    args,
                    config_path=config.resolve(),
                    base_path=initializer.resolve(),
                )


class DD1FormalEvidencePayloadTest(unittest.TestCase):
    def _probe(self, variant_id):
        is_hc = variant_id == "DD1-HC"
        is_hardgap = variant_id == "DD1-PairTop1-HardGap3"
        probe = {
            "schema": (
                "pivot.stageb.data_driven.pairtop1_hardgap3_probe_receipt/v1"
                if is_hardgap
                else (
                    "pivot.stageb.data_driven.dd1_hc_gap3_coverage_probe_receipt/v1"
                    if is_hc
                    else "pivot.stageb.data_driven.dd1_h_strict_probe_receipt/v2"
                )
            ),
            "status": "passed",
            "scope": "memory_and_protocol_probe_only_do_not_resume_into_formal",
            "checkpoint": {
                "optimizer_updates": 50,
                "iteration": 50,
                "checkpoint_reason": "max_train_iters",
            },
            "criterion": {
                "rank_supervision_contract_id": (
                    4 if is_hardgap else (3 if is_hc else 2)
                ),
            },
            "source": {"training_code_files_manifest_sha256": "a" * 64},
            "causal_audit": {
                "all_non_rank_model_tensors_bitwise_equal_to_dd1_h": True,
                "patch_optimizer_state_bitwise_equal_to_dd1_h": True,
            },
            "invariants": {
                "gap3_coverage_mask_uses_detached_patch_scores": True,
                "probe_checkpoint_is_forbidden_as_formal_resume_source": True,
            },
        }
        if is_hardgap:
            probe["criterion"].update(
                criterion_contract_version=4,
                assignment_weight=1.0,
                deployment_weight=1.0,
            )
            probe["data"] = {
                "scope": "official_assignment_full_321327_v1",
                "rows": 321327,
            }
            probe["fresh_start"] = {"resume": ""}
            probe["causal_audit"] = {
                "all_non_rank_model_tensors_bitwise_equal_to_pairtop1": True,
                "patch_optimizer_state_bitwise_equal_to_pairtop1": True,
            }
            probe["invariants"] = {
                "fifty_of_fifty_optimizer_updates_succeeded": True,
                "amp_step_skips_zero": True,
                "all_model_and_optimizer_tensors_are_finite": True,
                "no_teacher_logits_weights_or_loss_targets_are_used": True,
                "probe_checkpoint_is_forbidden_as_formal_resume_source": True,
            }
        return probe

    def _gate(self, variant_id):
        is_hc = variant_id == "DD1-HC"
        is_hardgap = variant_id == "DD1-PairTop1-HardGap3"
        return {
            "schema": (
                "pivot.stageb.data_driven.pairtop1_hardgap3_formal_gate/v1"
                if is_hardgap
                else (
                    "pivot.stageb.data_driven.dd1_hc_formal_gate/v1"
                    if is_hc
                    else "pivot.stageb.data_driven.dd1_h_formal_gate/v1"
                )
            ),
            "status": "sealed_before_training",
            "training": {
                "variant_id": variant_id,
                "output_dir": "/formal",
                "optimizer_updates": 5020,
            },
            "headline_evaluation": {
                "score_route": "patch_category_gate_then_full_text_rank",
                "category_gate_max_gap": 3.0,
            },
            "headline_gate": {"minimum_correct": 3943, "total": 4896},
        }

    def test_h_and_hc_use_distinct_fail_closed_schemas(self):
        for variant_id in (
            "DD1-H",
            "DD1-HC",
            "DD1-PairTop1-HardGap3",
        ):
            probe = self._probe(variant_id)
            gate = self._gate(variant_id)
            self.assertEqual(
                _validate_stage_b_data_driven_formal_evidence_payloads(
                    variant_id=variant_id,
                    probe_payload=probe,
                    gate_payload=gate,
                    expected_output="/formal",
                ),
                "a" * 64,
            )
            probe["criterion"]["rank_supervision_contract_id"] = 99
            with self.assertRaisesRegex(RuntimeError, "probe receipt drifted"):
                _validate_stage_b_data_driven_formal_evidence_payloads(
                    variant_id=variant_id,
                    probe_payload=probe,
                    gate_payload=gate,
                    expected_output="/formal",
                )

    def test_hc_requires_causal_patch_isolation_and_variant_attribution(self):
        probe = self._probe("DD1-HC")
        gate = self._gate("DD1-HC")
        probe["causal_audit"][
            "patch_optimizer_state_bitwise_equal_to_dd1_h"
        ] = False
        with self.assertRaisesRegex(RuntimeError, "causal audit drifted"):
            _validate_stage_b_data_driven_formal_evidence_payloads(
                variant_id="DD1-HC",
                probe_payload=probe,
                gate_payload=gate,
                expected_output="/formal",
            )
        probe = self._probe("DD1-HC")
        gate["training"]["variant_id"] = "DD1-H"
        with self.assertRaisesRegex(RuntimeError, "formal gate contract"):
            _validate_stage_b_data_driven_formal_evidence_payloads(
                variant_id="DD1-HC",
                probe_payload=probe,
                gate_payload=gate,
                expected_output="/formal",
            )


if __name__ == "__main__":
    unittest.main()
