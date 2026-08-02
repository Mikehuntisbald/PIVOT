import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.judge_stageb_fixed_gdino_top1_qwen import (
    EXTRACTION_SCHEMA,
    JUDGE_RUNTIME_POLICY,
    JUDGE_RUNTIME_POLICY_SHA256,
)
from tools.stageb_gdino_fixed_top1_probe_audit import (
    DATA_SCHEMA,
    EXPECTED_EXTRACTION_ROWS,
    FIXED_MAX_SCOPE,
    OBJECTIVE_CONTRACT,
    P3_LOCKED_CONTRACT,
    RESULT_AUDIT_SCHEMA,
    RESULT_AUDIT_KIND,
    SCHEMA,
    FIXED_TOP1_MILESTONES,
    SCOPE,
    FixedTop1ProbeError,
    _FixedTop1LineageReplay,
    _canonical_sha256,
    _resolve_fixed_top1_find_unused_params,
    _validate_queue,
    _validate_target_readiness,
    _validate_source,
    file_record,
    make_preflight,
    make_segment_lineage,
    validate_static,
    validate_verified_pairs,
    verify_evaluation_checkpoint,
)
from tools.verify_stageb_fixed_gdino_top1_vlm_results import (
    VerificationError as P3VerificationError,
    _accepted_pair as _p3_accepted_pair,
    _region_decision as _p3_region_decision,
    _validate_extraction_audit as _p3_validate_extraction_audit,
)
from tools.stageb_gdino_fixed_top1_selection import SelectionError


ROOT = Path(__file__).resolve().parents[1]
STRICT2031 = (
    ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
    / "eval_manifest.jsonl"
)
STRICT1607 = (
    ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
    / "semantic_stageb_union_image_disjoint_manifest.jsonl"
)
HASH_CONTRACT = {
    "schema": "stage-b-transform-row-hash-list-v1",
    "payload": "[[sample_id,train_row_sha256,deploy_row_sha256],...]",
    "ordering": "lexicographic_by_all_three_string_fields",
    "canonicalization": (
        "json.ensure_ascii=true,sort_keys=true,separators=(',',':'),"
        "allow_nan=false;sha256(utf8)"
    ),
    "transform_rows_scope": "accepted_output_rows",
    "extraction_transform_rows_scope": "all_extraction_rows",
}


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _jsonl_record(path: Path, rows: int):
    return {**file_record(path), "rows": rows}


class FixedTop1PairAuditTests(unittest.TestCase):
    checkpoint_sha = "1" * 64
    train_contract_sha = "2" * 64
    deploy_contract_sha = "3" * 64

    def _source_pair(self, index: int, *, image_id: int | None = None):
        sample_id = f"fixed-top1-test-{index}"
        return {
            "adapter_pair_schema": "stage-b-gdino-adapter-semantic-verified-pair-v1",
            "sample_id": sample_id,
            "source": "stage_b_gdino_adapter_semantic_verified",
            "dataset": "refcoco+",
            "split": "train",
            "image_id": int(image_id if image_id is not None else 900_000_000 + index),
            "ann_id": 100 + index,
            "ref_id": 200 + index,
            "sent_id": 300 + index,
            "class_id": 7,
            "file_name": f"COCO_train2014_{index:012d}.jpg",
            "sent": "the red cup",
            "try_tn": "the blue cup",
            "target_bbox_used": [1.0, 2.0, 30.0, 40.0],
            "tn_scope": "image_global_topk_verified",
            "global_tn_verified": True,
            "proposalset_proxy_verified": False,
            "visual_verified_negative": True,
            "semantic_verified_negative": True,
            "cached_proposal_coverage_only": True,
            "global_max_label_is_semantic_extrapolation": True,
            "all_900_gdino_queries_verified": False,
        }

    def _extraction_row(self, index: int, *, image_id: int | None = None):
        source_pair = self._source_pair(index, image_id=image_id)
        target_judgment = {
            "answer": "no",
            "confidence": 0.95,
            "short_reason": "verified negative",
        }
        region = {
            "region_id": f"{40 + index:064x}",
            "origins": ["deploy", "primary", "shadow"],
            "query_ids": [index],
            "bbox_xyxy_original": [1.0, 2.0, 31.0, 42.0],
            "max_overlap": {
                "kind": "target",
                "proposal_id": None,
                "iou": 1.0,
                "source_answer": "NO",
                "source_confidence": 0.95,
                "source_judgment_sha256": _canonical_sha256(target_judgment),
            },
        }
        return {
            "schema": EXTRACTION_SCHEMA,
            "sample_id": source_pair["sample_id"],
            "dataset": source_pair["dataset"],
            "split": source_pair["split"],
            "image_id": source_pair["image_id"],
            "ann_id": source_pair["ann_id"],
            "ref_id": source_pair["ref_id"],
            "sent_id": source_pair["sent_id"],
            "source_pair": {
                "row": source_pair,
                "row_sha256": _canonical_sha256(source_pair),
            },
            "positive_expression": source_pair["sent"],
            "negative_expression": source_pair["try_tn"],
            "checkpoint": {"sha256": self.checkpoint_sha},
            "config": {"unit": "model-config"},
            "data_config": {"unit": "data-config"},
            "code_sha256": "4" * 64,
            "transform": {
                "sha256": f"{10 + index:064x}",
                "static_contract_sha256": self.train_contract_sha,
            },
            "deploy_transform": {
                "sha256": f"{20 + index:064x}",
                "static_contract_sha256": self.deploy_contract_sha,
            },
            "image": {
                "path": f"/tmp/fixed-top1-test-{index}.jpg",
                "sha256": f"{60 + index:064x}",
                "width": 100,
                "height": 80,
            },
            "num_queries": 900,
            "valid_query_count": 900,
            "queries": {
                "primary": {"query_id": index},
                "shadow": {"query_id": index},
                "deploy": {"query_id": index},
            },
            "stability": {
                "epsilon": 0.001,
                "primary_shadow_agree": True,
                "primary_deploy_agree": True,
                "query_ids_by_origin": {
                    "primary": index,
                    "shadow": index,
                    "deploy": index,
                },
                "near_tie_query_ids": [],
            },
            "regions": [region],
            "source_verification": {
                "target": {
                    "bbox_xywh_original": [1.0, 2.0, 30.0, 40.0],
                    "judgment": target_judgment,
                },
                "proposals": [],
            },
            "claims": {
                "frozen_gdino_global_max_regions_extracted": True,
                "train_path_and_deploy_transform_regions_extracted": True,
                "all_900_gdino_queries_verified": False,
                "image_global_semantic_absence_proven": False,
                "portable_to_other_checkpoint_or_transform": False,
            },
            "holdout": {
                "strict2031_manifest_sha256": file_record(STRICT2031)["sha256"],
                "strict1607_manifest_sha256": file_record(STRICT1607)["sha256"],
                "image_disjoint": True,
            },
        }

    def _provenance(self, extraction):
        return {
            "checkpoint_path": "/tmp/checkpoint.pth",
            "checkpoint_sha256": extraction["checkpoint"]["sha256"],
            "image_path": extraction["image"]["path"],
            "image_sha256": extraction["image"]["sha256"],
            "train_transform_row_sha256": extraction["transform"]["sha256"],
            "deploy_transform_row_sha256": extraction["deploy_transform"]["sha256"],
            "train_transform_contract_sha256": extraction["transform"][
                "static_contract_sha256"
            ],
            "deploy_transform_contract_sha256": extraction["deploy_transform"][
                "static_contract_sha256"
            ],
        }

    def _extraction_contract(self, *_args, **_kwargs):
        return {
            "checkpoint": {"sha256": self.checkpoint_sha},
            "checkpoint_sha256": self.checkpoint_sha,
            "model_config": {"unit": "model-config"},
            "data_config": {"unit": "data-config"},
            "code_sha256": "4" * 64,
            "train_transform_contract_sha256": self.train_contract_sha,
            "deploy_transform_contract_sha256": self.deploy_contract_sha,
        }

    def _accepted_row(self, index: int, *, image_id: int | None = None):
        extraction = self._extraction_row(index, image_id=image_id)
        source_pair = extraction["source_pair"]["row"]
        decision = _p3_region_decision(extraction, extraction["regions"][0], None)
        return _p3_accepted_pair(
            source_pair, extraction, [decision], self._provenance(extraction)
        )

    def _validate(self, accepted: Path, audit_path: Path, *, expected_rows=2):
        with mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit.EXPECTED_EXTRACTION_ROWS",
            expected_rows,
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit._p3_validate_extraction_audit",
            side_effect=self._extraction_contract,
        ) as extraction_audit, mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit._p3_validate_provenance",
            side_effect=self._provenance,
        ) as provenance, mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit._p3_validate_source_lineage",
            return_value="f" * 64,
        ) as source_lineage:
            result = validate_verified_pairs(accepted, audit_path)
        self.last_provenance_replays = provenance.call_count
        self.last_source_lineage_replays = source_lineage.call_count
        self.last_extraction_audit_replays = extraction_audit.call_count
        return result

    def _bundle(self, directory: Path, accepted_rows=None, extraction_rows=None):
        accepted_rows = accepted_rows or [self._accepted_row(0), self._accepted_row(1)]
        if extraction_rows is None:
            extraction_rows = []
            for row in accepted_rows:
                index = int(str(row["sample_id"]).rsplit("-", 1)[1])
                extraction_rows.append(
                    self._extraction_row(index, image_id=int(row["image_id"]))
                )
        accepted = directory / "accepted.jsonl"
        rejected = directory / "rejected.jsonl"
        quarantine = directory / "quarantine.jsonl"
        extractions = directory / "extractions.jsonl"
        judgments = directory / "judgments.jsonl"
        extraction_audit = directory / "extraction.audit.json"
        audit_path = directory / "verification.audit.json"
        _write_jsonl(accepted, accepted_rows)
        _write_jsonl(rejected, [])
        _write_jsonl(quarantine, [])
        _write_jsonl(extractions, extraction_rows)
        _write_jsonl(judgments, [])
        _write_json(
            extraction_audit,
            {
                "schema": "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1",
                "kind": "completed_fixed_gdino_top1_vlm_extraction",
                "rows": len(extraction_rows),
            },
        )
        accepted_transform_rows = sorted(
            [
                [
                    row["sample_id"],
                    row["frozen_gdino_train_transform_row_sha256"],
                    row["frozen_gdino_deploy_transform_row_sha256"],
                ]
                for row in accepted_rows
            ],
            key=tuple,
        )
        extraction_transform_rows = sorted(
            [
                [
                    row["sample_id"],
                    row["transform"]["sha256"],
                    row["deploy_transform"]["sha256"],
                ]
                for row in extraction_rows
            ],
            key=tuple,
        )
        extraction_audit_record = {
            **file_record(extraction_audit),
            "schema": "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1",
            "kind": "completed_fixed_gdino_top1_vlm_extraction",
            "rows": len(extraction_rows),
        }
        audit = {
            "schema": RESULT_AUDIT_SCHEMA,
            "kind": RESULT_AUDIT_KIND,
            "inputs": {
                "extractions": _jsonl_record(extractions, len(extraction_rows)),
                "judgments": _jsonl_record(judgments, 0),
                "extraction_audit": extraction_audit_record,
                "strict2031": _jsonl_record(STRICT2031, 2031),
                "strict1607": _jsonl_record(STRICT1607, 1607),
            },
            "rows": len(extraction_rows),
            "regions": sum(len(row["regions"]) for row in extraction_rows),
            "decisions": {
                "accepted": len(accepted_rows),
                "rejected": 0,
                "quarantine": 0,
            },
            "row_reason_counts": {
                "all_union_regions_verified_no": len(extraction_rows)
            },
            "region_reason_counts": {
                "source_inherited_no": sum(
                    len(row["regions"]) for row in extraction_rows
                )
            },
            "region_method_counts": {
                "source": sum(len(row["regions"]) for row in extraction_rows)
            },
            "source_qwen_comparison_counts": {},
            "checkpoint_sha256": self.checkpoint_sha,
            "locked_contract": P3_LOCKED_CONTRACT,
            "transform_sha256": self.train_contract_sha,
            "train_transform_contract_sha256": self.train_contract_sha,
            "deploy_transform_contract_sha256": self.deploy_contract_sha,
            "transform_rows_sha256": _canonical_sha256(accepted_transform_rows),
            "extraction_transform_rows_sha256": _canonical_sha256(
                extraction_transform_rows
            ),
            "transform_rows_hash_contract": HASH_CONTRACT,
            "strict_image_overlap": {"strict2031": 0, "strict1607": 0},
            "outputs": {
                "accepted": _jsonl_record(accepted, len(accepted_rows)),
                "rejected": _jsonl_record(rejected, 0),
                "quarantine": _jsonl_record(quarantine, 0),
            },
            "scope": {
                "tn_scope_compatibility": "image_global_topk_verified",
                "fixed_gdino_max_scope": FIXED_MAX_SCOPE,
                "global_max_label_is_semantic_extrapolation": False,
                "all_900_gdino_queries_verified": False,
                "image_global_semantic_absence_proven": False,
                "portable_to_other_checkpoint_or_transform": False,
            },
        }
        _write_json(audit_path, audit)
        return accepted, audit_path, audit

    def _full_extraction_audit_fixture(self, directory: Path):
        checkpoint = directory / "checkpoint.pth"
        completion = directory / "protocol_train_complete.json"
        model_config = directory / "model_config.py"
        data_config = directory / "data_config.py"
        extractions = directory / "full-extractions.jsonl"
        checkpoint.write_bytes(b"checkpoint")
        _write_json(completion, {"status": "complete"})
        model_config.write_text("value = 1\n", encoding="utf-8")
        data_config.write_text("data = 1\n", encoding="utf-8")
        _write_jsonl(extractions, [{"sample_id": "a"}, {"sample_id": "b"}])

        checkpoint_record = {
            **file_record(checkpoint),
            "protocol_train_complete": file_record(completion),
        }
        train_contract = {"schema": "unit-train-transform-v1"}
        train_contract["sha256"] = _canonical_sha256(train_contract)
        deploy_contract = {"schema": "unit-deploy-transform-v1"}
        deploy_contract["sha256"] = _canonical_sha256(deploy_contract)
        extraction_record = {**file_record(extractions), "rows": 2}
        strict2031_record = _jsonl_record(STRICT2031, 2031)
        strict1607_record = _jsonl_record(STRICT1607, 1607)
        audit = {
            "schema": "stage-b-fixed-gdino-top1-vlm-extraction-audit-v1",
            "kind": "completed_fixed_gdino_top1_vlm_extraction",
            "rows": 2,
            "counts": {"rows": 2, "regions": 2},
            "manifest": dict(extraction_record),
            "output": {
                key: extraction_record[key]
                for key in ("path", "sha256", "size_bytes")
            },
            "checkpoint": checkpoint_record,
            "model_config": file_record(model_config),
            "data_config": file_record(data_config),
            "code": {"code_sha256": "4" * 64},
            "transform_contracts": {
                "train": train_contract,
                "deploy": deploy_contract,
            },
            "holdout": {
                "manifests": {
                    "strict2031": strict2031_record,
                    "strict1607": strict1607_record,
                }
            },
            "runtime": {"tie_epsilon": 0.001},
            "claims": {
                "train_path_and_deploy_transform_regions_extracted": True,
                "all_900_gdino_queries_verified": False,
                "image_global_semantic_absence_proven": False,
                "portable_to_other_checkpoint_or_transform": False,
            },
        }
        audit_path = directory / "full-extraction.audit.json"
        _write_json(audit_path, audit)
        return (
            audit_path,
            audit,
            extractions,
            extraction_record,
            strict2031_record,
            strict1607_record,
        )

    def test_verified_pairs_accept_exact_checkpoint_transform_and_holdout_contract(self):
        self.assertEqual(
            P3_LOCKED_CONTRACT["judge_runtime_policy"], JUDGE_RUNTIME_POLICY
        )
        self.assertEqual(
            P3_LOCKED_CONTRACT["judge_runtime_policy_sha256"],
            JUDGE_RUNTIME_POLICY_SHA256,
        )
        with tempfile.TemporaryDirectory() as raw:
            accepted, audit_path, _audit = self._bundle(Path(raw))
            result = self._validate(accepted, audit_path)
            self.assertEqual(result["annotation"]["rows"], 2)
            self.assertEqual(
                result["frozen_gdino_checkpoint_sha256"], self.checkpoint_sha
            )
            self.assertEqual(result["strict_image_overlap"]["strict2031"], 0)
            self.assertEqual(self.last_provenance_replays, 2)
            self.assertEqual(self.last_source_lineage_replays, 2)
            self.assertEqual(self.last_extraction_audit_replays, 1)

    def test_production_extraction_count_is_exactly_17738(self):
        self.assertEqual(EXPECTED_EXTRACTION_ROWS, 17_738)
        with tempfile.TemporaryDirectory() as raw:
            accepted, audit_path, _audit = self._bundle(Path(raw))
            with self.assertRaisesRegex(FixedTop1ProbeError, "17738"):
                validate_verified_pairs(accepted, audit_path)

    def test_authoritative_extraction_audit_rejects_manifest_config_and_code_tamper(self):
        for mutation, error_pattern in (
            ("manifest", "manifest record drifted"),
            ("config", "hash drift"),
            ("code", "code closure hash is malformed"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                (
                    audit_path,
                    audit,
                    extractions,
                    extraction_record,
                    strict2031_record,
                    strict1607_record,
                ) = self._full_extraction_audit_fixture(Path(raw))
                if mutation == "manifest":
                    audit["manifest"]["sha256"] = "e" * 64
                elif mutation == "config":
                    audit["model_config"]["sha256"] = "e" * 64
                else:
                    audit["code"]["code_sha256"] = "not-a-sha256"
                _write_json(audit_path, audit)
                with mock.patch(
                    "tools.verify_stageb_fixed_gdino_top1_vlm_results."
                    "EXPECTED_EXTRACTION_ROWS",
                    2,
                ), self.assertRaisesRegex(P3VerificationError, error_pattern):
                    _p3_validate_extraction_audit(
                        audit_path,
                        extraction_path=extractions,
                        extraction_record=extraction_record,
                        strict2031_record=strict2031_record,
                        strict1607_record=strict1607_record,
                    )

    def test_replay_binds_each_extraction_code_to_extraction_audit(self):
        with tempfile.TemporaryDirectory() as raw:
            rows = [self._accepted_row(0), self._accepted_row(1)]
            extractions = [self._extraction_row(0), self._extraction_row(1)]
            extractions[0]["code_sha256"] = "5" * 64
            accepted, audit_path, _audit = self._bundle(
                Path(raw), rows, extractions
            )
            with self.assertRaisesRegex(FixedTop1ProbeError, "code closure drifted"):
                self._validate(accepted, audit_path)

    def test_replay_rejects_unused_and_duplicate_judgments(self):
        with tempfile.TemporaryDirectory() as raw:
            accepted, audit_path, _audit = self._bundle(Path(raw))
            with mock.patch(
                "tools.stageb_gdino_fixed_top1_probe_audit._p3_load_judgments",
                return_value={("outside-sample", "outside-region"): {}},
            ), self.assertRaisesRegex(FixedTop1ProbeError, "unused judgment"):
                self._validate(accepted, audit_path)

        with tempfile.TemporaryDirectory() as raw:
            accepted, audit_path, _audit = self._bundle(Path(raw))
            with mock.patch(
                "tools.stageb_gdino_fixed_top1_probe_audit._p3_load_judgments",
                side_effect=P3VerificationError("duplicate judgment at row 2"),
            ), self.assertRaisesRegex(FixedTop1ProbeError, "duplicate judgment"):
                self._validate(accepted, audit_path)

    def test_recomputed_output_record_cannot_hide_false_fixed_max_claim(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            rows = [self._accepted_row(0), self._accepted_row(1)]
            rows[0]["global_max_label_is_semantic_extrapolation"] = True
            accepted, audit_path, _audit = self._bundle(directory, rows)
            with self.assertRaisesRegex(FixedTop1ProbeError, "exact P3 verifier replay"):
                self._validate(accepted, audit_path)

    def test_recomputed_audit_cannot_hide_strict_image_overlap(self):
        first_strict = json.loads(STRICT2031.read_text(encoding="utf-8").splitlines()[0])
        with tempfile.TemporaryDirectory() as raw:
            rows = [
                self._accepted_row(0, image_id=int(first_strict["image_id"])),
                self._accepted_row(1),
            ]
            accepted, audit_path, _audit = self._bundle(Path(raw), rows)
            with self.assertRaisesRegex(FixedTop1ProbeError, "overlaps strict2031"):
                self._validate(accepted, audit_path)

    def test_transform_row_aggregate_is_recomputed(self):
        with tempfile.TemporaryDirectory() as raw:
            accepted, audit_path, audit = self._bundle(Path(raw))
            audit["transform_rows_sha256"] = "f" * 64
            _write_json(audit_path, audit)
            with self.assertRaisesRegex(FixedTop1ProbeError, "transform-row hash"):
                self._validate(accepted, audit_path)

    def test_replay_rejects_tampered_bbox_and_extraction_hash(self):
        for mutation, expected_error in (
            ("bbox", "exact P3 verifier replay"),
            ("hash", "extraction row hash drifted"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                rows = [self._accepted_row(0), self._accepted_row(1)]
                if mutation == "bbox":
                    rows[0]["target_bbox_used"][0] += 5.0
                else:
                    rows[0]["fixed_gdino_extraction_row_sha256"] = "e" * 64
                accepted, audit_path, _audit = self._bundle(Path(raw), rows)
                with self.assertRaisesRegex(FixedTop1ProbeError, expected_error):
                    self._validate(accepted, audit_path)

    def test_replay_rejects_missing_extraction_and_fabricated_region(self):
        with tempfile.TemporaryDirectory() as raw:
            rows = [self._accepted_row(0), self._accepted_row(1)]
            extractions = [self._extraction_row(0), self._extraction_row(2)]
            accepted, audit_path, _audit = self._bundle(
                Path(raw), rows, extractions
            )
            with self.assertRaisesRegex(FixedTop1ProbeError, "no matching extraction"):
                self._validate(accepted, audit_path)

        with tempfile.TemporaryDirectory() as raw:
            rows = [self._accepted_row(0), self._accepted_row(1)]
            fabricated = copy.deepcopy(rows[0]["fixed_gdino_region_verifications"][0])
            fabricated["region_id"] = "d" * 64
            rows[0]["fixed_gdino_region_verifications"].append(fabricated)
            accepted, audit_path, _audit = self._bundle(Path(raw), rows)
            with self.assertRaisesRegex(FixedTop1ProbeError, "exact P3 verifier replay"):
                self._validate(accepted, audit_path)


class FixedTop1QueueAuditTests(unittest.TestCase):
    @staticmethod
    def _criterion(*, count=2, pointer=2):
        return {
            "fpr_positive_queue": torch.zeros(512),
            "fpr_negative_queue": torch.zeros(512),
            "fpr_queue_count": torch.tensor(count),
            "fpr_queue_ptr": torch.tensor(pointer),
        }

    def test_queue_accepts_unbounded_finite_scores_and_ignores_inactive_slots(self):
        criterion = self._criterion()
        criterion["fpr_positive_queue"][:2] = torch.tensor([-2.0, 1.7])
        criterion["fpr_negative_queue"][:2] = torch.tensor([3.5, -4.25])
        criterion["fpr_positive_queue"][2] = torch.nan
        criterion["fpr_negative_queue"][511] = torch.inf
        result = _validate_queue(criterion, require_warm=False)
        self.assertEqual(result["count"], 2)

    def test_queue_rejects_nan_or_inf_in_each_active_slice(self):
        for queue_name, value in (
            ("fpr_positive_queue", torch.nan),
            ("fpr_positive_queue", torch.inf),
            ("fpr_negative_queue", -torch.inf),
        ):
            with self.subTest(queue=queue_name, value=value):
                criterion = self._criterion()
                criterion[queue_name][1] = value
                with self.assertRaisesRegex(FixedTop1ProbeError, "non-finite active"):
                    _validate_queue(criterion, require_warm=False)

        criterion = self._criterion(count=512, pointer=37)
        criterion["fpr_negative_queue"][511] = torch.nan
        with self.assertRaisesRegex(FixedTop1ProbeError, "non-finite active"):
            _validate_queue(criterion, require_warm=True)

    def test_partial_queue_pointer_must_equal_active_count(self):
        for count, pointer in ((0, 1), (2, 0), (511, 0)):
            with self.subTest(count=count, pointer=pointer):
                criterion = self._criterion(count=count, pointer=pointer)
                with self.assertRaisesRegex(
                    FixedTop1ProbeError, "partial queue pointer"
                ):
                    _validate_queue(criterion, require_warm=False)

        full = self._criterion(count=512, pointer=0)
        self.assertEqual(_validate_queue(full, require_warm=True)["pointer"], 0)

    def test_queue_counters_require_exact_integer_buffers(self):
        for key in ("fpr_queue_count", "fpr_queue_ptr"):
            with self.subTest(key=key):
                criterion = self._criterion()
                criterion[key] = criterion[key].float()
                with self.assertRaisesRegex(FixedTop1ProbeError, "malformed"):
                    _validate_queue(criterion, require_warm=False)


class FixedTop1PreflightTests(unittest.TestCase):
    def test_find_unused_params_uses_cli_default_and_rejects_explicit_true(self):
        missing = type("Cfg", (), {})()
        explicit_false = type("Cfg", (), {"find_unused_params": False})()
        explicit_true = type("Cfg", (), {"find_unused_params": True})()
        self.assertIs(_resolve_fixed_top1_find_unused_params(missing), False)
        self.assertIs(
            _resolve_fixed_top1_find_unused_params(explicit_false), False
        )
        with self.assertRaisesRegex(FixedTop1ProbeError, "argparse default False"):
            _resolve_fixed_top1_find_unused_params(explicit_true)

    @staticmethod
    def _minimal_preflight(directory, milestones=FIXED_TOP1_MILESTONES):
        initial = directory / "rank-initial.pth"
        initial.write_bytes(b"audited rank initial")
        static = {
            "objective_contract": {"unit": "objective"},
            "config": {"path": "/tmp/fixed-top1-config.py"},
            "config_import_chain": [],
            "datasets": {"path": "/tmp/fixed-top1-datasets.json"},
            "data_audit": {"path": "/tmp/fixed-top1-data-audit.json"},
            "annotation": {"path": "/tmp/fixed-top1-pairs.jsonl"},
            "verified_pair_contract": {"scope": FIXED_MAX_SCOPE},
            "code": [],
            "orchestration": [],
        }
        preflight = {
            "schema": SCHEMA,
            "kind": "phase_preflight",
            "phase": "fixed-top1-confidence",
            "initial_checkpoint": file_record(initial),
            "fixed_gdino_source_binding": {
                "matches_rank_initial_baseline": True
            },
            "static": static,
            "launch": {
                "max_target": int(milestones[-1]),
                "milestones": list(milestones),
            },
        }
        preflight_path = directory / f"preflight-{milestones[-1]}.json"
        _write_json(preflight_path, preflight)
        return initial, preflight, preflight_path, static

    def _build_six_milestone_replay(self, directory):
        initial, preflight, preflight_path, static = self._minimal_preflight(
            directory
        )
        preflight_record = file_record(preflight_path)
        queue = {
            "count": 512,
            "pointer": 0,
            "capacity": 512,
            "minimum_for_q05": 256,
        }
        chain = {}
        previous_checkpoint = None
        previous_audit = None
        for target in FIXED_TOP1_MILESTONES:
            checkpoint = directory / f"checkpoint-{target}.pth"
            checkpoint.write_bytes(f"fixed-top1-{target}".encode("ascii"))
            checkpoint_record = {
                **file_record(checkpoint),
                "rank_sha256": "a" * 64,
                "confidence_sha256": f"{target:064x}",
            }
            source = initial if previous_checkpoint is None else previous_checkpoint
            lineage = directory / f"segment-{target}.json"
            _write_json(
                lineage,
                {
                    "schema": SCHEMA,
                    "kind": "segment_lineage",
                    "phase": "confidence",
                    "confidence_protocol": "fixed_gdino_top1_verified_v1",
                    "tn_scope": SCOPE,
                    "expected_target": target,
                    "initialization_mode": (
                        "pretrain" if previous_checkpoint is None else "resume"
                    ),
                    "ancestry": (
                        "phase_initial"
                        if previous_checkpoint is None
                        else "previous_milestone"
                    ),
                    "source_checkpoint": file_record(source),
                    "preflight": preflight_record,
                    "previous_audit": (
                        file_record(previous_audit) if previous_audit else None
                    ),
                    "recovery_inspection": None,
                },
            )
            audit = directory / f"checkpoint-{target}.audit.json"
            _write_json(
                audit,
                {
                    "schema": SCHEMA,
                    "kind": "milestone_checkpoint",
                    "phase": "confidence",
                    "confidence_protocol": "fixed_gdino_top1_verified_v1",
                    "tn_scope": SCOPE,
                    "iteration": target,
                    "global_batch": 8,
                    "initialization_mode": (
                        "pretrain_model_path"
                        if previous_checkpoint is None
                        else "resume"
                    ),
                    "queue": queue,
                    "segment_lineage": file_record(lineage),
                    "source_checkpoint": file_record(source),
                    "preflight": preflight_record,
                    "previous_audit": (
                        file_record(previous_audit) if previous_audit else None
                    ),
                    "objective_contract": static["objective_contract"],
                    "config": static["config"],
                    "config_import_chain": static["config_import_chain"],
                    "datasets": static["datasets"],
                    "data_audit": static["data_audit"],
                    "annotation": static["annotation"],
                    "verified_pair_contract": static["verified_pair_contract"],
                    "fixed_gdino_source_binding": preflight[
                        "fixed_gdino_source_binding"
                    ],
                    "code": static["code"],
                    "orchestration": static["orchestration"],
                    "checkpoint": checkpoint_record,
                },
            )
            chain[target] = {
                "checkpoint": checkpoint,
                "checkpoint_record": checkpoint_record,
                "source": source,
                "lineage": lineage,
                "audit": audit,
            }
            previous_checkpoint = checkpoint
            previous_audit = audit

        replay = object.__new__(_FixedTop1LineageReplay)
        replay.preflight_path = preflight_path.resolve()
        replay.preflight_record = preflight_record
        replay.preflight = preflight
        replay.static = static
        replay.milestone_cache = {}
        replay.segment_cache = {}
        replay.inspection_cache = {}
        replay._milestone_stack = set()
        replay._segment_stack = set()
        replay._inspection_stack = set()
        return replay, chain, queue

    def test_extended_segment_lineage_uses_adjacent_500_anchor_for_1000(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            _, _, preflight_path, _ = self._minimal_preflight(directory)
            checkpoint_500 = directory / "checkpoint-500.pth"
            checkpoint_500.write_bytes(b"checkpoint 500")
            audit_500 = directory / "checkpoint-500.audit.json"
            _write_json(
                audit_500,
                {
                    "schema": SCHEMA,
                    "kind": "milestone_checkpoint",
                    "phase": "confidence",
                    "iteration": 500,
                    "checkpoint": file_record(checkpoint_500),
                },
            )
            segment_1000 = make_segment_lineage(
                preflight_path=preflight_path,
                expected_target=1000,
                source_checkpoint=checkpoint_500,
                initialization_mode="resume",
                previous_audit_path=audit_500,
                recovery_inspection_path=None,
            )
            self.assertEqual(segment_1000["ancestry"], "previous_milestone")
            self.assertEqual(
                segment_1000["previous_audit"], file_record(audit_500)
            )

    def test_extended_segment_lineage_rejects_skip_and_prefix_escape(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            _, _, full_preflight_path, _ = self._minimal_preflight(directory)
            checkpoint_500 = directory / "checkpoint-500.pth"
            checkpoint_250 = directory / "checkpoint-250.pth"
            checkpoint_500.write_bytes(b"checkpoint 500")
            checkpoint_250.write_bytes(b"checkpoint 250")
            audit_500 = directory / "checkpoint-500.audit.json"
            _write_json(
                audit_500,
                {
                    "schema": SCHEMA,
                    "kind": "milestone_checkpoint",
                    "phase": "confidence",
                    "iteration": 500,
                    "checkpoint": file_record(checkpoint_500),
                },
            )
            audit_250 = directory / "checkpoint-250.audit.json"
            _write_json(
                audit_250,
                {
                    "schema": SCHEMA,
                    "kind": "milestone_checkpoint",
                    "phase": "confidence",
                    "iteration": 250,
                    "checkpoint": file_record(checkpoint_250),
                },
            )
            with self.assertRaisesRegex(
                FixedTop1ProbeError, "exact previous milestone"
            ):
                make_segment_lineage(
                    preflight_path=full_preflight_path,
                    expected_target=1000,
                    source_checkpoint=checkpoint_250,
                    initialization_mode="resume",
                    previous_audit_path=audit_250,
                    recovery_inspection_path=None,
                )

            _, _, bounded_preflight_path, _ = self._minimal_preflight(
                directory, milestones=(50, 100, 250, 500)
            )
            with self.assertRaisesRegex(
                FixedTop1ProbeError, "outside the preflight milestone prefix"
            ):
                make_segment_lineage(
                    preflight_path=bounded_preflight_path,
                    expected_target=1000,
                    source_checkpoint=checkpoint_500,
                    initialization_mode="resume",
                    previous_audit_path=audit_500,
                    recovery_inspection_path=None,
                )

    def test_deep_replay_accepts_complete_registered_milestone_chain(self):
        with tempfile.TemporaryDirectory() as raw:
            replay, chain, queue = self._build_six_milestone_replay(Path(raw))

            def validate_checkpoint_stub(**kwargs):
                target = int(kwargs["expected_target"])
                node = chain[target]
                self.assertEqual(
                    kwargs["checkpoint"].resolve(), node["checkpoint"].resolve()
                )
                self.assertEqual(
                    kwargs["source_checkpoint"].resolve(), node["source"].resolve()
                )
                self.assertTrue(kwargs["exact"])
                return {
                    "record": node["checkpoint_record"],
                    "iteration": target,
                    "initialization_mode": (
                        "pretrain_model_path" if target == 50 else "resume"
                    ),
                    "queue": queue,
                }

            with mock.patch(
                "tools.stageb_gdino_fixed_top1_probe_audit.validate_checkpoint",
                side_effect=validate_checkpoint_stub,
            ):
                result = replay.verify_milestone(
                    chain[1000]["audit"],
                    expected_checkpoint=chain[1000]["checkpoint"],
                )

        self.assertEqual(result["audit"]["iteration"], 1000)
        self.assertEqual(len(replay.milestone_cache), len(FIXED_TOP1_MILESTONES))
        self.assertEqual(len(replay.segment_cache), len(FIXED_TOP1_MILESTONES))

    def test_launcher_dry_run_constructs_all_extended_resume_commands(self):
        launcher = ROOT / "tools/run_stageb_gdino_fixed_top1_confidence_probe.sh"
        with tempfile.TemporaryDirectory() as raw:
            output_root = Path(raw) / "probe"
            environment = os.environ.copy()
            environment["PYTHON_BIN"] = "/bin/true"
            completed = subprocess.run(
                [
                    str(launcher),
                    "--source-checkpoint",
                    "/tmp/audited-rank.pth",
                    "--output-root",
                    str(output_root),
                    "--max-target",
                    "1000",
                    "--dry-run",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = [
            line
            for line in completed.stdout.splitlines()
            if "--max_train_iters" in line
        ]
        self.assertEqual(len(commands), len(FIXED_TOP1_MILESTONES))
        for index, (target, command) in enumerate(
            zip(FIXED_TOP1_MILESTONES, commands)
        ):
            self.assertIn(f"--max_train_iters {target} ", command)
            if index == 0:
                self.assertIn("--pretrain_model_path /tmp/audited-rank.pth", command)
                self.assertNotIn(" --resume ", command)
            else:
                previous = FIXED_TOP1_MILESTONES[index - 1]
                self.assertIn(" --resume ", command)
                self.assertIn(f"checkpoint_iter_{previous:06d}.pth", command)

    def test_formal_eval_wrapper_dispatches_fixed_top1_audit_read_only(self):
        wrapper = (
            ROOT / "tools/run_stageb_gdino_adapter_probe_eval.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "auto|two-phase|semantic-confidence|fixed-top1-confidence", wrapper
        )
        self.assertIn(
            "stageb-gdino-adapter-fixed-top1-confidence-probe-v1) "
            'selected_kind="fixed-top1-confidence"',
            wrapper,
        )
        self.assertIn(
            '"${PYTHON_BIN}" tools/stageb_gdino_fixed_top1_probe_audit.py '
            "verify-evaluation",
            wrapper,
        )
        self.assertIn(
            'elif audit.get("schema") == '
            '"stageb-gdino-adapter-fixed-top1-confidence-probe-v1":',
            wrapper,
        )
        self.assertIn('print(audit["config"]["path"])', wrapper)
        self.assertIn("dry-run does not deep-verify candidate checkpoint lineage", wrapper)
        self.assertIn("LINEAGE_POST_OUTPUT", wrapper)
        self.assertIn("pre/post evaluation lineage replay differs", wrapper)
        self.assertIn("CANDIDATE_REUSE_ALLOWED=0", wrapper)
        self.assertIn(
            "fixed-top1 candidate evaluation must be fresh after the pre-eval lineage seal",
            wrapper,
        )
        self.assertNotIn("rewrites LINEAGE_OUTPUT", wrapper)

    def test_verify_evaluation_exports_formal_dispatch_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            checkpoint = directory / "checkpoint_iter_000050.pth"
            audit_path = directory / "checkpoint_iter_000050.audit.json"
            preflight_path = directory / "probe_preflight.json"
            checkpoint.write_bytes(b"checkpoint")
            _write_json(preflight_path, {"sealed": True})
            _write_json(
                audit_path,
                {"preflight": file_record(preflight_path)},
            )
            replay = mock.Mock()
            replay.static = {
                "objective_contract": OBJECTIVE_CONTRACT,
                "config": {"path": str(ROOT / "config/ablations/cfg_stageb_gdino_score_adapter_fixed_top1_verified.py")},
                "config_import_chain": [],
                "datasets": {},
                "data_audit": {},
                "annotation": {},
                "source_annotation": {},
                "partition": {},
                "verified_pair_contract": {"scope": FIXED_MAX_SCOPE},
                "code": {},
                "orchestration": {},
            }
            replay.preflight = {
                "fixed_gdino_source_binding": {"matches_rank_initial_baseline": True}
            }
            replay.preflight_record = file_record(preflight_path)
            replay.milestone_cache = {audit_path: {}}
            replay.segment_cache = {directory / "segment.json": {}}
            replay.inspection_cache = {}
            replay.verify_milestone.return_value = {
                "audit": {
                    "iteration": 50,
                    "segment_lineage": {"path": str(directory / "segment.json")},
                    "source_checkpoint": {"sha256": "1" * 64},
                    "previous_audit": None,
                },
                "segment": {"lineage": {"recovery_inspection": None}},
                "checkpoint": file_record(checkpoint),
            }
            with mock.patch(
                "tools.stageb_gdino_fixed_top1_probe_audit._FixedTop1LineageReplay",
                return_value=replay,
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_probe_audit.verify_selection",
                side_effect=SelectionError("missing held-out authorization"),
            ), self.assertRaisesRegex(
                FixedTop1ProbeError, "not authorized by held-out selection"
            ):
                verify_evaluation_checkpoint(
                    checkpoint_path=checkpoint,
                    audit_path=audit_path,
                )
            selection = {
                "audit": {"path": str(directory / "selection.json"), "size_bytes": 1, "sha256": "a" * 64},
                "selected_checkpoint": file_record(checkpoint),
                "selected_milestone_audit": file_record(audit_path),
                "selected_iteration": 50,
                "calibration_root": {
                    "path": str((directory / "calibration_selection").resolve()),
                    "layout": "p0_and_six_digit_milestone_subdirectories",
                },
                "input_scope": "calibration_only",
                "strict_paths_consumed_for_scoring": False,
            }
            with mock.patch(
                "tools.stageb_gdino_fixed_top1_probe_audit._FixedTop1LineageReplay",
                return_value=replay,
            ), mock.patch(
                "tools.stageb_gdino_fixed_top1_probe_audit.verify_selection",
                return_value=selection,
            ):
                result = verify_evaluation_checkpoint(
                    checkpoint_path=checkpoint,
                    audit_path=audit_path,
                )
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["kind"], "evaluation_checkpoint_verification")
        self.assertEqual(result["confidence_protocol"], "fixed_gdino_top1_verified_v1")
        self.assertTrue(result["config"]["path"].endswith("fixed_top1_verified.py"))
        self.assertEqual(
            result["verified_pair_contract"]["scope"], FIXED_MAX_SCOPE
        )
        self.assertTrue(
            result["fixed_gdino_source_binding"]["matches_rank_initial_baseline"]
        )
        self.assertTrue(
            result["selection_authorization"]["formal_strict_authorization"]
        )
        self.assertEqual(
            result["selection_authorization"]["input_scope"], "calibration_only"
        )

    def test_static_config_and_launch_files_keep_the_p3_confidence_contract(self):
        fake_pairs = {
            "data_audit": {"path": "audit", "size_bytes": 1, "sha256": "1" * 64},
            "annotation": {
                "path": "pairs",
                "size_bytes": 1,
                "sha256": "2" * 64,
                "rows": 4000,
            },
            "inputs": {},
            "outputs": {},
            "frozen_gdino_checkpoint_sha256": "3" * 64,
            "train_transform_contract_sha256": "4" * 64,
            "deploy_transform_contract_sha256": "5" * 64,
            "transform_rows_sha256": "6" * 64,
            "extraction_transform_rows_sha256": "7" * 64,
            "transform_rows_hash_contract": HASH_CONTRACT,
            "strict_image_overlap": {"strict2031": 0, "strict1607": 0},
            "scope": {},
        }
        train_path = ROOT / "data/ablations/stageb_gdino_adapter_fixed_top1_verified_20260712/train_pairs.jsonl"
        accepted_path = Path("/tmp/fixed-top1-accepted.jsonl")
        fake_partition = {
            "audit": {"path": "/tmp/partition.json", "size_bytes": 1, "sha256": "8" * 64},
            "accepted": {"path": str(accepted_path), "size_bytes": 1, "sha256": "2" * 64, "rows": 4000},
            "train": {"path": str(train_path), "size_bytes": 1, "sha256": "9" * 64, "rows": 4000},
            "calibration": {"path": "/tmp/calibration.jsonl", "size_bytes": 1, "sha256": "a" * 64, "rows": 1000},
            "recommended_max_target": 500,
            "selection_readiness": {"pass": True, "errors": []},
        }
        original_read_json = __import__(
            "tools.stageb_gdino_fixed_top1_probe_audit", fromlist=["read_json"]
        ).read_json

        def fake_read_json(path):
            if str(path).endswith("verification_audit.json"):
                return {"outputs": {"accepted": {"path": str(accepted_path)}}}
            return original_read_json(path)

        with mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit.validate_verified_pairs",
            return_value=fake_pairs,
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit.verify_partition",
            return_value=fake_partition,
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit.read_json",
            side_effect=fake_read_json,
        ), mock.patch(
            "tools.extract_stageb_fixed_gdino_top1_vlm_manifest.transform_contract_from_cfg",
            return_value={"sha256": "4" * 64},
        ), mock.patch(
            "tools.extract_stageb_fixed_gdino_top1_vlm_manifest.deploy_transform_contract_from_cfg",
            return_value={"sha256": "5" * 64},
        ):
            static = validate_static()
        resolved = static["resolved_contract"]
        self.assertEqual(resolved["stage_b_gdino_adapter_train_mode"], "confidence_only")
        self.assertEqual(resolved["stage_b_gdino_gate_lr"], 3.0e-4)
        self.assertEqual(resolved["stage_b_gdino_gate_pool_temperature"], 0.01)
        self.assertEqual(resolved["stage_b_gdino_gate_topk"], 3)
        self.assertEqual(resolved["data_aug_hflip_prob"], 0.0)
        launch = (
            ROOT / "tools/run_stageb_gdino_fixed_top1_confidence_probe.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--nproc_per_node=\"${WORLD_SIZE}\"", launch)
        self.assertIn("MILESTONES=(50 100 250 500 1000)", launch)
        self.assertIn("--max-target", launch)
        self.assertIn("--pretrain_model_path", launch)
        self.assertIn("--resume", launch)
        self.assertNotIn("dataft-confidence", launch)

    def test_preflight_is_independent_rank_only_ddp2xb4_schema(self):
        source = Path("/tmp/rank-250.pth")
        source_audit = Path("/tmp/rank-250.audit.json")
        static = {
            "annotation": {"rows": 4000},
            "partition": {
                "selection_readiness": {"pass": True},
                "recommended_max_target": 500,
            },
            "verified_pair_contract": {
                "frozen_gdino_checkpoint_sha256": "a" * 64
            }
        }
        with mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit._validate_source",
            return_value={"path": str(source), "sha256": "b" * 64},
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit.validate_static",
            return_value=static,
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit._rank_source_baseline_sha256",
            return_value="a" * 64,
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit.file_record",
            return_value={"path": str(source_audit), "sha256": "c" * 64},
        ):
            preflight = make_preflight(
                source_kind="rank",
                source_checkpoint=source,
                source_audit=source_audit,
                world_size=2,
                per_gpu_batch=4,
            )
        self.assertEqual(preflight["schema"], SCHEMA)
        self.assertEqual(preflight["phase"], "fixed-top1-confidence")
        self.assertEqual(preflight["launch"]["global_batch"], 8)
        self.assertTrue(
            preflight["fixed_gdino_source_binding"][
                "matches_rank_initial_baseline"
            ]
        )

    def test_preflight_rejects_checkpoint_mismatch_and_non_rank_source(self):
        source = Path("/tmp/source.pth")
        audit = Path("/tmp/source.audit.json")
        static = {
            "annotation": {"rows": 4000},
            "partition": {
                "selection_readiness": {"pass": True},
                "recommended_max_target": 500,
            },
            "verified_pair_contract": {
                "frozen_gdino_checkpoint_sha256": "a" * 64
            }
        }
        with mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit._validate_source",
            return_value={},
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit.validate_static",
            return_value=static,
        ), mock.patch(
            "tools.stageb_gdino_fixed_top1_probe_audit._rank_source_baseline_sha256",
            return_value="b" * 64,
        ):
            with self.assertRaisesRegex(FixedTop1ProbeError, "different baseline"):
                make_preflight(
                    source_kind="rank",
                    source_checkpoint=source,
                    source_audit=audit,
                    world_size=2,
                    per_gpu_batch=4,
                )
        with self.assertRaisesRegex(FixedTop1ProbeError, "rank milestone"):
            _validate_source("dataft-confidence", source, audit)

    def test_max_target_prefix_is_bounded_by_available_rows(self):
        static = {
            "annotation": {"rows": 2000},
            "partition": {
                "selection_readiness": {"pass": True},
                "recommended_max_target": 250,
            },
        }
        self.assertEqual(
            _validate_target_readiness(static, max_target=250), 2000
        )
        with self.assertRaisesRegex(FixedTop1ProbeError, "largest supported"):
            _validate_target_readiness(static, max_target=500)

        extended = {
            "annotation": {"rows": 16000},
            "partition": {
                "selection_readiness": {"pass": True},
                "recommended_max_target": 1000,
            },
        }
        with self.assertRaisesRegex(FixedTop1ProbeError, "largest supported"):
            _validate_target_readiness(extended, max_target=500)
        self.assertEqual(
            _validate_target_readiness(extended, max_target=1000), 8000
        )


if __name__ == "__main__":
    unittest.main()
