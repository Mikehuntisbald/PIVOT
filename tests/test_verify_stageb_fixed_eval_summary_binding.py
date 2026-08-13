import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools import eval_text_groundingdino_refcoco_tn as text_eval
from tools.stageb_eval_records import (
    EvalManifest,
    RECORD_SCHEMA,
    TN_DERIVATION_ALGORITHM,
    TN_DERIVED_MANIFEST_BINDING_SCHEMA,
    load_eval_manifest,
    make_eval_record,
    tn_manifest_binding_summary_fields,
    tn_manifest_derivation_contract,
    write_tn_derived_manifest_binding,
)
from tools import verify_stageb_fixed_eval_summary_binding as binding


SMALL_TN_SECTIONS = {"strict2031": 2, "strict1607": 1}


def _small_ref_manifest_rows(split):
    return [
        {
            "image_id": 1000 + index,
            "ann_id": 2000 + index,
            "ref_id": 3000 + index,
            "sent_id": 4000 + index,
            "split": split,
        }
        for index in range(2)
    ]


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode(
        "utf-8"
    )


def _small_ref_contract():
    return {
        split: {
            "rows": 2,
            "sha256": hashlib.sha256(
                _jsonl_bytes(_small_ref_manifest_rows(split))
            ).hexdigest(),
        }
        for split in binding.REF_SPLITS
    }


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _two_phase_rank_milestone(
    root: Path, baseline: Path, checkpoint: Path, *, schema: str = binding.TWO_PHASE_SCHEMA
) -> Path:
    preflight = root / "rank.preflight.json"
    _write_json(
        preflight,
        {
            "schema": schema,
            "kind": "phase_preflight",
            "phase": "rank",
            "initial_checkpoint": binding._file_record(baseline.resolve()),
        },
    )
    milestone = root / "rank.milestone.json"
    _write_json(
        milestone,
        {
            "schema": schema,
            "kind": "milestone_checkpoint",
            "phase": "rank",
            "checkpoint": binding._file_record(checkpoint.resolve()),
            "preflight": binding._file_record(preflight.resolve()),
        },
    )
    return milestone


def _total_confidence_milestone_with_historical_rank(
    root: Path, baseline: Path, rank_checkpoint: Path, candidate: Path
) -> Path:
    rank_preflight = root / "historical-rank.preflight.json"
    _write_json(
        rank_preflight,
        {
            "schema": binding.TWO_PHASE_SCHEMA,
            "kind": "phase_preflight",
            "phase": "rank",
            "initial_checkpoint": binding._file_record(baseline.resolve()),
        },
    )
    rank_milestone = root / "historical-rank.milestone.json"
    _write_json(
        rank_milestone,
        {
            "schema": binding.TWO_PHASE_SCHEMA,
            "kind": "milestone_checkpoint",
            "phase": "rank",
            "checkpoint": binding._file_record(rank_checkpoint.resolve()),
            "preflight": binding._file_record(rank_preflight.resolve()),
        },
    )
    confidence_preflight = root / "total-confidence.preflight.json"
    _write_json(
        confidence_preflight,
        {
            "schema": binding.TOTAL_TRUST_SCHEMA,
            "kind": "phase_preflight",
            "phase": "confidence",
            "initial_checkpoint": binding._file_record(rank_checkpoint.resolve()),
            "initial_audit": binding._file_record(rank_milestone.resolve()),
        },
    )
    confidence_milestone = root / "total-confidence.milestone.json"
    _write_json(
        confidence_milestone,
        {
            "schema": binding.TOTAL_TRUST_SCHEMA,
            "kind": "milestone_checkpoint",
            "phase": "confidence",
            "checkpoint": binding._file_record(candidate.resolve()),
            "preflight": binding._file_record(confidence_preflight.resolve()),
        },
    )
    return confidence_milestone


def _trusted_non_p0_lineage(
    root: Path,
    schema: str,
    baseline: Path,
    candidate: Path,
    *,
    config: Path | None = None,
) -> Path:
    if config is None:
        config = root / "candidate_config.py"
        config.write_text("stage_b_gdino_score_adapter = True\n", encoding="utf-8")
    if schema in (binding.TWO_PHASE_SCHEMA, binding.TOTAL_TRUST_SCHEMA):
        milestone = _two_phase_rank_milestone(
            root, baseline, candidate, schema=schema
        )
        payload = {
            "schema": schema,
            "kind": "evaluation_checkpoint_verified",
            "checkpoint": binding._file_record(candidate.resolve()),
            "config": binding._file_record(config.resolve()),
            "audit": binding._file_record(milestone.resolve()),
        }
    else:
        source = root / "rank-source.pth"
        source.write_bytes(b"rank source")
        milestone = _two_phase_rank_milestone(root, baseline, source)
        phase = (
            "semantic-confidence"
            if schema == binding.SEMANTIC_SCHEMA
            else "fixed-top1-confidence"
        )
        preflight = root / f"{phase}.preflight.json"
        _write_json(
            preflight,
            {
                "schema": schema,
                "kind": "phase_preflight",
                "phase": phase,
                "initial_checkpoint": binding._file_record(source.resolve()),
                "initial_audit": binding._file_record(milestone.resolve()),
            },
        )
        checkpoint_audit = root / f"{phase}.milestone.json"
        _write_json(
            checkpoint_audit,
            {
                "schema": schema,
                "kind": "milestone_checkpoint",
                "checkpoint": binding._file_record(candidate.resolve()),
                "preflight": binding._file_record(preflight.resolve()),
            },
        )
        payload = {
            "schema": schema,
            "kind": "evaluation_checkpoint_verification",
            "checkpoint": binding._file_record(candidate.resolve()),
            "config": binding._file_record(config.resolve()),
            "checkpoint_audit": binding._file_record(checkpoint_audit.resolve()),
            "preflight": binding._file_record(preflight.resolve()),
            "verified": True,
        }
        if schema == binding.FIXED_TOP1_SCHEMA:
            payload["fixed_gdino_source_binding"] = {
                "checkpoint_sha256": binding._file_record(baseline.resolve())["sha256"],
                "matches_rank_initial_baseline": True,
            }
            selection_path = root / "fixed-top1.selection.json"
            _write_json(selection_path, {"sealed": True})
            payload["selection_authorization"] = {
                "schema": binding.SELECTION_SCHEMA,
                "audit": binding._file_record(selection_path.resolve()),
                "selected_checkpoint": binding._file_record(candidate.resolve()),
                "selected_milestone_audit": binding._file_record(
                    checkpoint_audit.resolve()
                ),
                "selected_iteration": 50,
                "calibration_root": {
                    "path": str(
                        (selection_path.parent.parent / "calibration_selection").resolve()
                    ),
                    "layout": "p0_and_six_digit_milestone_subdirectories",
                },
                "input_scope": "calibration_only",
                "strict_paths_consumed_for_scoring": False,
                "formal_strict_authorization": True,
            }
    lineage = root / "trusted-lineage.json"
    _write_json(lineage, payload)
    return lineage


def _selection_replay_stub(
    path,
    *,
    expected_checkpoint,
    expected_milestone_audit,
    expected_calibration_root=None,
):
    return {
        "audit": binding._file_record(Path(path).resolve()),
        "selected_checkpoint": binding._file_record(
            Path(expected_checkpoint).resolve()
        ),
        "selected_milestone_audit": binding._file_record(
            Path(expected_milestone_audit).resolve()
        ),
        "selected_iteration": 50,
        "calibration_root": {
            "path": str(Path(expected_calibration_root).resolve()),
            "layout": "p0_and_six_digit_milestone_subdirectories",
        },
        "input_scope": "calibration_only",
        "strict_paths_consumed_for_scoring": False,
        "verified": True,
    }


def _trusted_p0_lineage(
    root: Path,
    baseline: Path,
    candidate: Path,
    *,
    config: Path | None = None,
) -> Path:
    if config is None:
        config = root / "candidate_config.py"
        config.write_text("stage_b_gdino_score_adapter = True\n", encoding="utf-8")
    audit = {
        "schema": binding.P0_SCHEMA,
        "kind": "p0_checkpoint_audit",
        "baseline": binding._file_record(baseline.resolve()),
        "p0_checkpoint": binding._file_record(candidate.resolve()),
        "config": binding._file_record(config.resolve()),
        "functional_identity": {
            "rank_score_equals_base": True,
            "confidence_score_equals_base": True,
            "rank_residual_exact_zero": True,
            "confidence_gate_exact_zero": True,
        },
        "intended_use": "evaluation_only_same_records_parity",
    }
    sidecar = root / "p0.audit.json"
    _write_json(sidecar, audit)
    lineage = root / "trusted-lineage.json"
    _write_json(
        lineage,
        {
            "schema": binding.P0_SCHEMA,
            "kind": "p0_checkpoint_and_sidecar_verified",
            "checkpoint": binding._file_record(candidate.resolve()),
            "sidecar": binding._file_record(sidecar.resolve()),
            "audit": audit,
        },
    )
    return lineage


class FixedEvalSummaryBindingTest(unittest.TestCase):
    def _metric_input_fixture(self, root: Path):
        bindings = {}
        records_by_side = {}
        for side in ("baseline", "candidate"):
            ref_records = {}
            tn_records = {}
            for split in binding.REF_SPLITS:
                path = root / side / f"{split}.records.jsonl"
                _write_jsonl(path, [{"side": side, "split": split}])
                ref_records[split] = binding._file_record(path.resolve())
            for section in binding.TN_SECTIONS:
                path = root / side / f"{section}.records.jsonl"
                _write_jsonl(path, [{"side": side, "section": section}])
                tn_records[section] = binding._file_record(path.resolve())
            records_by_side[side] = {"ref": ref_records, "tn": tn_records}
            bindings[side] = {
                "schema": binding.SCHEMA,
                "kind": "completed_fixed_eval_summary_binding",
                "pass": True,
                "ref8": {
                    "splits": {
                        split: {"records": ref_records[split]}
                        for split in binding.REF_SPLITS
                    }
                },
                "tn": {
                    section: {
                        "records": tn_records[section],
                        "manifest_sha256": binding._file_record(
                            binding._resolve(binding.TN_SOURCE_MANIFESTS[section])
                        )["sha256"],
                        "source_manifest_sha256": binding._file_record(
                            binding._resolve(binding.TN_SOURCE_MANIFESTS[section])
                        )["sha256"],
                    }
                    for section in binding.TN_SECTIONS
                },
            }
        gates = {}
        fprs = {}
        for section in binding.TN_SECTIONS:
            gates[section] = {
                "schema": "stageb-dual-gate-v1",
                "input_files": {
                    side: [
                        *[
                            records_by_side[side]["ref"][split]
                            for split in binding.REF_SPLITS
                        ],
                        records_by_side[side]["tn"][section],
                    ]
                    for side in ("baseline", "candidate")
                }
            }
            gates[section]["input_files"][
                "identity_is_from_the_same_bytes_used_for_metrics"
            ] = True
            fprs[section] = {
                "schema": "stageb-fpr95-record-comparison-v1",
                "validation": {
                    "pass": True,
                    "baseline_manifest_binding_mode": "source_to_derived_v1",
                    "candidate_manifest_binding_mode": "source_to_derived_v1",
                    "manifest_path": binding.TN_SOURCE_MANIFESTS[section],
                    "manifest_sha256": binding._file_record(
                        binding._resolve(binding.TN_SOURCE_MANIFESTS[section])
                    )["sha256"],
                    "baseline_records": records_by_side["baseline"]["tn"][section][
                        "path"
                    ],
                    "candidate_records": records_by_side["candidate"]["tn"][section][
                        "path"
                    ],
                },
                "input_files": {
                    "manifest": binding._file_record(
                        binding._resolve(binding.TN_SOURCE_MANIFESTS[section])
                    ),
                    "baseline_records": records_by_side["baseline"]["tn"][section],
                    "candidate_records": records_by_side["candidate"]["tn"][section],
                    "identity_is_from_the_same_bytes_used_for_metrics": True,
                }
            }
        return bindings, fprs, gates, records_by_side

    def _fixture(self, root: Path):
        checkpoint = root / "weights" / "checkpoint0000.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixed checkpoint")
        config = root / "config.py"
        config.write_text("stage_b_gdino_score_adapter = False\n", encoding="utf-8")
        run_id = binding.checkpoint_run_prefix(checkpoint.resolve())
        eval_dir = root / "eval"
        runtime = {
            "batch_size": 16,
            "num_workers": 4,
            "seed": 42,
            "threshold_tprs": [0.75, 0.9, 0.95],
            "score_thresholds": [0.5],
        }
        preflight = {
            "schema": "stageb-fixed-protocol-v1",
            "kind": "fixed_stageb_eval_preflight",
            "checkpoint": binding._file_record(checkpoint.resolve()),
            "config": binding._file_record(config.resolve()),
            "runtime": runtime,
        }
        preflight_path = eval_dir / "protocol_eval_preflight.json"
        _write_json(preflight_path, preflight)

        ref_summary_rows = []
        ref_completion = {}
        for split_index, split in enumerate(binding.REF_SPLITS):
            manifest_path = (
                eval_dir
                / "ref8/refcoco_eval_inputs"
                / binding.REF_SPLIT_MANIFEST_FILES[split]
            )
            manifest_rows = _small_ref_manifest_rows(split)
            _write_jsonl(manifest_path, manifest_rows)
            record_path = (
                eval_dir
                / "ref8/per_example_records"
                / f"{run_id}__{split}.records.jsonl"
            )
            manifest_sha = binding._file_record(manifest_path.resolve())["sha256"]
            values = ((0.6, 0.8), (0.1, 0.4))
            rows = [
                {
                    "schema": RECORD_SCHEMA,
                    "task": "ref",
                    "manifest_key": f"ref:{split}",
                    "manifest_sha256": manifest_sha,
                    "manifest_n": len(values),
                    "manifest_index": index,
                    "sample_id": binding.sample_id_from_meta(
                        manifest_rows[index], task="ref", split=split, index=index
                    ),
                    "split": split,
                    "image_id": 1000 + index,
                    "ann_id": 2000 + index,
                    "ref_id": 3000 + index,
                    "sent_id": 4000 + index,
                    "run_id": run_id,
                    "valid": True,
                    "top1_iou": top1,
                    "all_query_best_iou": oracle,
                    "correct50": top1 >= 0.5,
                }
                for index, (top1, oracle) in enumerate(values)
            ]
            _write_jsonl(record_path, rows)
            ref_summary_rows.append(
                {
                    "run_id": run_id,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_name": checkpoint.name,
                    "dataset": split,
                    "seconds": 1.0,
                    "batch_size": runtime["batch_size"],
                    "num_workers": runtime["num_workers"],
                    "seed": runtime["seed"] + split_index * 100000,
                    "max_batches": 0,
                    "num_expressions": 2,
                    "valid_mask_expressions": 2,
                    "invalid_mask_expressions": 0,
                    "acc50": 0.5,
                    "acc25": 0.5,
                    "mean_iou": 0.35,
                    "recall50@all_queries": 0.5,
                    "recall25@all_queries": 1.0,
                    "mean_best_iou@all_queries": 0.6000000000000001,
                    "records_jsonl": str(record_path.resolve()),
                    "manifest_sha256": manifest_sha,
                    "manifest_n": 2,
                    "invalid_records": 0,
                }
            )
            record_file = binding._file_record(record_path.resolve())
            ref_completion[split] = {
                "path": record_file["path"],
                "sha256": record_file["sha256"],
            }
        ref_summary_path = eval_dir / "ref8/summary.json"
        _write_json(ref_summary_path, {"refcoco": ref_summary_rows, "tn": []})

        tn_completion = {}
        tn_summary_records = {}
        for section, count in SMALL_TN_SECTIONS.items():
            source_path = eval_dir / section / "locked_source_manifest.jsonl"
            source_rows = []
            for index in range(count):
                source_rows.append(
                    {
                        "sample_id": f"{section}:{index}",
                        "image_id": 5000 + index,
                        "ann_id": 6000 + index,
                        "ref_id": 7000 + index,
                        "sent_id": 8000 + index,
                        "instances": [
                            {
                                "pair_source": "refcoco+_unc",
                                "positive_phrase": "red object",
                                "raw_phrase": "blue object",
                            }
                        ],
                    }
                )
            _write_jsonl(source_path, source_rows)
            derived_path = (
                eval_dir
                / section
                / "tn_eval_inputs"
                / "tn_refcocop_val_refcocog_umd_val.jsonl"
            )
            derived_rows = []
            row_mapping = []
            for index, source_row in enumerate(source_rows):
                derived_row = copy.deepcopy(source_row)
                derived_row["instances"][0]["text_is_negative"] = True
                derived_row["tn_eval_split"] = "refcocop_val"
                derived_row["tn_eval_pair_source"] = "refcoco+_unc"
                derived_row["tn_eval_source_split"] = "val"
                derived_rows.append(derived_row)
                row_mapping.append(
                    {
                        "derived_index": index,
                        "source_index": index,
                        "sample_id": source_row["sample_id"],
                        "pair_source": "refcoco+_unc",
                        "source_split": "val",
                        "eval_split": "refcocop_val",
                    }
                )
            _write_jsonl(derived_path, derived_rows)
            write_tn_derived_manifest_binding(
                source_manifest_path=source_path,
                derived_manifest_path=derived_path,
                row_mapping=row_mapping,
                requested_splits=["refcocop_val", "refcocog_umd_val"],
                max_pairs=0,
                max_pairs_per_split=0,
                holdout_level="none",
            )
            manifest = load_eval_manifest(
                derived_path,
                task="tn",
                split="global",
                manifest_key="tn_global",
            )
            record_path = (
                eval_dir
                / section
                / "per_example_records"
                / f"{run_id}__tn_global.records.jsonl"
            )
            rows = []
            for index in range(count):
                rows.append(
                    make_eval_record(
                        manifest,
                        index=index,
                        run_id=run_id,
                        valid=True,
                        values={
                            "pos_score": 0.8 - index * 0.4,
                            "neg_score": 0.2 + index * 0.3,
                            "pos_iou": 0.7,
                            "neg_iou": 0.1,
                        },
                    )
                )
            _write_jsonl(record_path, rows)
            metrics = binding._tn_metrics(
                rows,
                threshold_tprs=runtime["threshold_tprs"],
                score_thresholds=runtime["score_thresholds"],
            )
            summary_row = {
                **metrics,
                "run_id": run_id,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_name": checkpoint.name,
                "seconds": 1.0,
                "batch_size": runtime["batch_size"],
                "num_workers": runtime["num_workers"],
                "seed": runtime["seed"],
                "max_batches": 0,
                "invalid_positive_pairs": 0,
                "invalid_negative_pairs": 0,
                "records_jsonl": str(record_path.resolve()),
                "manifest_sha256": manifest.sha256,
                "manifest_n": count,
                **tn_manifest_binding_summary_fields(manifest),
                "invalid_records": 0,
                "by_split": {},
                "by_category": {},
            }
            summary_path = eval_dir / section / "summary.json"
            _write_json(summary_path, {"refcoco": [], "tn": [summary_row]})
            record_file = binding._file_record(record_path.resolve())
            tn_completion[section] = {
                "path": record_file["path"],
                "sha256": record_file["sha256"],
                "manifest_binding": {
                    "schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
                    "algorithm": TN_DERIVATION_ALGORITHM,
                    "source_manifest": dict(manifest.tn_binding.source_manifest),
                    "derived_manifest": dict(manifest.tn_binding.derived_manifest),
                    "binding": binding._file_record(manifest.tn_binding.path),
                    "row_mapping_sha256": manifest.tn_binding.row_mapping_sha256,
                    "row_mapping_verified_against_source_and_derived": True,
                    "all_locked_source_rows_preserved_in_order": True,
                },
            }
            tn_summary_records[section] = binding._file_record(summary_path.resolve())

        completion = {
            "schema": "stageb-fixed-protocol-v1",
            "kind": "fixed_stageb_eval_complete",
            "preflight": binding._file_record(preflight_path.resolve()),
            "outputs": {
                "summaries": {
                    "ref8": binding._file_record(ref_summary_path.resolve()),
                    **tn_summary_records,
                },
                "ref_records": ref_completion,
                "tn_records": tn_completion,
            },
        }
        preflight["tn_manifest_derivation_contract"] = tn_manifest_derivation_contract()
        preflight["strict_manifests"] = {
            section: {
                **binding._file_record(
                    (eval_dir / section / "locked_source_manifest.jsonl").resolve()
                ),
                "rows": count,
            }
            for section, count in SMALL_TN_SECTIONS.items()
        }
        _write_json(preflight_path, preflight)
        completion["preflight"] = binding._file_record(preflight_path.resolve())
        _write_json(eval_dir / "protocol_eval_complete.json", completion)
        return eval_dir, checkpoint.resolve(), run_id

    def _audit(self, eval_dir, checkpoint):
        source_manifests = self._tn_sources(eval_dir)
        with patch.object(binding, "TN_SECTIONS", SMALL_TN_SECTIONS), patch.object(
            binding, "TN_SOURCE_MANIFESTS", source_manifests
        ), patch.object(binding, "REF_SPLIT_CONTRACT", _small_ref_contract()):
            return binding.audit_evaluation(eval_dir, checkpoint)

    def _tn_sources(self, eval_dir):
        return {
            section: str(
                (eval_dir / section / "locked_source_manifest.jsonl").resolve()
            )
            for section in SMALL_TN_SECTIONS
        }

    def _reseal_completion(self, eval_dir):
        completion_path = eval_dir / "protocol_eval_complete.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["preflight"] = binding._file_record(
            (eval_dir / "protocol_eval_preflight.json").resolve()
        )
        for section in ("ref8", *SMALL_TN_SECTIONS):
            completion["outputs"]["summaries"][section] = binding._file_record(
                (eval_dir / section / "summary.json").resolve()
            )
        for split in binding.REF_SPLITS:
            path = next(
                (eval_dir / "ref8/per_example_records").glob(f"*__{split}.records.jsonl")
            )
            record = binding._file_record(path.resolve())
            completion["outputs"]["ref_records"][split].update(
                path=record["path"], sha256=record["sha256"]
            )
        for section in SMALL_TN_SECTIONS:
            path = next((eval_dir / section / "per_example_records").glob("*.records.jsonl"))
            record = binding._file_record(path.resolve())
            completion["outputs"]["tn_records"][section].update(
                path=record["path"], sha256=record["sha256"]
            )
        _write_json(completion_path, completion)

    def test_happy_path_replays_top1_headroom_and_tn_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir, checkpoint, run_id = self._fixture(Path(temporary))
            result = self._audit(eval_dir, checkpoint)
            self.assertTrue(result["pass"])
            self.assertEqual(result["expected_run_id"], run_id)
            self.assertEqual(set(result["ref8"]["splits"]), set(binding.REF_SPLITS))
            self.assertEqual(
                result["ref8"]["splits"]["refcoco_val"]["metrics"][
                    "recall25@all_queries"
                ],
                1.0,
            )
            self.assertEqual(result["tn"]["strict2031"]["n"], 2)

    def test_production_ref8_contract_is_exact(self):
        expected = {
            "refcoco_val": (10834, "ac1ab43019a03dcc65ba3530469b6dcb2ac01be836b795ae5a3b1bdb56b6431d"),
            "refcoco_testA": (5657, "47278ef1043382235a151cd90d1e6c18c79d30bb71cb4eb7df1932abc622946e"),
            "refcoco_testB": (5095, "41687648194225a693da5c42c5448eb1a9f4d2f59ca4cd138d4063d818116c8f"),
            "refcocop_val": (10758, "1eef48a64e7c118b736aa6d383d164ff70af3504285a2cb43a34c02631b5f6de"),
            "refcocop_testA": (5726, "57a0fb2342f120d49a1174084a7748cb18ff75a7b789bf2ddaf6c8555dce1105"),
            "refcocop_testB": (4889, "49fe753d28a45cfb47f3d33cf5fbe34a1fda0ae111c7dcd24063c68e2b411d36"),
            "refcocog_val": (4896, "6a21fccf3d2330aaf72a3ee16cd1863f29470abc3ebfa64d098c04cf7d10e925"),
            "refcocog_test": (9602, "6c1c9bf2006344167bdce1859578faf83ca594383cc1acac62792c3e6a0f0a1d"),
        }
        self.assertEqual(
            {
                split: (value["rows"], value["sha256"])
                for split, value in binding.REF_SPLIT_CONTRACT.items()
            },
            expected,
        )
        self.assertEqual(
            binding.REF_SPLIT_MANIFEST_FILES,
            {
                "refcoco_val": "refcoco_unc_val.jsonl",
                "refcoco_testA": "refcoco_unc_testA.jsonl",
                "refcoco_testB": "refcoco_unc_testB.jsonl",
                "refcocop_val": "refcocoplus_unc_val.jsonl",
                "refcocop_testA": "refcocoplus_unc_testA.jsonl",
                "refcocop_testB": "refcocoplus_unc_testB.jsonl",
                "refcocog_val": "refcocog_umd_val.jsonl",
                "refcocog_test": "refcocog_umd_test.jsonl",
            },
        )

    def test_rejects_canonical_ref_manifest_filename_or_content_drift(self):
        for mutation in ("filename", "content"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                eval_dir, checkpoint, _ = self._fixture(Path(temporary))
                manifest = (
                    eval_dir
                    / "ref8/refcoco_eval_inputs"
                    / binding.REF_SPLIT_MANIFEST_FILES["refcoco_val"]
                )
                if mutation == "filename":
                    manifest.rename(manifest.with_name("wrong.jsonl"))
                else:
                    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
                    rows[0]["sent_id"] += 1
                    _write_jsonl(manifest, rows)
                with self.assertRaises(binding.SummaryBindingError):
                    self._audit(eval_dir, checkpoint)

    def test_rejects_ref_five_field_identity_drift_from_canonical_manifest(self):
        mutations = {
            "sample_id": lambda row: row.update(sample_id="wrong-but-unique"),
            "image_id": lambda row: row.update(image_id=999001),
            "ann_id": lambda row: row.update(ann_id=999002),
            "ref_id": lambda row: row.update(ref_id=999003),
            "sent_id": lambda row: row.update(sent_id=999004),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                eval_dir, checkpoint, _ = self._fixture(Path(temporary))
                records = next(
                    (eval_dir / "ref8/per_example_records").glob(
                        "*__refcoco_val.records.jsonl"
                    )
                )
                rows = [json.loads(line) for line in records.read_text().splitlines()]
                mutate(rows[0])
                _write_jsonl(records, rows)
                self._reseal_completion(eval_dir)
                with self.assertRaisesRegex(
                    binding.SummaryBindingError, "canonical manifest"
                ):
                    self._audit(eval_dir, checkpoint)

    def test_rejects_reduced_or_wrong_hash_ref_split_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir, checkpoint, _ = self._fixture(Path(temporary))
            with patch.object(
                binding, "TN_SECTIONS", SMALL_TN_SECTIONS
            ), patch.object(
                binding, "TN_SOURCE_MANIFESTS", self._tn_sources(eval_dir)
            ), self.assertRaisesRegex(
                binding.SummaryBindingError, "locked official split"
            ):
                binding.audit_evaluation(eval_dir, checkpoint)

        with tempfile.TemporaryDirectory() as temporary:
            eval_dir, checkpoint, _ = self._fixture(Path(temporary))
            contract = _small_ref_contract()
            contract["refcoco_val"] = {"rows": 2, "sha256": "f" * 64}
            with patch.object(binding, "TN_SECTIONS", SMALL_TN_SECTIONS), patch.object(
                binding, "TN_SOURCE_MANIFESTS", self._tn_sources(eval_dir)
            ), patch.object(
                binding, "REF_SPLIT_CONTRACT", contract
            ), self.assertRaisesRegex(
                binding.SummaryBindingError, "manifest SHA-256 mismatch"
            ):
                binding.audit_evaluation(eval_dir, checkpoint)

    def test_ref_accumulator_records_oracle_iou(self):
        row = {
            "image_id": 1,
            "ann_id": 2,
            "ref_id": 3,
            "sent_id": 4,
            "sample_id": "ref:1:2:3:4",
        }
        manifest = EvalManifest(
            path=Path("unused.jsonl"),
            task="ref",
            manifest_key="ref:refcoco_val",
            split="refcoco_val",
            sha256="a" * 64,
            rows=[row],
        )
        accumulator = text_eval.RefCocoTextAccumulator(
            [1], manifest=manifest, run_id="checkpoint"
        )
        accumulator.update(
            {
                "pred_logits": torch.tensor([[[5.0, -5.0], [0.0, -5.0]]]),
                "pred_boxes": torch.tensor(
                    [[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]]]
                ),
            },
            [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                    "phrase_to_token_mask": torch.tensor([[True, False]]),
                }
            ],
        )
        record = accumulator.eval_records[0]
        self.assertLess(record["top1_iou"], 0.5)
        self.assertAlmostEqual(record["all_query_best_iou"], 1.0, places=4)

    def test_rejects_ref_summary_metric_or_oracle_record_tampering(self):
        mutations = ("summary_acc50", "missing_oracle", "oracle_below_top1")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                eval_dir, checkpoint, _ = self._fixture(Path(temporary))
                if mutation == "summary_acc50":
                    path = eval_dir / "ref8/summary.json"
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    summary["refcoco"][0]["acc50"] = 0.0
                    _write_json(path, summary)
                else:
                    path = next(
                        (eval_dir / "ref8/per_example_records").glob(
                            "*__refcoco_val.records.jsonl"
                        )
                    )
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if mutation == "missing_oracle":
                        rows[0].pop("all_query_best_iou")
                    else:
                        rows[0]["all_query_best_iou"] = 0.1
                    _write_jsonl(path, rows)
                self._reseal_completion(eval_dir)
                with self.assertRaises(binding.SummaryBindingError):
                    self._audit(eval_dir, checkpoint)

    def test_rejects_ref_or_tn_record_run_id_tampering(self):
        locations = (
            eval_dir_path
            for eval_dir_path in (
                "ref8/per_example_records/*__refcoco_val.records.jsonl",
                "strict2031/per_example_records/*.records.jsonl",
            )
        )
        for pattern in locations:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as temporary:
                eval_dir, checkpoint, _ = self._fixture(Path(temporary))
                path = next(eval_dir.glob(pattern))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["run_id"] = "wrong_checkpoint"
                _write_jsonl(path, rows)
                self._reseal_completion(eval_dir)
                with self.assertRaisesRegex(binding.SummaryBindingError, "run_id mismatch"):
                    self._audit(eval_dir, checkpoint)

    def test_rejects_tn_float32_mean_or_median_summary_tampering(self):
        for metric in ("score_gap_mean", "score_gap_median"):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as temporary:
                eval_dir, checkpoint, _ = self._fixture(Path(temporary))
                summary_path = eval_dir / "strict2031/summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["tn"][0][metric] += 1e-7
                _write_json(summary_path, summary)
                self._reseal_completion(eval_dir)
                with self.assertRaisesRegex(
                    binding.SummaryBindingError, f"{metric} does not match records"
                ):
                    self._audit(eval_dir, checkpoint)

    def test_tn_source_derived_and_row_mapping_tampering_fail_closed(self):
        mutations = ("source", "derived", "sidecar_mapping", "record_mapping")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                eval_dir, checkpoint, _ = self._fixture(Path(temporary))
                section = eval_dir / "strict2031"
                if mutation == "source":
                    path = section / "locked_source_manifest.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["instances"][0]["positive_phrase"] = "tampered source"
                    _write_jsonl(path, rows)
                elif mutation == "derived":
                    path = (
                        section
                        / "tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl"
                    )
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["instances"][0]["raw_phrase"] = "tampered derived"
                    _write_jsonl(path, rows)
                elif mutation == "sidecar_mapping":
                    path = (
                        section
                        / "tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl.binding.json"
                    )
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["row_mapping"][0]["source_index"] = 1
                    _write_json(path, payload)
                else:
                    path = next(
                        (section / "per_example_records").glob("*.records.jsonl")
                    )
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["source_manifest_index"] = 1
                    _write_jsonl(path, rows)
                with self.assertRaises(binding.SummaryBindingError):
                    self._audit(eval_dir, checkpoint)

    def test_legacy_unbound_derived_fixed_artifact_requires_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir, checkpoint, _ = self._fixture(Path(temporary))
            preflight_path = eval_dir / "protocol_eval_preflight.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight.pop("tn_manifest_derivation_contract")
            _write_json(preflight_path, preflight)
            self._reseal_completion(eval_dir)
            with self.assertRaisesRegex(
                binding.SummaryBindingError, "legacy derived artifacts must be rerun"
            ):
                self._audit(eval_dir, checkpoint)

    def test_rejects_summary_checkpoint_and_checkpoint_content_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir, checkpoint, _ = self._fixture(Path(temporary))
            summary_path = eval_dir / "strict1607/summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["tn"][0]["checkpoint"] = str(checkpoint.parent / "other.pth")
            _write_json(summary_path, summary)
            self._reseal_completion(eval_dir)
            with self.assertRaisesRegex(binding.SummaryBindingError, "checkpoint does not match"):
                self._audit(eval_dir, checkpoint)

        with tempfile.TemporaryDirectory() as temporary:
            eval_dir, checkpoint, _ = self._fixture(Path(temporary))
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(binding.SummaryBindingError, "preflight"):
                self._audit(eval_dir, checkpoint)

    def test_trusted_lineage_binds_all_supported_schemas_to_same_baseline(self):
        schemas = (
            binding.P0_SCHEMA,
            binding.TWO_PHASE_SCHEMA,
            binding.TOTAL_TRUST_SCHEMA,
            binding.SEMANTIC_SCHEMA,
            binding.FIXED_TOP1_SCHEMA,
        )
        for schema in schemas:
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline = root / "baseline.pth"
                candidate = root / "candidate.pth"
                baseline.write_bytes(b"authoritative baseline")
                candidate.write_bytes(b"candidate")
                lineage = (
                    _trusted_p0_lineage(root, baseline, candidate)
                    if schema == binding.P0_SCHEMA
                    else _trusted_non_p0_lineage(root, schema, baseline, candidate)
                )
                with patch.object(
                    binding,
                    "verify_selection",
                    side_effect=_selection_replay_stub,
                ):
                    result = binding._audit_trusted_lineage(
                        lineage,
                        candidate_checkpoint=binding._file_record(candidate.resolve()),
                        expected_baseline_checkpoint=baseline,
                    )
                self.assertTrue(result["pass"])
                self.assertTrue(
                    result[
                        "eval_preflight_current_lineage_checkpoint_same_file_record"
                    ]
                )
                self.assertEqual(
                    result["root_authoritative_baseline_checkpoint"],
                    binding._file_record(baseline.resolve()),
                )

    def test_total_trust_accepts_historical_rank_only_at_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pth"
            rank_checkpoint = root / "rank.pth"
            candidate = root / "candidate.pth"
            baseline.write_bytes(b"authoritative baseline")
            rank_checkpoint.write_bytes(b"rank checkpoint")
            candidate.write_bytes(b"candidate")
            milestone = _total_confidence_milestone_with_historical_rank(
                root, baseline, rank_checkpoint, candidate
            )
            config = root / "candidate_config.py"
            config.write_text("stage_b_gdino_score_adapter = True\n", encoding="utf-8")
            lineage = root / "lineage.json"
            _write_json(
                lineage,
                {
                    "schema": binding.TOTAL_TRUST_SCHEMA,
                    "kind": "evaluation_checkpoint_verified",
                    "checkpoint": binding._file_record(candidate.resolve()),
                    "config": binding._file_record(config.resolve()),
                    "audit": binding._file_record(milestone.resolve()),
                },
            )
            result = binding._audit_trusted_lineage(
                lineage,
                candidate_checkpoint=binding._file_record(candidate.resolve()),
                expected_baseline_checkpoint=baseline,
            )
            self.assertTrue(result["pass"])
            self.assertEqual(
                result["root_authoritative_baseline_checkpoint"],
                binding._file_record(baseline.resolve()),
            )

    def test_total_trust_rejects_historical_schema_on_new_confidence_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pth"
            candidate = root / "candidate.pth"
            baseline.write_bytes(b"baseline")
            candidate.write_bytes(b"candidate")
            preflight = root / "historical-confidence.preflight.json"
            _write_json(
                preflight,
                {
                    "schema": binding.TWO_PHASE_SCHEMA,
                    "kind": "phase_preflight",
                    "phase": "confidence",
                    "initial_checkpoint": binding._file_record(baseline.resolve()),
                },
            )
            milestone = root / "historical-confidence.milestone.json"
            _write_json(
                milestone,
                {
                    "schema": binding.TWO_PHASE_SCHEMA,
                    "kind": "milestone_checkpoint",
                    "phase": "confidence",
                    "checkpoint": binding._file_record(candidate.resolve()),
                    "preflight": binding._file_record(preflight.resolve()),
                },
            )
            config = root / "candidate_config.py"
            config.write_text("stage_b_gdino_score_adapter = True\n", encoding="utf-8")
            lineage = root / "lineage.json"
            _write_json(
                lineage,
                {
                    "schema": binding.TOTAL_TRUST_SCHEMA,
                    "kind": "evaluation_checkpoint_verified",
                    "checkpoint": binding._file_record(candidate.resolve()),
                    "config": binding._file_record(config.resolve()),
                    "audit": binding._file_record(milestone.resolve()),
                },
            )
            with self.assertRaises(binding.SummaryBindingError):
                binding._audit_trusted_lineage(
                    lineage,
                    candidate_checkpoint=binding._file_record(candidate.resolve()),
                    expected_baseline_checkpoint=baseline,
                )

    def test_candidate_eval_config_must_match_all_supported_lineage_schemas(self):
        schemas = (
            binding.P0_SCHEMA,
            binding.TWO_PHASE_SCHEMA,
            binding.TOTAL_TRUST_SCHEMA,
            binding.SEMANTIC_SCHEMA,
            binding.FIXED_TOP1_SCHEMA,
        )
        for schema in schemas:
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                eval_dir, candidate, _ = self._fixture(root)
                baseline = root / "baseline.pth"
                baseline.write_bytes(b"authoritative baseline")
                preflight = json.loads(
                    (eval_dir / "protocol_eval_preflight.json").read_text(
                        encoding="utf-8"
                    )
                )
                eval_config = Path(preflight["config"]["path"])
                lineage = (
                    _trusted_p0_lineage(
                        root, baseline, candidate, config=eval_config
                    )
                    if schema == binding.P0_SCHEMA
                    else _trusted_non_p0_lineage(
                        root, schema, baseline, candidate, config=eval_config
                    )
                )
                with patch.object(
                    binding, "TN_SECTIONS", SMALL_TN_SECTIONS
                ), patch.object(
                    binding, "TN_SOURCE_MANIFESTS", self._tn_sources(eval_dir)
                ), patch.object(
                    binding, "REF_SPLIT_CONTRACT", _small_ref_contract()
                ), patch.object(
                    binding, "verify_selection", side_effect=_selection_replay_stub
                ):
                    result = binding.audit_evaluation(
                        eval_dir,
                        candidate,
                        trusted_lineage=lineage,
                        expected_baseline_checkpoint=baseline,
                    )
                self.assertEqual(
                    result["lineage_binding"]["evaluation_config"],
                    result["evaluation_config"],
                )

                wrong_config = root / "wrong_config.py"
                wrong_config.write_text("wrong = True\n", encoding="utf-8")
                payload = json.loads(lineage.read_text(encoding="utf-8"))
                if schema == binding.P0_SCHEMA:
                    payload["audit"]["config"] = binding._file_record(
                        wrong_config.resolve()
                    )
                    sidecar = Path(payload["sidecar"]["path"])
                    _write_json(sidecar, payload["audit"])
                    payload["sidecar"] = binding._file_record(sidecar.resolve())
                else:
                    payload["config"] = binding._file_record(wrong_config.resolve())
                _write_json(lineage, payload)
                with patch.object(
                    binding, "TN_SECTIONS", SMALL_TN_SECTIONS
                ), patch.object(
                    binding, "TN_SOURCE_MANIFESTS", self._tn_sources(eval_dir)
                ), patch.object(
                    binding, "REF_SPLIT_CONTRACT", _small_ref_contract()
                ), patch.object(
                    binding, "verify_selection", side_effect=_selection_replay_stub
                ), self.assertRaisesRegex(
                    binding.SummaryBindingError,
                    "lineage config does not exactly match",
                ):
                    binding.audit_evaluation(
                        eval_dir,
                        candidate,
                        trusted_lineage=lineage,
                        expected_baseline_checkpoint=baseline,
                    )

    def test_fixed_top1_lineage_requires_selection_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pth"
            candidate = root / "candidate.pth"
            baseline.write_bytes(b"authoritative baseline")
            candidate.write_bytes(b"candidate")
            lineage = _trusted_non_p0_lineage(
                root, binding.FIXED_TOP1_SCHEMA, baseline, candidate
            )
            payload = json.loads(lineage.read_text(encoding="utf-8"))
            payload.pop("selection_authorization")
            _write_json(lineage, payload)
            with self.assertRaisesRegex(
                binding.SummaryBindingError, "selection authorization"
            ):
                binding._audit_trusted_lineage(
                    lineage,
                    candidate_checkpoint=binding._file_record(candidate.resolve()),
                    expected_baseline_checkpoint=baseline,
                )

    def test_trusted_lineage_rejects_wrong_root_baseline_and_checkpoint_toctou(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pth"
            wrong_baseline = root / "wrong-baseline.pth"
            candidate = root / "candidate.pth"
            baseline.write_bytes(b"authoritative baseline")
            wrong_baseline.write_bytes(b"weaker alternate baseline")
            candidate.write_bytes(b"candidate before evaluation")
            lineage = _trusted_non_p0_lineage(
                root, binding.FIXED_TOP1_SCHEMA, baseline, candidate
            )
            with patch.object(
                binding, "verify_selection", side_effect=_selection_replay_stub
            ), self.assertRaisesRegex(
                binding.SummaryBindingError, "authoritative baseline checkpoint"
            ):
                binding._audit_trusted_lineage(
                    lineage,
                    candidate_checkpoint=binding._file_record(candidate.resolve()),
                    expected_baseline_checkpoint=wrong_baseline,
                )

            candidate.write_bytes(b"candidate replaced after lineage verification")
            with patch.object(
                binding, "verify_selection", side_effect=_selection_replay_stub
            ), self.assertRaisesRegex(binding.SummaryBindingError, "drifted"):
                binding._audit_trusted_lineage(
                    lineage,
                    candidate_checkpoint=binding._file_record(candidate.resolve()),
                    expected_baseline_checkpoint=baseline,
                )

    def test_audit_requires_lineage_and_expected_baseline_as_a_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir, checkpoint, _ = self._fixture(Path(temporary))
            with patch.object(binding, "TN_SECTIONS", SMALL_TN_SECTIONS), patch.object(
                binding, "REF_SPLIT_CONTRACT", _small_ref_contract()
            ), self.assertRaisesRegex(
                binding.SummaryBindingError, "must be provided together"
            ):
                binding.audit_evaluation(
                    eval_dir,
                    checkpoint,
                    trusted_lineage=Path(temporary) / "unused.json",
                )

    def test_final_metric_inputs_bind_gate_bytes_and_current_disk_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            bindings, fprs, gates, records = self._metric_input_fixture(
                Path(temporary)
            )
            result = binding.validate_final_metric_input_binding(
                baseline_binding=bindings["baseline"],
                candidate_binding=bindings["candidate"],
                fpr_reports=fprs,
                dual_gate_reports=gates,
            )
            self.assertTrue(result["pass"])
            self.assertEqual(result["record_files_per_side"], 10)

            drifted_gate = copy.deepcopy(gates)
            drifted_gate["strict2031"]["input_files"]["candidate"][0][
                "sha256"
            ] = "f" * 64
            with self.assertRaisesRegex(
                binding.SummaryBindingError, "dual gate candidate inputs differ"
            ):
                binding.validate_final_metric_input_binding(
                    baseline_binding=bindings["baseline"],
                    candidate_binding=bindings["candidate"],
                    fpr_reports=fprs,
                    dual_gate_reports=drifted_gate,
                )

            drifted_fpr = copy.deepcopy(fprs)
            drifted_fpr["strict1607"]["input_files"]["candidate_records"][
                "sha256"
            ] = "e" * 64
            with self.assertRaisesRegex(
                binding.SummaryBindingError, "FPR candidate input differs"
            ):
                binding.validate_final_metric_input_binding(
                    baseline_binding=bindings["baseline"],
                    candidate_binding=bindings["candidate"],
                    fpr_reports=drifted_fpr,
                    dual_gate_reports=gates,
                )

            current_path = Path(records["candidate"]["ref"]["refcoco_val"]["path"])
            current_path.write_bytes(b"changed after metric parsing\n")
            with self.assertRaisesRegex(
                binding.SummaryBindingError, "changed after it was parsed"
            ):
                binding.validate_final_metric_input_binding(
                    baseline_binding=bindings["baseline"],
                    candidate_binding=bindings["candidate"],
                    fpr_reports=fprs,
                    dual_gate_reports=gates,
                )

    def test_strict_final_writer_executes_with_bound_metric_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindings, fprs, gates, _ = self._metric_input_fixture(root / "records")
            baseline_checkpoint_path = root / "baseline.pth"
            candidate_checkpoint_path = root / "candidate.pth"
            baseline_checkpoint_path.write_bytes(b"baseline")
            candidate_checkpoint_path.write_bytes(b"candidate")
            baseline_checkpoint = binding._file_record(
                baseline_checkpoint_path.resolve()
            )
            candidate_checkpoint = binding._file_record(
                candidate_checkpoint_path.resolve()
            )
            trusted_lineage_path = root / "trusted-lineage.json"
            _write_json(trusted_lineage_path, {"sealed": True})
            trusted_lineage = binding._file_record(trusted_lineage_path.resolve())
            bindings["baseline"].update(
                checkpoint=baseline_checkpoint,
                lineage_binding=None,
                official_ref_contract={
                    "all_eight_exact_rows_and_manifest_sha256": True
                },
            )
            bindings["candidate"].update(
                checkpoint=candidate_checkpoint,
                lineage_binding={
                    "pass": True,
                    "root_authoritative_baseline_checkpoint": baseline_checkpoint,
                    "trusted_lineage_output": trusted_lineage,
                },
                official_ref_contract={
                    "all_eight_exact_rows_and_manifest_sha256": True
                },
            )
            for section in binding.TN_SECTIONS:
                fprs[section].update(
                    {
                        "global": {"candidate_minus_baseline_fpr95": -0.1},
                    }
                )
                gates[section].update(
                    validation={"pass": True},
                    gate={
                        "pass": True,
                        "global_fpr95_lower": True,
                        "every_required_ref_split_acc50_higher": True,
                    },
                    required_ref_splits=list(binding.REF_SPLITS),
                    refcoco={
                        split: {
                            "improved": True,
                            "candidate_minus_baseline_acc50": 0.01,
                        }
                        for split in binding.REF_SPLITS
                    },
                )
            evidence = [
                bindings["baseline"],
                bindings["candidate"],
                {
                    "schema": "stageb-lineage-pre-post-equality-v1",
                    "kind": "completed_lineage_pre_post_equality",
                    "pre_eval_lineage": trusted_lineage,
                    "post_eval_lineage": {
                        **trusted_lineage,
                        "path": str((root / "trusted-lineage.post.json").resolve()),
                    },
                    "same_bytes": True,
                    "same_json": True,
                    "pass": True,
                },
                {"kind": "fixed_stageb_paired_eval_protocol"},
                {"pass": True},
                fprs["strict2031"],
                fprs["strict1607"],
                gates["strict2031"],
                gates["strict1607"],
            ]
            evidence_names = (
                "baseline_summary_binding.json",
                "candidate_summary_binding.json",
                "lineage_replay_equality.json",
                "paired_protocol_audit.json",
                "paired_record_identity.json",
                "strict2031_fpr95_comparison.json",
                "strict1607_fpr95_comparison.json",
                "primary_strict2031.json",
                "supplemental_strict1607.json",
            )
            evidence_paths = []
            for name, value in zip(evidence_names, evidence):
                path = root / name
                _write_json(path, value)
                evidence_paths.append(path)

            wrapper = (
                Path(__file__).resolve().parents[1]
                / "tools/run_stageb_gdino_adapter_probe_eval.sh"
            ).read_text(encoding="utf-8")
            function_start = wrapper.index("write_final_acceptance_status()")
            body_start = wrapper.index("<<'PY'\n", function_start) + len("<<'PY'\n")
            body_end = wrapper.index("\nPY\n}", body_start)
            output = root / "final_acceptance_status.json"
            completed = subprocess.run(
                [
                    "/usr/bin/python3",
                    "-",
                    str(output),
                    *[str(path) for path in evidence_paths],
                ],
                input=wrapper[body_start:body_end],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            final = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(final["final_acceptance_claimed"])
            self.assertTrue(final["metric_input_binding"]["pass"])

    def test_atomic_output_writer_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            binding._write_json(output, {"schema": binding.SCHEMA, "pass": True})
            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
