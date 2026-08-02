import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import tools.stageb_fixed_protocol_audit as protocol
from tools.stageb_dependency_audit import local_python_dependency_paths
from tools.stageb_eval_records import (
    load_eval_manifest,
    make_eval_record,
    tn_manifest_binding_summary_fields,
    tn_manifest_derivation_contract,
    write_tn_derived_manifest_binding,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(*, task, key, manifest_hash, index, n, sample_id, image_id):
    row = {
        "schema": "stageb-eval-record-v1",
        "task": task,
        "manifest_key": key,
        "manifest_sha256": manifest_hash,
        "manifest_n": n,
        "manifest_index": index,
        "sample_id": sample_id,
        "split": "refcoco_val",
        "image_id": image_id,
        "ann_id": 100 + index,
        "ref_id": 200 + index,
        "sent_id": 300 + index,
        "valid": True,
    }
    if task == "tn":
        row.update({"pos_score": 0.7, "neg_score": 0.2})
    else:
        row.update({"top1_iou": 0.75, "correct50": True})
    return row


class StageBFixedProtocolCompletionVerifyTest(unittest.TestCase):
    def test_eval_code_closure_includes_adapter_dataset_and_transformer(self):
        root = Path(__file__).resolve().parents[1]
        paths = local_python_dependency_paths(
            protocol.EVAL_CODE_ENTRIES,
            root=root,
            include=protocol.EVAL_CODE_INCLUDE,
        )
        relative = {path.relative_to(root).as_posix() for path in paths}
        self.assertIn("models/GroundingDINO/stage_b_gdino_score_adapter.py", relative)
        self.assertIn("models/GroundingDINO/transformer.py", relative)
        self.assertIn("datasets/patch_episode.py", relative)

    def test_acceptance_closure_includes_final_gate_and_record_comparators(self):
        root = Path(__file__).resolve().parents[1]
        relative = {
            Path(row["path"]).relative_to(root).as_posix()
            for row in protocol._acceptance_code_records()
        }
        self.assertIn("tools/make_stageb_gdino_adapter_p0.py", relative)
        self.assertIn("tools/verify_stageb_dual_gate.py", relative)
        self.assertIn("tools/verify_stageb_p0_record_parity.py", relative)
        self.assertIn("tools/compare_stageb_fpr95_records.py", relative)
        self.assertIn("tools/stageb_eval_records.py", relative)
        orchestration = {
            Path(row["path"]).relative_to(root).as_posix()
            for row in protocol._acceptance_orchestration_records()
        }
        self.assertEqual(
            orchestration,
            {
                "tools/run_stageb_gdino_adapter_probe_eval.sh",
                "tools/run_stageb_fixed_dual_gate.sh",
            },
        )

    def test_train_code_closure_is_recursive_and_includes_adapter_path(self):
        root = Path(__file__).resolve().parents[1]
        paths = local_python_dependency_paths(
            protocol.TRAIN_CODE_ENTRIES,
            root=root,
            include=protocol.TRAIN_CODE_INCLUDE,
        )
        relative = {path.relative_to(root).as_posix() for path in paths}
        self.assertIn("main.py", relative)
        self.assertIn("engine.py", relative)
        self.assertIn("datasets/odvg.py", relative)
        self.assertIn("models/GroundingDINO/stage_b_gdino_score_adapter.py", relative)
        self.assertIn("models/GroundingDINO/transformer.py", relative)

    def test_resolved_adapter_protocol_records_p3_confidence_contract(self):
        config = protocol._path(
            "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py"
        )
        model_protocol = protocol._resolved_eval_contract(
            protocol._load_cfg(config)
        )["model_protocol"]
        expected = {
            "stage_b_gdino_gate_pool_temperature": 0.01,
            "stage_b_gdino_gate_topk": 3,
            "stage_b_gdino_confidence_objective": "detached_recent_q05_trust",
            "stage_b_gdino_fpr_temperature": 0.1,
            "stage_b_gdino_fpr_margin": 0.0,
            "stage_b_gdino_paired_margin_weight": 0.25,
            "stage_b_gdino_paired_margin": 0.05,
            "stage_b_gdino_positive_trust_margin": 0.02,
            "stage_b_gdino_positive_trust_weight": 1.0,
            "stage_b_gdino_queue_size": 512,
            "stage_b_gdino_queue_min_count": 256,
        }
        for key, value in expected.items():
            self.assertEqual(model_protocol[key], value)

    def _fixture(self, root):
        output = root / "eval"
        for section in ("ref8", "strict2031", "strict1607"):
            directory = output / section
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(
                json.dumps({"section": section}), encoding="utf-8"
            )

        ref_hash = "a" * 64
        _write_jsonl(
            output / "ref8/per_example_records/ref.records.jsonl",
            [
                _record(
                    task="ref",
                    key="ref:refcoco_val",
                    manifest_hash=ref_hash,
                    index=0,
                    n=1,
                    sample_id="ref:0",
                    image_id=10,
                )
            ],
        )

        specs = {}
        for label, count in (("strict2031", 2), ("strict1607", 1)):
            manifest = root / f"{label}.manifest.jsonl"
            manifest_rows = [
                {
                    "sample_id": f"sample:{index}",
                    "image_id": 20 + index,
                    "ann_id": 100 + index,
                    "ref_id": 200 + index,
                    "sent_id": 300 + index,
                    "manifest_schema": "stageb_vlm_verified_strict_tn_v2",
                    "visual_verified_negative": True,
                    "coverage_pass": True,
                    "tn_eval_pair_source": "refcoco+_unc",
                    "instances": [
                        {
                            "pair_source": "refcoco+_unc",
                            "positive_phrase": "red object",
                            "raw_phrase": "blue object",
                        }
                    ],
                }
                for index in range(count)
            ]
            _write_jsonl(manifest, manifest_rows)
            manifest_hash = protocol._sha256(manifest)
            specs[label] = {
                "path": str(manifest),
                "rows": count,
                "sha256": manifest_hash,
                "source_counts": {"refcoco+_unc": count},
            }
            derived = (
                output
                / label
                / "tn_eval_inputs"
                / "tn_refcocop_val_refcocog_umd_val.jsonl"
            )
            derived_rows = []
            mapping = []
            for index, source_row in enumerate(manifest_rows):
                derived_row = json.loads(json.dumps(source_row))
                derived_row["instances"][0]["text_is_negative"] = True
                derived_row["tn_eval_split"] = "refcocop_val"
                derived_row["tn_eval_pair_source"] = "refcoco+_unc"
                derived_row["tn_eval_source_split"] = "val"
                derived_rows.append(derived_row)
                mapping.append(
                    {
                        "derived_index": index,
                        "source_index": index,
                        "sample_id": source_row["sample_id"],
                        "pair_source": "refcoco+_unc",
                        "source_split": "val",
                        "eval_split": "refcocop_val",
                    }
                )
            _write_jsonl(derived, derived_rows)
            write_tn_derived_manifest_binding(
                source_manifest_path=manifest,
                derived_manifest_path=derived,
                row_mapping=mapping,
                requested_splits=["refcocop_val", "refcocog_umd_val"],
                max_pairs=0,
                max_pairs_per_split=0,
                holdout_level="none",
            )
            eval_manifest = load_eval_manifest(
                derived,
                task="tn",
                split="global",
                manifest_key="tn_global",
            )
            record_path = output / label / "per_example_records/tn.records.jsonl"
            _write_jsonl(
                record_path,
                [
                    make_eval_record(
                        eval_manifest,
                        index=index,
                        run_id="candidate",
                        valid=True,
                        values={"pos_score": 0.7, "neg_score": 0.2},
                    )
                    for index in range(count)
                ],
            )
            protocol._write_json(
                output / label / "summary.json",
                {
                    "refcoco": [],
                    "tn": [
                        {
                            "manifest_n": count,
                            "manifest_sha256": eval_manifest.sha256,
                            **tn_manifest_binding_summary_fields(eval_manifest),
                            "records_jsonl": str(record_path.resolve()),
                        }
                    ],
                },
            )

        preflight = output / "protocol_eval_preflight.json"
        config = protocol._path(
            "config/ablations/cfg_stageb_gdino_score_adapter_dataft.py"
        )
        checkpoint = root / "candidate.pth"
        checkpoint.write_bytes(b"candidate checkpoint")
        resolved = protocol._resolved_eval_contract(protocol._load_cfg(config))
        protocol._write_json(
            preflight,
            {
                "schema": "stageb-fixed-protocol-v1",
                "kind": "fixed_stageb_eval_preflight",
                "config": protocol._file_record(config),
                "config_import_chain": protocol._config_import_records(config),
                "checkpoint": protocol._file_record(checkpoint),
                "baseline_train_complete": None,
                **resolved,
                "runtime": {
                    "device": "cpu",
                    "batch_size": 1,
                    "num_workers": 0,
                    "seed": 42,
                    "amp": False,
                    "topk": [1],
                    "threshold_tprs": [0.75, 0.9, 0.95],
                    "score_thresholds": [0.5],
                    "tn_splits": ["refcocop_val", "refcocog_umd_val"],
                    "ref_splits": ["refcoco_val"],
                },
                "code": protocol._dependency_records(
                    protocol.EVAL_CODE_ENTRIES,
                    include=protocol.EVAL_CODE_INCLUDE,
                ),
                "orchestration": [
                    protocol._file_record(protocol._path(path))
                    for path in protocol.EVAL_ORCHESTRATION
                ],
                "acceptance_code": protocol._acceptance_code_records(),
                "acceptance_orchestration": (
                    protocol._acceptance_orchestration_records()
                ),
                "strict_manifests": {
                    label: {
                        **spec,
                        "size_bytes": Path(spec["path"]).stat().st_size,
                    }
                    for label, spec in specs.items()
                },
                "tn_manifest_derivation_contract": tn_manifest_derivation_contract(),
            },
        )
        return output, specs

    def _seal(self, output):
        complete = output / "protocol_eval_complete.json"
        protocol._write_json(
            complete,
            {
                "schema": "stageb-fixed-protocol-v1",
                "kind": "fixed_stageb_eval_complete",
                "preflight": protocol._file_record(
                    output / "protocol_eval_preflight.json"
                ),
                "outputs": protocol._audit_eval_outputs(output),
            },
        )

    def test_verify_recomputes_summary_and_record_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, specs = self._fixture(Path(temporary))
            with (
                patch.object(protocol, "REF_SPLITS", ["refcoco_val"]),
                patch.object(protocol, "STRICT_MANIFESTS", specs),
            ):
                self._seal(output)
                protocol._verify_eval_completion(output)

                summary = output / "ref8/summary.json"
                original_summary = summary.read_bytes()
                summary.write_text(json.dumps({"tampered": True}), encoding="utf-8")
                with self.assertRaisesRegex(protocol.ProtocolError, "outputs changed"):
                    protocol._verify_eval_completion(output)
                summary.write_bytes(original_summary)

                record = output / "strict2031/per_example_records/tn.records.jsonl"
                rows = [json.loads(line) for line in record.read_text().splitlines()]
                rows[0]["neg_score"] = 0.9
                _write_jsonl(record, rows)
                with self.assertRaisesRegex(protocol.ProtocolError, "outputs changed"):
                    protocol._verify_eval_completion(output)

    def test_verify_rejects_invalid_or_reordered_records_before_hash_compare(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, specs = self._fixture(Path(temporary))
            with (
                patch.object(protocol, "REF_SPLITS", ["refcoco_val"]),
                patch.object(protocol, "STRICT_MANIFESTS", specs),
            ):
                self._seal(output)
                record = output / "strict2031/per_example_records/tn.records.jsonl"
                rows = [json.loads(line) for line in record.read_text().splitlines()]
                rows[0]["valid"] = False
                _write_jsonl(record, rows)
                with self.assertRaisesRegex(protocol.ProtocolError, "invalid records"):
                    protocol._verify_eval_completion(output)

    def test_postflight_rejects_source_derived_and_mapping_tampering(self):
        for mutation in ("source", "derived", "mapping"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                output, specs = self._fixture(Path(temporary))
                with (
                    patch.object(protocol, "REF_SPLITS", ["refcoco_val"]),
                    patch.object(protocol, "STRICT_MANIFESTS", specs),
                ):
                    self._seal(output)
                    if mutation == "source":
                        path = Path(specs["strict2031"]["path"])
                        rows = [json.loads(line) for line in path.read_text().splitlines()]
                        rows[0]["instances"][0]["positive_phrase"] = "tampered"
                        _write_jsonl(path, rows)
                    elif mutation == "derived":
                        path = (
                            output
                            / "strict2031/tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl"
                        )
                        rows = [json.loads(line) for line in path.read_text().splitlines()]
                        rows[0]["instances"][0]["raw_phrase"] = "tampered"
                        _write_jsonl(path, rows)
                    else:
                        path = (
                            output
                            / "strict2031/tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl.binding.json"
                        )
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        payload["row_mapping"][0]["source_index"] = 1
                        protocol._write_json(path, payload)
                    with self.assertRaises(protocol.ProtocolError):
                        protocol._verify_eval_completion(output)

    def test_verify_rejects_a_changed_config_import_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, specs = self._fixture(Path(temporary))
            with (
                patch.object(protocol, "REF_SPLITS", ["refcoco_val"]),
                patch.object(protocol, "STRICT_MANIFESTS", specs),
            ):
                preflight_path = output / "protocol_eval_preflight.json"
                preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
                preflight["config_import_chain"][0]["sha256"] = "0" * 64
                protocol._write_json(preflight_path, preflight)
                self._seal(output)
                with self.assertRaisesRegex(protocol.ProtocolError, "import chain changed"):
                    protocol._verify_eval_completion(output)

    def test_verify_rejects_incomplete_code_or_resolved_model_records(self):
        for field in ("code", "orchestration", "resolved_model_architecture"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                output, specs = self._fixture(Path(temporary))
                with (
                    patch.object(protocol, "REF_SPLITS", ["refcoco_val"]),
                    patch.object(protocol, "STRICT_MANIFESTS", specs),
                ):
                    preflight_path = output / "protocol_eval_preflight.json"
                    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
                    if field == "code":
                        preflight["code"].pop()
                        expected = "dependency closure changed"
                    elif field == "orchestration":
                        preflight["orchestration"].pop()
                        expected = "orchestration changed"
                    else:
                        preflight["resolved_model_architecture"]["hidden_dim"] = 999
                        expected = "resolved configuration drifted"
                    protocol._write_json(preflight_path, preflight)
                    self._seal(output)
                    with self.assertRaisesRegex(protocol.ProtocolError, expected):
                        protocol._verify_eval_completion(output)

    def test_acceptance_script_drift_rejects_reuse_and_compare(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "acceptance.sh"
            script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            with patch.object(
                protocol,
                "ACCEPTANCE_ORCHESTRATION",
                (str(script),),
            ):
                output, specs = self._fixture(root)
                with (
                    patch.object(protocol, "REF_SPLITS", ["refcoco_val"]),
                    patch.object(protocol, "STRICT_MANIFESTS", specs),
                ):
                    self._seal(output)
                    protocol._verify_eval_completion(output)
                    script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        protocol.ProtocolError, "acceptance orchestration changed"
                    ):
                        protocol._verify_eval_completion(output)
                    with self.assertRaisesRegex(
                        protocol.ProtocolError, "acceptance orchestration changed"
                    ):
                        protocol._cmd_compare_evals(
                            Namespace(
                                baseline_dir=str(output),
                                candidate_dir=str(output),
                                output=str(root / "paired.json"),
                            )
                        )

    def test_p0_builder_drift_rejects_reuse_and_compare(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder = root / "make_stageb_gdino_adapter_p0.py"
            builder.write_text("VERSION = 1\n", encoding="utf-8")

            def acceptance_snapshot():
                return [protocol._file_record(builder)]

            with patch.object(
                protocol,
                "_acceptance_code_records",
                side_effect=acceptance_snapshot,
            ):
                output, specs = self._fixture(root)
                with (
                    patch.object(protocol, "REF_SPLITS", ["refcoco_val"]),
                    patch.object(protocol, "STRICT_MANIFESTS", specs),
                ):
                    self._seal(output)
                    protocol._verify_eval_completion(output)
                    builder.write_text("VERSION = 2\n", encoding="utf-8")
                    with self.assertRaisesRegex(
                        protocol.ProtocolError, "acceptance code dependency closure"
                    ):
                        protocol._verify_eval_completion(output)
                    with self.assertRaisesRegex(
                        protocol.ProtocolError, "acceptance code dependency closure"
                    ):
                        protocol._cmd_compare_evals(
                            Namespace(
                                baseline_dir=str(output),
                                candidate_dir=str(output),
                                output=str(root / "paired.json"),
                            )
                        )

    def test_shared_parent_config_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = str((root / "parent.py").resolve())
            baseline_leaf = str((root / "baseline.py").resolve())
            candidate_leaf = str((root / "candidate.py").resolve())
            baseline = {
                "config": {"path": baseline_leaf},
                "config_import_chain": [
                    {"path": baseline_leaf, "sha256": "1" * 64},
                    {"path": parent, "sha256": "a" * 64},
                ],
            }
            candidate = {
                "config": {"path": candidate_leaf},
                "config_import_chain": [
                    {"path": candidate_leaf, "sha256": "2" * 64},
                    {"path": parent, "sha256": "b" * 64},
                ],
            }
            with self.assertRaisesRegex(protocol.ProtocolError, "parent config hash differs"):
                protocol._shared_config_parent_hashes(baseline, candidate)

    def test_train_postflight_replay_detects_dependency_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight_path = root / "protocol_train_preflight.json"
            recorded = {
                "schema": "stageb-fixed-protocol-v1",
                "kind": "fixed_pure_stageb_dataft_train_preflight",
                "initial_checkpoint": {"path": str(root / "stagea.pth")},
                "evaluation_data": {"root": str(root / "data")},
                "launch": {
                    "world_size": 2,
                    "per_gpu_batch": 9,
                    "allow_nonfinal_stagea_name": False,
                },
                "code": [{"path": "main.py", "sha256": "a" * 64}],
            }
            protocol._write_json(preflight_path, recorded)
            replayed = dict(recorded)
            replayed["code"] = [{"path": "main.py", "sha256": "b" * 64}]
            with patch.object(
                protocol,
                "_make_train_preflight_payload",
                return_value=replayed,
            ):
                with self.assertRaisesRegex(protocol.ProtocolError, "no longer matches"):
                    protocol._verify_train_preflight(preflight_path)


if __name__ == "__main__":
    unittest.main()
