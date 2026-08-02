import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import build_stageb_u2_final_receipt as final


class BuildStageBU2FinalReceiptTest(unittest.TestCase):
    def _write_json(self, path: Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )

    def _record(self, path: Path):
        raw = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def _seal(self, payload, field):
        result = dict(payload)
        result[field] = final.canonical_json_sha256(result)
        return result

    def _record_file(self, root: Path, name: str):
        path = root / "records" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        return path

    def _fixture(self, root: Path):
        splits = tuple(final.REF_SPLITS)
        seeds = {split: 42 + 100000 * index for index, split in enumerate(splits)}
        contract = {
            split: {
                "rows": 2,
                "sha256": hashlib.sha256(split.encode("ascii")).hexdigest(),
            }
            for split in splits
        }
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"one physical U2 checkpoint\n")
        checkpoint_record = self._record(checkpoint)
        sweep_config = root / "sweep_config.py"
        sweep_config.write_text("stage_b_u0_category_gate_max_gap = 1.0\n", encoding="ascii")
        sweep_config_record = self._record(sweep_config)

        baseline_records = {
            split: self._record_file(root, f"baseline_{split}") for split in splits
        }
        candidate_records = {
            split: self._record_file(root, f"candidate_{split}") for split in splits
        }

        def ref_row(split, acc50, record_path, *, candidate=False, config_record=None):
            row = {
                "dataset": split,
                "seed": seeds[split],
                **final.FORMAL_RUNTIME,
                "manifest_n": contract[split]["rows"],
                "manifest_sha256": contract[split]["sha256"],
                "num_expressions": contract[split]["rows"],
                "valid_mask_expressions": contract[split]["rows"],
                "invalid_records": 0,
                "invalid_mask_expressions": 0,
                "acc50": acc50,
                "run_id": "u2" if candidate else "b58",
                "records_jsonl": str(record_path),
            }
            if candidate:
                row.update(
                    {
                        "amp": True,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_record["sha256"],
                        "config": config_record["path"],
                        "config_sha256": config_record["sha256"],
                    }
                )
            return row

        baseline_ref8 = root / "baseline_ref8.json"
        baseline_rows = [
            ref_row(split, 0.5, baseline_records[split]) for split in splits
        ]
        self._write_json(baseline_ref8, {"refcoco": baseline_rows, "tn": []})

        exact_gaps = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
        sweep_rows = []
        for split in final.VAL_SPLITS:
            for gap in exact_gaps:
                record_path = self._record_file(root, f"sweep_{split}_{gap:g}")
                row = ref_row(
                    split,
                    0.6 if gap == 3.0 else 0.55,
                    record_path,
                    candidate=True,
                    config_record=sweep_config_record,
                )
                row.update(
                    {
                        "category_gate_max_gap": gap,
                        "category_gate_sweep_contract": final.SWEEP_CONTRACT,
                        "category_gate_single_forward_gap_count": len(exact_gaps),
                    }
                )
                sweep_rows.append(row)
        sweep_summary = root / "selection_sweep.json"
        self._write_json(sweep_summary, {"refcoco": sweep_rows, "tn": []})

        policy = {"schema": "selection-policy/v1", "objective": "fixed"}
        selection_body = {
            "schema": final.SELECTION_SCHEMA,
            "selection_frozen": True,
            "selection": {"max_gap": 3.0},
            "contract": {
                "evaluation_runtime": final.FORMAL_RUNTIME,
                "canonical_seeds": {split: seeds[split] for split in final.VAL_SPLITS},
                "val_splits": list(final.VAL_SPLITS),
                "sweep_contract": final.SWEEP_CONTRACT,
                "exact_gaps": list(exact_gaps),
                "expected_query_count": 900,
            },
            "selection_policy": policy,
            "selection_policy_sha256": final.canonical_json_sha256(policy),
            "inputs": {
                "sweep_summary": self._record(sweep_summary),
                "baseline_summary": self._record(baseline_ref8),
            },
        }
        selection_receipt = root / "selection_receipt.json"
        selection_payload = self._seal(selection_body, "payload_sha256")
        self._write_json(selection_receipt, selection_payload)

        config = root / "gap3_config.py"
        config.write_text(
            "stage_b_u0_category_preserving_patch_gate = True\n"
            "stage_b_u0_patch_rank = True\n"
            "stage_b_gdino_score_adapter = True\n"
            "enable_patch_branch = True\n"
            "stage_b_u2_category_complete_supervision = True\n"
            "stage_b_u0_category_gate_max_gap = 3.0\n"
            f"stage_b_u0_category_gate_selection_receipt = {str(selection_receipt)!r}\n"
            "stage_b_u0_category_gate_selection_payload_sha256 = "
            f"{selection_payload['payload_sha256']!r}\n",
            encoding="ascii",
        )
        config_record = self._record(config)

        final_val3 = root / "final_val3.json"
        self._write_json(
            final_val3,
            {
                "refcoco": [
                    ref_row(
                        split,
                        0.6,
                        candidate_records[split],
                        candidate=True,
                        config_record=config_record,
                    )
                    for split in final.VAL_SPLITS
                ],
                "tn": [],
            },
        )
        test5 = root / "test5.json"
        self._write_json(
            test5,
            {
                "refcoco": [
                    ref_row(
                        split,
                        0.6,
                        candidate_records[split],
                        candidate=True,
                        config_record=config_record,
                    )
                    for split in final.TEST_SPLITS
                ],
                "tn": [],
            },
        )

        training_body = {
            "schema": final.TRAINING_SCHEMA,
            "checkpoint": {
                "file": checkpoint_record,
                "optimizer_updates": 100,
                "args": {
                    "batch_size": 56,
                    "seed": 42,
                    "max_train_iters": 100,
                    "stage_b_u0_patch_rank": True,
                    "stage_b_u2_category_complete_supervision": True,
                    "stage_b_gdino_score_adapter": True,
                },
            },
            "lineage": {"durable_checkpoint": checkpoint_record},
            "invariants": {
                "formal_checkpoint_sha256_exact": True,
                "effective_args_equal_checkpoint_args": True,
                "frozen_tensor_hash_equal_initializer_to_u100": True,
                "merged_r100_p50_teacher_frozen": True,
                "shared_patch_backbone_frozen": True,
                "transition_audit_recomputed_equal": True,
                "category_complete_data_receipt_replayed": True,
            },
        }
        training_receipt = root / "training_receipt.json"
        self._write_json(
            training_receipt, self._seal(training_body, "receipt_sha256")
        )

        strict_paths = {}
        strict_rows = {}
        for n in final.STRICT_SIZES:
            manifest_sha = hashlib.sha256(f"tn-{n}".encode("ascii")).hexdigest()
            candidate_record = self._record_file(root, f"candidate_tn_{n}")
            baseline_record = self._record_file(root, f"baseline_tn_{n}")

            def tn_row(record_path, fpr, candidate=False):
                row = {
                    "seed": 42,
                    **final.FORMAL_RUNTIME,
                    "manifest_n": n,
                    "manifest_sha256": manifest_sha,
                    "num_pairs": n,
                    "invalid_records": 0,
                    "run_id": "u2" if candidate else "b58",
                    "records_jsonl": str(record_path),
                    "fpr95tpr": fpr,
                    "actual_tpr_at_95tpr": 0.951,
                }
                if candidate:
                    row.update(
                        {
                            "amp": True,
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_record["sha256"],
                            "config": str(config),
                            "config_sha256": config_record["sha256"],
                        }
                    )
                return row

            candidate_row = tn_row(candidate_record, 0.4, candidate=True)
            baseline_row = tn_row(baseline_record, 0.5)
            candidate_summary = root / f"candidate_strict{n}.json"
            baseline_summary = root / f"baseline_strict{n}.json"
            self._write_json(candidate_summary, {"refcoco": [], "tn": [candidate_row]})
            self._write_json(baseline_summary, {"refcoco": [], "tn": [baseline_row]})
            strict_paths[n] = {
                "candidate_summary": candidate_summary,
                "baseline_summary": baseline_summary,
                "candidate_record": candidate_record,
                "baseline_record": baseline_record,
            }
            strict_rows[n] = (candidate_row, baseline_row)

        def validation_group(n, manifest_sha, *, ref=False):
            side = {
                "duplicates": 0,
                "invalid": 0,
                "exact_manifest_index_order": True,
                "manifest_n": n,
                "manifest_sha256": manifest_sha,
                "records_n": n,
            }
            return {
                "baseline": dict(side),
                "candidate": dict(side),
                "paired": {
                    "n_match": True,
                    "manifest_hash_match": True,
                    "sample_id_order_match": True,
                    "image_id_order_match": True,
                    "all_query_best_iou_exact_match": True if ref else None,
                },
            }

        record_gates = {}
        for n in final.STRICT_SIZES:
            candidate_row, baseline_row = strict_rows[n]
            groups = {
                f"ref:{split}": validation_group(
                    contract[split]["rows"], contract[split]["sha256"], ref=True
                )
                for split in splits
            }
            groups["tn_global"] = validation_group(
                n, candidate_row["manifest_sha256"]
            )
            ref_report = {
                split: {
                    "n": contract[split]["rows"],
                    "baseline_acc50": 0.5,
                    "candidate_acc50": 0.6,
                    "improved": True,
                }
                for split in splits
            }
            report = {
                "schema": final.RECORD_GATE_SCHEMA,
                "target_tpr": 0.95,
                "required_ref_splits": list(splits),
                "evaluated_ref_splits": list(reversed(splits)),
                "gate": {
                    "pass": True,
                    "every_required_ref_split_acc50_higher": True,
                    "global_fpr95_lower": True,
                    "bootstrap_ci_is_informational_not_a_gate": True,
                },
                "validation": {"pass": True, "errors": [], "groups": groups},
                "input_files": {
                    "identity_is_from_the_same_bytes_used_for_metrics": True,
                    "baseline": [
                        self._record(baseline_records[split]) for split in splits
                    ]
                    + [self._record(strict_paths[n]["baseline_record"])],
                    "candidate": [
                        self._record(candidate_records[split]) for split in splits
                    ]
                    + [self._record(strict_paths[n]["candidate_record"])],
                },
                "refcoco": ref_report,
                "tn_global": {
                    "n": n,
                    "improved": True,
                    "baseline": {"fpr": 0.5, "actual_tpr": 0.951},
                    "candidate": {"fpr": 0.4, "actual_tpr": 0.951},
                },
            }
            report_path = root / f"strict{n}_record_gate.json"
            self._write_json(report_path, report)
            record_gates[n] = report_path

        inputs = {
            "training_receipt": training_receipt,
            "selection_receipt": selection_receipt,
            "config": config,
            "checkpoint": checkpoint,
            "val_sweep_summary": sweep_summary,
            "final_val3_summary": final_val3,
            "test5_summary": test5,
            "strict2031_summary": strict_paths[2031]["candidate_summary"],
            "strict1607_summary": strict_paths[1607]["candidate_summary"],
            "strict2031_record_gate": record_gates[2031],
            "strict1607_record_gate": record_gates[1607],
            "baseline_ref8_summary": baseline_ref8,
            "baseline_strict2031_summary": strict_paths[2031]["baseline_summary"],
            "baseline_strict1607_summary": strict_paths[1607]["baseline_summary"],
            "expected_checkpoint_sha256": checkpoint_record["sha256"],
            "split_contract": contract,
            "canonical_seeds": seeds,
            "repo_root": root,
            "_selection_rebuilder": lambda **_kwargs: json.loads(
                selection_receipt.read_text(encoding="utf-8")
            ),
            "_record_gate_rebuilder": lambda report: report,
        }
        return inputs

    def _mutate_json(self, path: Path, callback):
        payload = json.loads(path.read_text(encoding="utf-8"))
        callback(payload)
        self._write_json(path, payload)

    def test_builds_canonical_receipt_and_publishes_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._fixture(root)
            receipt = final.build_final_receipt_payload(**inputs)
            self.assertEqual(receipt["status"], "pass")
            self.assertEqual(len(receipt["inputs"]), len(final.INPUT_ROLES))
            self.assertTrue(
                all(row["improved"] for row in receipt["results"]["ref8_acc50"].values())
            )
            unhashed = dict(receipt)
            del unhashed["payload_sha256"]
            self.assertEqual(
                receipt["payload_sha256"], final.canonical_json_sha256(unhashed)
            )

            output_json = root / "final_result_receipt.json"
            output_md = root / "final_result.md"
            _, first = final.build_and_publish(
                output_json=output_json, output_md=output_md, **inputs
            )
            _, second = final.build_and_publish(
                output_json=output_json, output_md=output_md, **inputs
            )
            self.assertEqual(first, {"json": "created", "markdown": "created"})
            self.assertEqual(
                second,
                {"json": "already_identical", "markdown": "already_identical"},
            )
            self.assertIn("Status: PASS", output_md.read_text(encoding="ascii"))

    def test_refuses_to_overwrite_different_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._fixture(root)
            output = root / "final.json"
            output.write_text("different\n", encoding="ascii")
            with self.assertRaisesRegex(final.FinalReceiptError, "refusing to overwrite"):
                final.build_and_publish(output_json=output, **inputs)

    def test_rejects_selection_other_than_v2_gap3(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            path = inputs["selection_receipt"]

            def mutate(payload):
                payload["schema"] = "v1"
                del payload["payload_sha256"]
                payload["payload_sha256"] = final.canonical_json_sha256(payload)

            self._mutate_json(path, mutate)
            with self.assertRaisesRegex(final.FinalReceiptError, "schema must be v2"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_selection_not_reproduced_by_v2_selector(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            replayed = json.loads(
                inputs["selection_receipt"].read_text(encoding="utf-8")
            )
            replayed["selection"]["max_gap"] = 2.0
            inputs["_selection_rebuilder"] = lambda **_kwargs: replayed
            with self.assertRaisesRegex(final.FinalReceiptError, "not exactly reproducible"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_final_config_selection_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            inputs["config"].write_text(
                inputs["config"].read_text(encoding="ascii").replace("3.0", "4.0", 1),
                encoding="ascii",
            )
            with self.assertRaisesRegex(final.FinalReceiptError, "max_gap"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_resolved_arbiter_or_merged_eval_config(self):
        for field in (
            "stage_b_row_text_arbiter",
            "stage_b_gdino_adapter_merged_eval_only",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                inputs = self._fixture(Path(temporary))
                with inputs["config"].open("a", encoding="ascii") as handle:
                    handle.write(f"{field} = True\n")
                with self.assertRaisesRegex(final.FinalReceiptError, field):
                    final.build_final_receipt_payload(**inputs)

    def test_rejects_disabled_resolved_headline_branch(self):
        for field in (
            "stage_b_u0_category_preserving_patch_gate",
            "stage_b_u0_patch_rank",
            "stage_b_gdino_score_adapter",
            "enable_patch_branch",
            "stage_b_u2_category_complete_supervision",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                inputs = self._fixture(Path(temporary))
                with inputs["config"].open("a", encoding="ascii") as handle:
                    handle.write(f"{field} = False\n")
                with self.assertRaisesRegex(final.FinalReceiptError, field):
                    final.build_final_receipt_payload(**inputs)

    def test_rejects_non_b16_w4_full_canonical_final_val(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            self._mutate_json(
                inputs["final_val3_summary"],
                lambda payload: payload["refcoco"][0].__setitem__("batch_size", 32),
            )
            with self.assertRaisesRegex(final.FinalReceiptError, "B16/W4/full/canonical"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_ref8_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            self._mutate_json(
                inputs["test5_summary"],
                lambda payload: payload["refcoco"][0].__setitem__("acc50", 0.5),
            )
            with self.assertRaisesRegex(final.FinalReceiptError, "Ref8 gate failed"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_fpr95_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            self._mutate_json(
                inputs["strict2031_summary"],
                lambda payload: payload["tn"][0].__setitem__("fpr95tpr", 0.5),
            )
            with self.assertRaisesRegex(final.FinalReceiptError, "FPR95 gate failed"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_record_gate_without_validation_and_gate_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            self._mutate_json(
                inputs["strict1607_record_gate"],
                lambda payload: payload["validation"].__setitem__("pass", False),
            )
            with self.assertRaisesRegex(final.FinalReceiptError, "validation.pass"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_record_gate_not_exactly_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))

            def drifted(report):
                replayed = json.loads(json.dumps(report))
                replayed["gate"]["global_fpr95_lower"] = False
                return replayed

            inputs["_record_gate_rebuilder"] = drifted
            with self.assertRaisesRegex(final.FinalReceiptError, "not exactly reproducible"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_changed_baseline_tn_record_bound_by_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            summary = json.loads(
                inputs["baseline_strict2031_summary"].read_text(encoding="utf-8")
            )
            record_path = Path(summary["tn"][0]["records_jsonl"])
            with record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"changed": True}) + "\n")
            with self.assertRaisesRegex(final.FinalReceiptError, "file identity mismatch"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_different_real_baseline_tn_record_in_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self._fixture(root)
            replacement = self._record_file(root, "other_real_baseline_tn")

            def mutate(payload):
                payload["input_files"]["baseline"][-1] = self._record(replacement)

            self._mutate_json(inputs["strict2031_record_gate"], mutate)
            with self.assertRaisesRegex(final.FinalReceiptError, "baseline TN record path"):
                final.build_final_receipt_payload(**inputs)

    def test_rejects_candidate_checkpoint_binding_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._fixture(Path(temporary))
            self._mutate_json(
                inputs["final_val3_summary"],
                lambda payload: payload["refcoco"][0].__setitem__(
                    "checkpoint_sha256", "0" * 64
                ),
            )
            with self.assertRaisesRegex(final.FinalReceiptError, "checkpoint SHA-256"):
                final.build_final_receipt_payload(**inputs)


if __name__ == "__main__":
    unittest.main()
