import hashlib
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools.build_stageb_paper_results_manifest import (
    OUTPUT_SCHEMA,
    REF_SPLITS,
    SPEC_SCHEMA,
    PaperManifestBuildError,
    build_manifest,
    main,
    validate_manifest,
    write_manifest_new,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BuildStageBPaperResultsManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_contract = {
            split: {
                "rows": 4,
                "sha256": hashlib.sha256(
                    f"manifest:{split}".encode("ascii")
                ).hexdigest(),
            }
            for split in REF_SPLITS
        }
        self._contract_patches = (
            patch(
                "tools.build_stageb_paper_results_manifest.REF_SPLIT_CONTRACT",
                fixture_contract,
            ),
            patch(
                "tools.aggregate_stageb_paper_results.REF_SPLIT_CONTRACT",
                fixture_contract,
            ),
        )
        for contract_patch in self._contract_patches:
            contract_patch.start()
            self.addCleanup(contract_patch.stop)

    def _fixture(self, root: Path, *, expected_seeds=(17,)) -> Path:
        checkpoint = root / "train" / "checkpoint.pth"
        config = root / "train" / "config.py"
        data = root / "train" / "data.jsonl"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint-seed-17")
        config.write_text("seed = 17\n", encoding="utf-8")
        _write_jsonl(data, [{"train": 1}])

        sources = {}
        source_rows = {}
        for strict, eval_split in (
            ("strict2031", "refcocop_val"),
            ("strict1607", "refcocog_umd_val"),
        ):
            source_path = root / "protocol" / f"{strict}.jsonl"
            rows = [
                {
                    "sample_id": f"{strict}:{index}",
                    "image_id": index // 2,
                    "ann_id": 100 + index,
                    "ref_id": 200 + index,
                    "sent_id": 300 + index,
                    "eval_split": eval_split,
                    "instances": [{"replace_category": ["color"]}],
                }
                for index in range(4)
            ]
            _write_jsonl(source_path, rows)
            sources[strict] = source_path
            source_rows[strict] = rows

        run_id = "baseline_seed17"
        ref_root = root / "eval" / "ref8"
        ref_summary_rows = []
        for split_index, split in enumerate(REF_SPLITS):
            records_path = ref_root / "per_example_records" / f"{split}.records.jsonl"
            manifest_sha = hashlib.sha256(f"manifest:{split}".encode()).hexdigest()
            records = [
                {
                    "schema": "stageb-eval-record-v1",
                    "task": "ref",
                    "manifest_key": f"ref:{split}",
                    "manifest_sha256": manifest_sha,
                    "manifest_n": 4,
                    "manifest_index": index,
                    "sample_id": f"{split}:{index}",
                    "image_id": split_index * 10 + index // 2,
                    "ann_id": 10 + index,
                    "ref_id": 20 + index,
                    "sent_id": 30 + index,
                    "split": split,
                    "run_id": run_id,
                    "valid": True,
                    "correct50": index < 3,
                    "top1_iou": 0.8 if index < 3 else 0.1,
                }
                for index in range(4)
            ]
            _write_jsonl(records_path, records)
            ref_summary_rows.append(
                {
                    "dataset": split,
                    "run_id": run_id,
                    "checkpoint": str(checkpoint),
                    "records_jsonl": str(records_path),
                    "manifest_n": 4,
                    "manifest_sha256": manifest_sha,
                    "num_expressions": 4,
                    "acc50": 0.75,
                    "max_batches": 0,
                    "invalid_records": 0,
                }
            )
        _write_json(ref_root / "summary.json", {"refcoco": ref_summary_rows, "tn": []})

        tn_roots = {}
        for strict in ("strict2031", "strict1607"):
            tn_root = root / "eval" / strict
            tn_roots[strict] = tn_root
            records_path = tn_root / "per_example_records" / "tn.records.jsonl"
            source_path = sources[strict]
            source_sha = _sha256(source_path)
            records = [
                {
                    "schema": "stageb-eval-record-v1",
                    "task": "tn",
                    "manifest_key": "tn_global",
                    "manifest_sha256": source_sha,
                    "manifest_n": 4,
                    "manifest_index": index,
                    "sample_id": source["sample_id"],
                    "image_id": source["image_id"],
                    "split": source["eval_split"],
                    "run_id": run_id,
                    "valid": True,
                    "pos_score": 0.8 - index * 0.05,
                    "neg_score": 0.2 + index * 0.02,
                }
                for index, source in enumerate(source_rows[strict])
            ]
            _write_jsonl(records_path, records)
            _write_json(
                tn_root / "summary.json",
                {
                    "refcoco": [],
                    "tn": [
                        {
                            "run_id": run_id,
                            "checkpoint": str(checkpoint),
                            "records_jsonl": str(records_path),
                            "manifest_n": 4,
                            "manifest_sha256": source_sha,
                            "num_pairs": 4,
                            "fpr95tpr": 0.0,
                            "max_batches": 0,
                            "invalid_records": 0,
                            "source_manifest_path": str(source_path),
                            "source_manifest_sha256": source_sha,
                            "source_manifest_size_bytes": source_path.stat().st_size,
                            "source_manifest_n": 4,
                        }
                    ],
                },
            )

        spec = root / "paper_build_spec.json"
        _write_json(
            spec,
            {
                "schema": SPEC_SCHEMA,
                "expected_train_seeds": list(expected_seeds),
                "baseline_experiment": "baseline",
                "strict_source_manifests": {
                    strict: {"path": str(path), "expected_n": 4}
                    for strict, path in sources.items()
                },
                "bootstrap": {"iterations": 20, "confidence": 0.95, "seed": 9},
                "experiments": [
                    {
                        "id": "baseline",
                        "label": "Baseline (one train seed)",
                        "runs": [
                            {
                                "train_seed": 17,
                                "checkpoint": str(checkpoint),
                                "config": str(config),
                                "data": [str(data)],
                                "ref_eval_root": str(ref_root),
                                "strict2031_eval_root": str(tn_roots["strict2031"]),
                                "strict1607_eval_root": str(tn_roots["strict1607"]),
                            }
                        ],
                    }
                ],
            },
        )
        return spec

    def _evaluation_root_from_spec(self, spec: Path) -> Path:
        payload = json.loads(spec.read_text(encoding="utf-8"))
        run = payload["experiments"][0]["runs"][0]
        root = spec.parent / "sealed_final_evaluation"
        primary = root / "ref8_strict2031"
        supplemental = root / "strict1607"
        primary_records = primary / "per_example_records"
        supplemental_records = supplemental / "per_example_records"
        primary_records.mkdir(parents=True)
        supplemental_records.mkdir(parents=True)

        ref_summary = json.loads(
            (Path(run["ref_eval_root"]) / "summary.json").read_text(encoding="utf-8")
        )
        for row in ref_summary["refcoco"]:
            source = Path(row["records_jsonl"])
            destination = primary_records / source.name
            shutil.copy2(source, destination)
            row["records_jsonl"] = str(destination)
        strict2031_summary = json.loads(
            (Path(run["strict2031_eval_root"]) / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        strict2031_row = strict2031_summary["tn"][0]
        source = Path(strict2031_row["records_jsonl"])
        destination = primary_records / "strict2031_tn.records.jsonl"
        shutil.copy2(source, destination)
        strict2031_row["records_jsonl"] = str(destination)
        _write_json(
            primary / "summary.json",
            {"refcoco": ref_summary["refcoco"], "tn": [strict2031_row]},
        )

        strict1607_summary = json.loads(
            (Path(run["strict1607_eval_root"]) / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        strict1607_row = strict1607_summary["tn"][0]
        source = Path(strict1607_row["records_jsonl"])
        destination = supplemental_records / "strict1607_tn.records.jsonl"
        shutil.copy2(source, destination)
        strict1607_row["records_jsonl"] = str(destination)
        _write_json(
            supplemental / "summary.json",
            {"refcoco": [], "tn": [strict1607_row]},
        )

        checkpoint = Path(run["checkpoint"])
        config = Path(run["config"])
        data = [str(Path(value)) for value in run["data"]]
        input_paths = [checkpoint, config, *[Path(value) for value in data]]
        input_records = []
        rehash_records = []
        for path in input_paths:
            stat = path.stat()
            record = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "roles": ["fixture"],
            }
            input_records.append(record)
            rehash_records.append(
                {
                    "path": record["path"],
                    "expected_sha256": record["sha256"],
                    "observed_sha256": record["sha256"],
                    "observed_size_bytes": record["size_bytes"],
                    "observed_mtime_ns": record["mtime_ns"],
                    "passed": True,
                }
            )

        def evaluation_artifact(path: Path, *roles: str) -> dict:
            resolved = path.resolve()
            stat = resolved.stat()
            return {
                "path": str(resolved),
                "roles": sorted(set(roles)),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256(resolved),
            }

        ref_artifacts = {}
        for row in ref_summary["refcoco"]:
            split = row["dataset"]
            ref_artifacts[split] = {
                "summary_acc50": row["acc50"],
                "manifest_n": row["manifest_n"],
                "manifest_sha256": row["manifest_sha256"],
                "records": evaluation_artifact(
                    Path(row["records_jsonl"]), f"ref:{split}"
                ),
            }

        def tn_artifact(row: dict, split: str) -> dict:
            return {
                "summary_fpr95": row["fpr95tpr"],
                "manifest_binding_mode": "source_to_derived_v1",
                "manifest_n": row["manifest_n"],
                "source_manifest_sha256": row["source_manifest_sha256"],
                "derived_manifest_sha256": row["manifest_sha256"],
                "records": evaluation_artifact(
                    Path(row["records_jsonl"]), split, "tn_records"
                ),
            }

        postflight = {
            "schema": "pivot.stageb.paper_evaluation_postflight/v1",
            "status": "passed",
            "profile": "final",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
                "run_id": "fixture",
            },
            "input_rehash": {"status": "passed", "records": rehash_records},
            "artifacts": {
                "primary_summary": evaluation_artifact(
                    primary / "summary.json", "ref8", "strict2031"
                ),
                "supplemental_summary": evaluation_artifact(
                    supplemental / "summary.json", "strict1607"
                ),
                "ref8": ref_artifacts,
                "strict2031": tn_artifact(strict2031_row, "strict2031"),
                "strict1607": tn_artifact(strict1607_row, "strict1607"),
            },
            "contracts": {
                "ref_split_set_exact": True,
                "full_per_example_records": True,
                "zero_invalid_records": True,
                "locked_manifest_binding": True,
                "checkpoint_consistent_across_all_rows": True,
                "strict1607_skip_ref_observed": True,
            },
        }
        postflight_path = root / "postflight.json"
        _write_json(postflight_path, postflight)
        postflight_artifact = {
            "path": str(postflight_path.resolve()),
            "sha256": _sha256(postflight_path),
            "size_bytes": postflight_path.stat().st_size,
        }
        launch = {
            "schema": "pivot.stageb.paper_evaluation_launch/v1",
            "status": "completed",
            "output_dir": str(root.resolve()),
            "source": {
                "kind": "pivot_token_ablation_training_run",
                "training_seed": int(run["train_seed"]),
                "training_run_id": f"L4:{int(run['train_seed'])}",
                "checkpoint": str(checkpoint.resolve()),
                "config": str(config.resolve()),
                "training_data": data,
            },
            "protocol": {
                "profile": "final",
                "processes": ["ref8_strict2031", "strict1607"],
            },
            "completed_phases": [
                {
                    "phase_id": "ref8_strict2031",
                    "status": "completed",
                    "returncode": 0,
                },
                {
                    "phase_id": "strict1607",
                    "status": "completed",
                    "returncode": 0,
                },
            ],
            "inputs": {"algorithm": "sha256", "records": input_records},
            "postflight": postflight,
            "postflight_artifact": postflight_artifact,
        }
        _write_json(root / "launch_manifest.json", launch)
        return root

    def test_builds_sha_bound_ten_record_manifest_and_validates_aggregator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            manifest = build_manifest(spec)

            self.assertEqual(manifest["schema"], OUTPUT_SCHEMA)
            self.assertEqual(
                manifest["protocol"]["headline_release_provenance"]["status"],
                "unverified_legacy_manifest",
            )
            self.assertEqual(manifest["expected_train_seeds"], [17])
            run = manifest["experiments"][0]["runs"][0]
            self.assertEqual(run["train_seed"], 17)
            self.assertEqual(set(run["results"]["ref"]["records"]), set(REF_SPLITS))
            self.assertEqual(set(run["results"]["tn"]), {"strict2031", "strict1607"})
            for artifact in (
                list(run["results"]["ref"]["records"].values())
                + [run["results"]["tn"][split]["records"] for split in ("strict2031", "strict1607")]
            ):
                self.assertEqual(len(artifact["sha256"]), 64)
                self.assertGreater(artifact["size_bytes"], 0)
            validate_manifest(manifest, bootstrap_iterations=3)

    def test_evaluation_root_mode_derives_training_and_result_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            evaluation_root = self._evaluation_root_from_spec(spec)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["experiments"][0]["runs"] = [
                {
                    "train_seed": 17,
                    "evaluation_root": str(evaluation_root),
                    "expected_training_run_id": "L4:17",
                }
            ]
            _write_json(spec, payload)
            manifest = build_manifest(spec)
            run = manifest["experiments"][0]["runs"][0]
            self.assertEqual(run["train_seed"], 17)
            self.assertEqual(
                Path(run["artifacts"]["checkpoint"]["path"]).name,
                "checkpoint.pth",
            )
            self.assertEqual(len(run["artifacts"]["data"]), 1)
            self.assertEqual(set(run["results"]["ref"]["records"]), set(REF_SPLITS))
            self.assertEqual(set(run["results"]["tn"]), {"strict2031", "strict1607"})
            validate_manifest(manifest, bootstrap_iterations=3)

            payload["experiments"][0]["runs"][0][
                "expected_training_run_id"
            ] = "L3:17"
            _write_json(spec, payload)
            with self.assertRaisesRegex(
                PaperManifestBuildError, "source run_id.*L3:17"
            ):
                build_manifest(spec)

    def test_evaluation_root_rehashes_every_postflight_output_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            evaluation_root = self._evaluation_root_from_spec(spec)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["experiments"][0]["runs"] = [
                {
                    "train_seed": 17,
                    "evaluation_root": str(evaluation_root),
                    "expected_training_run_id": "L4:17",
                }
            ]
            _write_json(spec, payload)

            postflight = json.loads(
                (evaluation_root / "postflight.json").read_text(encoding="utf-8")
            )
            record_path = Path(
                postflight["artifacts"]["ref8"]["refcoco_val"]["records"][
                    "path"
                ]
            )
            rendered = record_path.read_bytes()
            self.assertIn(b"refcoco_val:0", rendered)
            record_path.write_bytes(
                rendered.replace(b"refcoco_val:0", b"refcoco_val:9", 1)
            )
            with self.assertRaisesRegex(PaperManifestBuildError, "SHA-256 mismatch"):
                build_manifest(spec)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            evaluation_root = self._evaluation_root_from_spec(spec)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["experiments"][0]["runs"] = [
                {
                    "train_seed": 17,
                    "evaluation_root": str(evaluation_root),
                    "expected_training_run_id": "L4:17",
                }
            ]
            _write_json(spec, payload)
            postflight_path = evaluation_root / "postflight.json"
            launch_path = evaluation_root / "launch_manifest.json"
            postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
            postflight.pop("artifacts")
            _write_json(postflight_path, postflight)
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["postflight"] = postflight
            launch["postflight_artifact"] = {
                "path": str(postflight_path.resolve()),
                "sha256": _sha256(postflight_path),
                "size_bytes": postflight_path.stat().st_size,
            }
            _write_json(launch_path, launch)
            with self.assertRaisesRegex(
                PaperManifestBuildError, "postflight.artifacts: expected an object"
            ):
                build_manifest(spec)

    def test_evaluation_root_mode_rejects_screen_profile_and_mixed_explicit_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            evaluation_root = self._evaluation_root_from_spec(spec)
            launch_path = evaluation_root / "launch_manifest.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["protocol"]["profile"] = "screen_validation"
            _write_json(launch_path, launch)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["experiments"][0]["runs"] = [
                {
                    "train_seed": 17,
                    "evaluation_root": str(evaluation_root),
                    "expected_training_run_id": "L4:17",
                }
            ]
            _write_json(spec, payload)
            with self.assertRaisesRegex(
                PaperManifestBuildError, "only the final evaluation profile"
            ):
                build_manifest(spec)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            evaluation_root = self._evaluation_root_from_spec(spec)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            original = payload["experiments"][0]["runs"][0]
            payload["experiments"][0]["runs"] = [
                {
                    "train_seed": 17,
                    "evaluation_root": str(evaluation_root),
                    "expected_training_run_id": "L4:17",
                    "checkpoint": original["checkpoint"],
                }
            ]
            _write_json(spec, payload)
            with self.assertRaisesRegex(
                PaperManifestBuildError, "cannot be mixed"
            ):
                build_manifest(spec)

    def test_missing_duplicate_and_cross_seed_checkpoint_contracts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root, expected_seeds=(17, 23))
            with self.assertRaisesRegex(PaperManifestBuildError, r"missing=\[23\]"):
                build_manifest(spec)

            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["expected_train_seeds"] = [17, 17]
            _write_json(spec, payload)
            with self.assertRaisesRegex(PaperManifestBuildError, "contains duplicates"):
                build_manifest(spec)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            duplicate = dict(payload["experiments"][0]["runs"][0])
            payload["experiments"][0]["runs"].append(duplicate)
            _write_json(spec, payload)
            with self.assertRaisesRegex(PaperManifestBuildError, "duplicate train seed"):
                build_manifest(spec)

    def test_summary_selection_checkpoint_and_record_contracts_fail_closed(self):
        mutations = ("ambiguous", "checkpoint", "record")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                spec = self._fixture(root)
                payload = json.loads(spec.read_text(encoding="utf-8"))
                ref_root = Path(
                    payload["experiments"][0]["runs"][0]["ref_eval_root"]
                )
                summary_path = ref_root / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if mutation == "ambiguous":
                    summary["refcoco"].append(dict(summary["refcoco"][0]))
                    expected = "expected exactly one summary row"
                elif mutation == "checkpoint":
                    summary["refcoco"][0]["checkpoint"] = str(root / "wrong.pth")
                    expected = "expected exactly one summary row"
                else:
                    record_path = Path(summary["refcoco"][0]["records_jsonl"])
                    rows = [json.loads(line) for line in record_path.read_text().splitlines()]
                    rows[0]["manifest_index"] = 99
                    _write_jsonl(record_path, rows)
                    expected = "manifest_index does not match"
                _write_json(summary_path, summary)
                with self.assertRaisesRegex(PaperManifestBuildError, expected):
                    build_manifest(spec)

    def test_atomic_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_manifest(self._fixture(root))
            output = root / "result_manifest.json"
            write_manifest_new(manifest, output)
            original = output.read_bytes()
            with self.assertRaisesRegex(PaperManifestBuildError, "refusing to overwrite"):
                write_manifest_new(manifest, output)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_writes_once_and_rejects_existing_output_before_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            output = root / "cli_manifest.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["--spec", str(spec), "--output", str(output)]), 0
                )
            original = output.read_bytes()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(
                    main(["--spec", str(root / "now-missing.json"), "--output", str(output)]),
                    2,
                )
            self.assertIn("refusing to overwrite", errors.getvalue())
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
