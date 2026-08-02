import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from tools import build_stageb_data_driven_new_head_lr_preregistration as builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _seal(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("canonical_payload_sha256", None)
    result["canonical_payload_sha256"] = hashlib.sha256(
        builder._canonical_bytes(result)
    ).hexdigest()
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


class NewHeadLRPreregistrationFixture:
    def __init__(self):
        self.context = tempfile.TemporaryDirectory()
        self.root = Path(self.context.name).resolve()
        (self.root / "config/ablations").mkdir(parents=True)
        (self.root / "tools").mkdir()
        self.main_path = self.root / "main.py"
        self.evaluator_path = self.root / "tools/eval_new_head.py"
        self.selector_path = self.root / "tools/select_new_head_lr.py"
        self.main_path.write_text("def main():\n    pass\n", encoding="ascii")
        self.evaluator_path.write_text("EVALUATOR = True\n", encoding="ascii")
        self.selector_path.write_text("SELECTOR = True\n", encoding="ascii")

        partition_root = self.root / "data/new_head_partition"
        support_root = self.root / "data/support_partition"
        initializer_root = self.root / "outputs/initializers"
        partition_root.mkdir(parents=True)
        support_root.mkdir(parents=True)
        initializer_root.mkdir(parents=True)

        self.train_records = {}
        self.dev_screen_records = {}
        for index, manifest_name in enumerate(builder.SOURCE_MANIFESTS):
            train_path = (
                partition_root
                / "d1_category_complete/train"
                / manifest_name
            )
            dev_path = (
                partition_root
                / "d0_ordinary_primary/dev_screen"
                / manifest_name
            )
            train_path.parent.mkdir(parents=True, exist_ok=True)
            dev_path.parent.mkdir(parents=True, exist_ok=True)
            train_path.write_text(
                json.dumps({"source": manifest_name, "row": index}) + "\n",
                encoding="ascii",
            )
            dev_path.write_text(
                json.dumps({"source": manifest_name, "dev": index}) + "\n",
                encoding="ascii",
            )
            self.train_records[manifest_name] = {
                **_file_record(train_path),
                "rows": 1,
                "ordered_identity_stream_sha256": hashlib.sha256(
                    f"train-{manifest_name}".encode("ascii")
                ).hexdigest(),
            }
            self.dev_screen_records[manifest_name] = {
                **_file_record(dev_path),
                "rows": 1,
                "ordered_identity_stream_sha256": hashlib.sha256(
                    f"dev-{manifest_name}".encode("ascii")
                ).hexdigest(),
            }

        self.partition_receipt_path = partition_root / "receipt.json"
        partition = _seal(
            {
                "schema": builder.PARTITION_SCHEMA,
                "source_manifest_order": list(builder.SOURCE_MANIFESTS),
                "outputs": {
                    "d1_category_complete": {
                        "train": self.train_records,
                    },
                    "d0_ordinary_primary": {
                        "dev_screen": self.dev_screen_records,
                    },
                },
                "partition_summary": {
                    "train": {
                        "rows": 3,
                        "rows_by_manifest": {
                            name: 1 for name in builder.SOURCE_MANIFESTS
                        },
                    },
                    "dev_screen": {
                        "rows": 3,
                        "rows_by_manifest": {
                            name: 1 for name in builder.SOURCE_MANIFESTS
                        },
                    },
                },
                "invariants": {
                    "fixture_partition_is_paired": True,
                    "fixture_dev_screen_is_shared": True,
                },
            }
        )
        _write_json(self.partition_receipt_path, partition)

        self.support_tsv = support_root / "filtered_support.tsv"
        support_image = support_root / "support.jpg"
        support_image.write_bytes(b"support")
        self.support_tsv.write_text(
            f"path\tclass_id\n{support_image}\t1\n",
            encoding="ascii",
        )
        self.support_receipt_path = support_root / "receipt.json"
        support = _seal(
            {
                "schema": builder.SUPPORT_SCHEMA,
                "inputs": {
                    "partition_receipt": _file_record(
                        self.partition_receipt_path
                    )
                },
                "partition": {
                    "schema": builder.PARTITION_SCHEMA,
                    "receipt": _file_record(self.partition_receipt_path),
                    "canonical_payload_sha256": partition[
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
                "outputs": {
                    "runtime_support_tsv": {
                        **_file_record(self.support_tsv),
                        "rows": 1,
                    }
                },
                "runtime_bank": {"candidate_rows": 1, "class_count": 1},
                "training_class_coverage": {
                    "required_class_count": 1,
                    "covered_class_count": 1,
                    "missing_class_ids": [],
                },
                "invariants": {
                    "fixture_support_is_shared": True,
                    "fixture_training_classes_are_covered": True,
                },
            }
        )
        _write_json(self.support_receipt_path, support)

        self.a0_initializer_path = initializer_root / "a0.pth"
        self.a0_initializer_path.write_bytes(b"sealed-a0")
        self.pair_receipt_path = initializer_root / "pair.json"
        pair = {
            "schema": builder.PAIR_SCHEMA,
            "status": "passed",
            "absolute_initializer": _file_record(self.a0_initializer_path),
            "relational_initializer": {
                "path": str(initializer_root / "unused-a1.pth"),
                "size_bytes": 0,
                "sha256": "0" * 64,
            },
            "invariants": {
                "b58_is_only_tensor_checkpoint_source": True,
                "all_common_non_rank_non_contract_tensors_bitwise_equal": True,
                "patch_and_confidence_initialization_bitwise_equal": True,
                "rank_subtree_is_the_only_parameterized_architecture_intervention": True,
                "no_teacher_u1000_u5020_or_old_initializer_tensor_source": True,
            },
        }
        _write_json(self.pair_receipt_path, pair)

        self.dataset_path = self.root / "config/datasets_dd1_new_head.json"
        dataset = {
            "train": [
                {
                    "dataset_mode": "patch_episode",
                    "root": "/",
                    "anno": self.train_records[name]["path"],
                    "stage_b_data_driven_variant": "dd1_category_complete",
                    "stage_b_data_driven_partition": "train",
                    "stage_b_data_driven_manifest_sha256": self.train_records[
                        name
                    ]["sha256"],
                    "stage_b_data_driven_receipt": str(
                        self.partition_receipt_path
                    ),
                    "stage_b_data_driven_receipt_sha256": _sha256(
                        self.partition_receipt_path
                    ),
                    "stage_b_data_driven_support_receipt": str(
                        self.support_receipt_path
                    ),
                    "stage_b_data_driven_support_receipt_sha256": _sha256(
                        self.support_receipt_path
                    ),
                    "support_patch_tsv": str(self.support_tsv),
                    "support_patch_use_embedding": False,
                    "support_patch_max_per_class": 200,
                    "patch_bank_cache": False,
                    "patch_bank_cache_write": False,
                    "mix_weight": 2.0,
                }
                for name in builder.SOURCE_MANIFESTS
            ],
            "val": [],
        }
        _write_json(self.dataset_path, dataset)

        candidate_root = self.root / "outputs/lr_probe_u1000"
        candidate_values = (
            (
                "lr3e5",
                3e-5,
                "cfg_stageb_data_driven_dd1_new_head_lr3e5_probe_u1000_20260723.py",
            ),
            (
                "lr1e4",
                1e-4,
                "cfg_stageb_data_driven_dd1_new_head_lr1e4_probe_u1000_20260723.py",
            ),
            (
                "lr3e4",
                3e-4,
                "cfg_stageb_data_driven_dd1_new_head_lr3e4_probe_u1000_20260723.py",
            ),
        )
        candidates = tuple(
            builder.CandidateSpec(
                label=label,
                rank_lr=rank_lr,
                config_path=self.root / "config/ablations" / filename,
                training_output_dir=candidate_root / label,
                dev_screen_eval_dir=(
                    candidate_root
                    / label
                    / "evaluations/new_head_dev_screen"
                ),
            )
            for label, rank_lr, filename in candidate_values
        )
        self.spec = builder.PreregistrationSpec(
            repo_root=self.root,
            dataset_path=self.dataset_path,
            dataset_sha256=_sha256(self.dataset_path),
            partition_receipt_path=self.partition_receipt_path,
            partition_receipt_sha256=_sha256(self.partition_receipt_path),
            support_receipt_path=self.support_receipt_path,
            support_receipt_sha256=_sha256(self.support_receipt_path),
            a0_initializer_path=self.a0_initializer_path,
            a0_initializer_sha256=_sha256(self.a0_initializer_path),
            pair_receipt_path=self.pair_receipt_path,
            pair_receipt_sha256=_sha256(self.pair_receipt_path),
            main_path=self.main_path,
            evaluator_path=self.evaluator_path,
            selector_path=self.selector_path,
            candidates=candidates,
        )
        self.base_config_path = (
            self.root
            / "config/ablations/"
            "cfg_stageb_data_driven_dd1_new_head_lr_probe_20260723.py"
        )
        self.base_values = builder._required_config_values(self.spec)
        self.base_values.update(
            {
                "stage_b_data_driven_rank_lr": 3e-4,
                "stage_b_data_driven_rank_dim": 128,
                "stage_b_data_driven_rank_num_heads": 4,
                "stage_b_data_driven_rank_image_level_policy": "last",
                "stage_b_data_driven_rank_image_levels": 2,
                "stage_b_data_driven_rank_image_pool_size": 8,
                "stage_b_data_driven_rank_image_pool_policy": (
                    "valid_extent_masked_adaptive_avg_v1"
                ),
                "stage_b_data_driven_rank_box_fourier_bands": 16,
                "stage_b_data_driven_rank_ffn_dim": 512,
                "stage_b_data_driven_rank_dropout": 0.0,
                "stage_b_data_driven_head_init_seed": 42,
                "stage_b_data_driven_rank_negative_iou_threshold": 0.3,
                "stage_b_data_driven_positive_iou_threshold": 0.5,
                "stage_b_data_driven_patch_negative_iou_threshold": 0.3,
                "stage_b_data_driven_temperature": 0.1,
                "stage_b_data_driven_category_margin": 0.1,
            }
        )
        self.write_base_config()
        for candidate in candidates:
            candidate.config_path.write_text(
                "from "
                + builder.PROBE_BASE_MODULE
                + " import *  # noqa: F401,F403\n\n"
                + f"stage_b_data_driven_rank_lr = {candidate.rank_lr!r}\n",
                encoding="ascii",
            )

    def close(self):
        self.context.cleanup()

    def write_base_config(self):
        self.base_config_path.write_text(
            "".join(
                f"{key} = {value!r}\n"
                for key, value in sorted(self.base_values.items())
            ),
            encoding="ascii",
        )


class NewHeadLRPreregistrationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = NewHeadLRPreregistrationFixture()

    def tearDown(self):
        self.fixture.close()

    def test_builds_canonical_preregistration_with_exact_selection_contract(self):
        payload = builder.build_preregistration(self.fixture.spec)
        self.assertEqual(payload["schema"], builder.SCHEMA)
        self.assertEqual(payload["status"], "preregistered")
        self.assertEqual(payload["candidate_rank_lrs"], [3e-5, 1e-4, 3e-4])
        self.assertEqual(payload["optimizer_updates_per_candidate"], 1000)
        self.assertEqual(payload["selection_partition"], "dev_screen")
        self.assertEqual(payload["selection_metric"], "macro_ref3_acc50")
        self.assertEqual(
            payload["secondary_selection_metric"],
            "macro_ref3_mean_listwise_nll",
        )
        self.assertEqual(
            payload["tie_break_rule"],
            [
                "max_macro_ref3_acc50",
                "min_macro_ref3_mean_listwise_nll",
                "min_rank_lr",
            ],
        )
        self.assertEqual(
            payload["evaluation_contract"]["manifest_variant"],
            "d0_ordinary_primary",
        )
        self.assertEqual(payload["evaluation_contract"]["rows"], 3)
        self.assertEqual(
            payload["training_contract"]["fresh_start"]["resume"], ""
        )
        self.assertEqual(payload["training_contract"]["batch_size"], 64)
        self.assertTrue(payload["training_contract"]["amp"])
        self.assertTrue(all(payload["invariants"].values()))
        self.assertEqual(
            {
                item["effective_training_contract_without_rank_lr_sha256"]
                for item in payload["candidates"]
            },
            {
                payload["candidates"][0][
                    "effective_training_contract_without_rank_lr_sha256"
                ]
            },
        )
        canonical = dict(payload)
        digest = canonical.pop("canonical_payload_sha256")
        self.assertEqual(
            digest, hashlib.sha256(builder._canonical_bytes(canonical)).hexdigest()
        )

    def test_rejects_thin_leaf_or_shared_training_semantic_drift(self):
        candidate = self.fixture.spec.candidates[1]
        candidate.config_path.write_text(
            candidate.config_path.read_text(encoding="ascii")
            + "batch_size = 32\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            builder.NewHeadLRPreregistrationError,
            "thin probe config",
        ):
            builder.build_preregistration(self.fixture.spec)

        candidate.config_path.write_text(
            "from "
            + builder.PROBE_BASE_MODULE
            + " import *  # noqa: F401,F403\n\n"
            + f"stage_b_data_driven_rank_lr = {candidate.rank_lr!r}\n",
            encoding="ascii",
        )
        self.fixture.base_values["stage_b_data_driven_patch_lr"] = 1e-4
        self.fixture.write_base_config()
        with self.assertRaisesRegex(
            builder.NewHeadLRPreregistrationError,
            "fixed training contract drifted",
        ):
            builder.build_preregistration(self.fixture.spec)

    def test_rejects_dataset_binding_even_when_outer_sha_is_updated(self):
        dataset = json.loads(self.fixture.dataset_path.read_text(encoding="ascii"))
        dataset["train"][0]["stage_b_data_driven_manifest_sha256"] = "0" * 64
        _write_json(self.fixture.dataset_path, dataset)
        drifted_spec = replace(
            self.fixture.spec,
            dataset_sha256=_sha256(self.fixture.dataset_path),
        )
        self.fixture.base_values[
            "stage_b_data_driven_new_head_dataset_config_sha256"
        ] = drifted_spec.dataset_sha256
        self.fixture.write_base_config()
        with self.assertRaisesRegex(
            builder.NewHeadLRPreregistrationError,
            "training dataset row 0 contract drifted",
        ):
            builder.build_preregistration(drifted_spec)

    def test_rejects_receipt_hash_drift_and_preexisting_candidate_output(self):
        self.fixture.support_receipt_path.write_text(
            self.fixture.support_receipt_path.read_text(encoding="ascii") + " ",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            builder.NewHeadLRPreregistrationError,
            "support receipt SHA256 drifted",
        ):
            builder.build_preregistration(self.fixture.spec)

        self.fixture.support_receipt_path.write_text(
            self.fixture.support_receipt_path.read_text(encoding="ascii").rstrip()
            + "\n",
            encoding="ascii",
        )
        output_dir = self.fixture.spec.candidates[0].training_output_dir
        output_dir.mkdir(parents=True)
        with self.assertRaisesRegex(
            builder.NewHeadLRPreregistrationError,
            "must not exist before preregistration",
        ):
            builder.build_preregistration(self.fixture.spec)

    def test_atomic_writer_never_replaces_existing_or_concurrent_output(self):
        payload = builder.build_preregistration(self.fixture.spec)
        output = self.fixture.root / "selection/preregistration.json"
        builder.write_preregistration_new(payload, output)
        original = output.read_bytes()
        with self.assertRaisesRegex(
            builder.NewHeadLRPreregistrationError,
            "refusing to overwrite",
        ):
            builder.write_preregistration_new(payload, output)
        self.assertEqual(output.read_bytes(), original)

        concurrent = self.fixture.root / "selection/concurrent.json"

        def publish_winner(_source, destination):
            Path(destination).write_bytes(b"concurrent-winner\n")
            raise FileExistsError(destination)

        with mock.patch.object(
            builder, "_rename_noreplace", side_effect=publish_winner
        ):
            with self.assertRaisesRegex(
                builder.NewHeadLRPreregistrationError,
                "refusing concurrent overwrite",
            ):
                builder.write_preregistration_new(payload, concurrent)
        self.assertEqual(concurrent.read_bytes(), b"concurrent-winner\n")
        self.assertEqual(
            list(concurrent.parent.glob(f".{concurrent.name}.tmp-*")), []
        )

    def test_cli_supports_output_and_rejects_it_before_rebuilding(self):
        output = self.fixture.root / "cli/preregistration.json"
        with mock.patch.object(
            builder, "default_spec", return_value=self.fixture.spec
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(builder.main(["--output", str(output)]), 0)
        original = output.read_bytes()
        with mock.patch.object(
            builder,
            "default_spec",
            side_effect=AssertionError("existing output must be rejected first"),
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(builder.main(["--output", str(output)]), 2)
        self.assertIn("refusing to overwrite", errors.getvalue())
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
