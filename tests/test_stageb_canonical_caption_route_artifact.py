import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import stageb_canonical_caption_route_artifact as route


class CanonicalCaptionRouteArtifactTest(unittest.TestCase):
    def _write_fixture(self, root: Path):
        canonical_classes = root / "canonical_classes.json"
        canonical_classes.write_text(
            json.dumps(
                [
                    {"id": 1, "raw_name": "person", "norm_name": "person"},
                    {"id": 2, "raw_name": "bowl", "norm_name": "bowl"},
                ]
            ),
            encoding="utf-8",
        )
        source_files = {
            "patch_config": root / "patch.py",
            "patch_checkpoint": root / "patch.pth",
            "external_config": root / "external.py",
            "external_checkpoint": root / "external.pth",
        }
        for label, path in source_files.items():
            path.write_bytes(label.encode("ascii"))
        results = {}
        manifests = {}
        for split in route.VAL_SPLITS:
            manifest = root / f"{split}.jsonl"
            manifest_rows = []
            for index in range(110):
                caption = "person" if index < 100 else "bowl"
                manifest_rows.append(
                    {
                        "image_id": index,
                        "instances": [
                            {
                                "canonical_name": caption,
                                "class_id": 1 if caption == "person" else 2,
                                "raw_phrase": caption,
                            }
                        ],
                    }
                )
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )
            manifests[split] = manifest

            common = {
                "diagnostic_only": True,
                "formal_gate_eligible": False,
                "dataset": split,
                "num_expressions": 110,
                "max_images": 0,
                "max_batches": 0,
                "checkpoint": str(source_files["patch_checkpoint"]),
                "patch_checkpoint_sha256": route._sha256(
                    source_files["patch_checkpoint"]
                ),
                "patch_config": str(source_files["patch_config"]),
                "patch_config_sha256": route._sha256(source_files["patch_config"]),
                "external_gdino_checkpoint": str(
                    source_files["external_checkpoint"]
                ),
                "external_gdino_checkpoint_sha256": route._sha256(
                    source_files["external_checkpoint"]
                ),
                "external_gdino_config": str(source_files["external_config"]),
                "external_gdino_config_sha256": route._sha256(
                    source_files["external_config"]
                ),
                "external_gdino_query_count": 900,
                "batch_size": 32,
                "num_workers": 4,
                "seed_protocol": route.SEED_PROTOCOL,
                "seed": route.EXPECTED_SEEDS[split],
                "canonical_stage_a_caption_contract": dict(route.CAPTION_CONTRACT),
            }

            def groups(person_correct, bowl_correct):
                return {
                    "person": {
                        "num_expressions": 100,
                        "acc50": person_correct / 100,
                        "mean_iou_top1": 0.5,
                    },
                    "bowl": {
                        "num_expressions": 10,
                        "acc50": bowl_correct / 10,
                        "mean_iou_top1": 0.5,
                    },
                }

            rows = []
            base = {
                **common,
                **route.BASE_IDENTITY_FIELDS,
                "run_id": (
                    "fixture:diagnostic_external_gdino_base_identity="
                    "direct_global_argmax"
                ),
                "by_canonical_stage_a_caption": groups(50, 5),
                "acc50": 55 / 110,
            }
            rows.append(base)
            candidate_correct = {
                "max_patch_p025_w003125_v1": (54, 6),
                "max_external_p05_w0046875_v1": (53, 5),
                "top_query_patch_w0_v1": (50, 5),
                "top_query_external_w0_v1": (50, 5),
            }
            for descriptor_id in route.CANDIDATE_DESCRIPTOR_IDS:
                descriptor = route.DESCRIPTOR_REGISTRY[descriptor_id]
                person_correct, bowl_correct = candidate_correct[descriptor_id]
                rows.append(
                    {
                        **common,
                        "diagnostic_descriptor_kind": "external_gdino_rank_transfer",
                        "diagnostic_transfer_mode": descriptor["transfer_mode"],
                        "diagnostic_iou_power": descriptor["iou_power"],
                        "diagnostic_patch_weight": descriptor["patch_weight"],
                        "diagnostic_text_weight": descriptor["text_weight"],
                        "by_canonical_stage_a_caption": groups(
                            person_correct, bowl_correct
                        ),
                        "acc50": (person_correct + bowl_correct) / 110,
                    }
                )
            result = root / f"{split}.json"
            result.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
            results[split] = result
        return results, manifests, canonical_classes

    def _expected_n(self):
        return {split: 110 for split in route.VAL_SPLITS}

    def test_full_validation_evidence_freezes_only_eligible_caption(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            route, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            results, manifests, canonical_classes = self._write_fixture(Path(temporary))
            payload = route.build_caption_route_artifact(
                result_paths=results,
                manifest_paths=manifests,
                canonical_classes_path=canonical_classes,
            )
        self.assertEqual(
            payload["route"]["overrides"],
            {"person": "max_patch_p025_w003125_v1"},
        )
        self.assertEqual(payload["route"]["default_descriptor_id"], route.DEFAULT_DESCRIPTOR_ID)
        self.assertEqual(
            payload["caption_decisions"]["bowl"]["selected_descriptor_id"],
            route.DEFAULT_DESCRIPTOR_ID,
        )
        self.assertTrue(
            all(row["delta_correct"] == 4 for row in payload["validation_route_summary"].values())
        )

    def test_pilot_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            route, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            results, manifests, canonical_classes = self._write_fixture(Path(temporary))
            path = results[route.VAL_SPLITS[0]]
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows:
                row["max_images"] = 1024
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(route.CaptionRouteArtifactError, "max_images"):
                route.build_caption_route_artifact(
                    result_paths=results,
                    manifest_paths=manifests,
                    canonical_classes_path=canonical_classes,
                )

    def test_caption_counts_must_match_bound_manifest(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            route, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            results, manifests, canonical_classes = self._write_fixture(Path(temporary))
            path = results[route.VAL_SPLITS[1]]
            rows = json.loads(path.read_text(encoding="utf-8"))
            rows[0]["by_canonical_stage_a_caption"]["person"]["num_expressions"] = 99
            rows[0]["by_canonical_stage_a_caption"]["person"]["acc50"] = 50 / 99
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(route.CaptionRouteArtifactError, "caption counts"):
                route.build_caption_route_artifact(
                    result_paths=results,
                    manifest_paths=manifests,
                    canonical_classes_path=canonical_classes,
                )

    def test_rehashed_mapping_tamper_fails_rebuild_verification(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            route, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            root = Path(temporary)
            results, manifests, canonical_classes = self._write_fixture(root)
            artifact = root / "route.json"
            route.create_caption_route_artifact(
                artifact,
                result_paths=results,
                manifest_paths=manifests,
                canonical_classes_path=canonical_classes,
            )
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            payload["route"]["overrides"]["person"] = "top_query_patch_w0_v1"
            unsigned = {
                key: value for key, value in payload.items() if key != "artifact_identity"
            }
            payload["artifact_identity"]["sha256"] = route.canonical_sha256(unsigned)
            artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                route.CaptionRouteArtifactError, "bound validation evidence"
            ):
                route.load_and_verify_caption_route_artifact(artifact)

    def test_base_identity_contract_field_drift_fails(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            route, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            results, manifests, canonical_classes = self._write_fixture(Path(temporary))
            path = results[route.VAL_SPLITS[2]]
            rows = json.loads(path.read_text(encoding="utf-8"))
            rows[0]["uses_patch_top50_admission"] = True
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(route.CaptionRouteArtifactError, "exactly one row"):
                route.build_caption_route_artifact(
                    result_paths=results,
                    manifest_paths=manifests,
                    canonical_classes_path=canonical_classes,
                )

    def test_bound_source_file_drift_fails_verification(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            route, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            root = Path(temporary)
            results, manifests, canonical_classes = self._write_fixture(root)
            artifact = root / "route.json"
            payload = route.create_caption_route_artifact(
                artifact,
                result_paths=results,
                manifest_paths=manifests,
                canonical_classes_path=canonical_classes,
            )
            Path(payload["source_component_files"]["external_config"]["path"]).write_text(
                "drift", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                route.CaptionRouteArtifactError,
                "external_config SHA-256",
            ):
                route.load_and_verify_caption_route_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
