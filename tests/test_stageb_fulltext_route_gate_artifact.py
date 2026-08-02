import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import stageb_fulltext_route_gate_artifact as gate


class FullTextRouteGateArtifactTest(unittest.TestCase):
    def _fixture(self, root: Path):
        baseline_checkpoint = root / "baseline.pth"
        baseline_checkpoint.write_bytes(b"baseline")
        route_path = root / "caption_route.json"
        route_path.write_text("{}\n", encoding="utf-8")
        route_payload = {
            "artifact_identity": {
                "algorithm": gate.IDENTITY_ALGORITHM,
                "sha256": "a" * 64,
            },
            "route": {
                "overrides": {
                    "person": gate.GATED_DESCRIPTOR_ID,
                    "bowl": "max_external_p05_w0046875_v1",
                }
            },
        }
        formal_path = root / "formal_v2.json"
        formal_unsigned = {
            "schema": "stageb-external-gdino-rank-transfer-formal-artifact-v2",
            "components": {
                "merged_external": {
                    "base_baseline_checkpoint": gate._file_record(
                        baseline_checkpoint
                    )
                }
            },
            "inference_contract": {
                "routing": {
                    "artifact_identity": route_payload["artifact_identity"]
                }
            },
        }
        formal_payload = {
            **formal_unsigned,
            "artifact_identity": {
                "algorithm": gate.IDENTITY_ALGORITHM,
                "sha256": gate.canonical_sha256(formal_unsigned),
            },
        }
        formal_path.write_text(json.dumps(formal_payload), encoding="utf-8")

        base_paths = {}
        routed_paths = {}
        manifest_paths = {}
        summary_rows = []
        for split in gate.VAL_SPLITS:
            manifest = root / f"{split}.manifest.jsonl"
            base_path = root / f"{split}.base.jsonl"
            routed_path = root / f"{split}.routed.jsonl"
            manifest_rows = []
            base_rows = []
            routed_rows = []
            phrases = ["p", "p two", "p two three", "p two three four", "bowl", "chair"]
            captions = ["person", "person", "person", "person", "bowl", "chair"]
            base_correct = [False, False, True, False, True, False]
            routed_correct = [True, True, False, False, True, False]
            for index, (phrase, caption) in enumerate(zip(phrases, captions)):
                manifest_rows.append(
                    {
                        "image_id": index,
                        "instances": [{"raw_phrase": phrase, "class_id": 1}],
                    }
                )
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )
            manifest_sha = gate._sha256(manifest)
            for index, caption in enumerate(captions):
                common = {
                    "schema": "stageb-eval-record-v1",
                    "task": "ref",
                    "split": split,
                    "manifest_index": index,
                    "manifest_sha256": manifest_sha,
                    "manifest_n": 6,
                    "valid": True,
                    "sample_id": f"{split}:{index}",
                    "image_id": index,
                    "ann_id": index,
                    "ref_id": index,
                    "sent_id": index,
                    "all_query_best_iou": 0.9,
                }
                base_rows.append(
                    {
                        **common,
                        "run_id": "baseline-run",
                        "correct50": base_correct[index],
                    }
                )
                routed_rows.append(
                    {
                        **common,
                        "run_id": "routed-run",
                        "correct50": routed_correct[index],
                        "canonical_class_norm": caption,
                        "caption_route_descriptor_id": route_payload["route"][
                            "overrides"
                        ].get(caption, gate.DEFAULT_DESCRIPTOR_ID),
                        "caption_route_selection_artifact_identity_sha256": "a" * 64,
                        "external_transfer_artifact_sha256": formal_payload[
                            "artifact_identity"
                        ]["sha256"],
                    }
                )
            base_path.write_text(
                "".join(json.dumps(row) + "\n" for row in base_rows),
                encoding="utf-8",
            )
            routed_path.write_text(
                "".join(json.dumps(row) + "\n" for row in routed_rows),
                encoding="utf-8",
            )
            base_paths[split] = base_path
            routed_paths[split] = routed_path
            manifest_paths[split] = manifest
            summary_rows.append(
                {
                    "checkpoint": str(baseline_checkpoint),
                    "dataset": split,
                    "num_expressions": 6,
                    "manifest_n": 6,
                    "manifest_sha256": manifest_sha,
                    "batch_size": 32,
                    "num_workers": 4,
                    "max_batches": 0,
                    "invalid_records": 0,
                    "records_jsonl": str(base_path),
                    "run_id": "baseline-run",
                    "acc50": sum(base_correct) / 6,
                }
            )
        baseline_summary = root / "baseline_summary.json"
        baseline_summary.write_text(
            json.dumps({"refcoco": summary_rows}), encoding="utf-8"
        )
        return {
            "base_record_paths": base_paths,
            "routed_record_paths": routed_paths,
            "manifest_paths": manifest_paths,
            "baseline_summary_path": baseline_summary,
            "caption_route_artifact_path": route_path,
            "formal_routed_artifact_path": formal_path,
        }, route_payload

    def _expected_n(self):
        return {split: 6 for split in gate.VAL_SPLITS}

    def test_pooled_validation_maximum_selects_two_tokens(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            gate, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            kwargs, route_payload = self._fixture(Path(temporary))
            with mock.patch.object(
                gate,
                "load_and_verify_caption_route_artifact",
                return_value=route_payload,
            ):
                payload = gate.build_fulltext_route_gate_artifact(**kwargs)
        predicate = payload["route"]["conditional_overrides"]["person"]["predicate"]
        self.assertEqual(predicate["max_tokens"], 2)
        self.assertEqual(
            payload["route"]["unconditional_overrides"],
            {"bowl": "max_external_p05_w0046875_v1"},
        )
        self.assertTrue(
            all(row["delta_correct"] == 2 for row in payload["validation_route_summary"].values())
        )
        self.assertFalse(payload["iteration_provenance"]["test_blind_holdout_claim"])

    def test_rehashed_threshold_tamper_fails_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            gate, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            root = Path(temporary)
            kwargs, route_payload = self._fixture(root)
            artifact = root / "gate.json"
            with mock.patch.object(
                gate,
                "load_and_verify_caption_route_artifact",
                return_value=route_payload,
            ):
                gate.create_fulltext_route_gate_artifact(artifact, **kwargs)
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                payload["route"]["conditional_overrides"]["person"]["predicate"][
                    "max_tokens"
                ] = 3
                unsigned = {
                    key: value for key, value in payload.items() if key != "artifact_identity"
                }
                payload["artifact_identity"]["sha256"] = gate.canonical_sha256(unsigned)
                artifact.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    gate.FullTextRouteGateArtifactError, "bound validation evidence"
                ):
                    gate.load_and_verify_fulltext_route_gate_artifact(artifact)

    def test_baseline_summary_metric_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            gate, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            kwargs, route_payload = self._fixture(Path(temporary))
            summary_path = kwargs["baseline_summary_path"]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["refcoco"][0]["acc50"] = 0.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with mock.patch.object(
                gate,
                "load_and_verify_caption_route_artifact",
                return_value=route_payload,
            ), self.assertRaisesRegex(
                gate.FullTextRouteGateArtifactError, "acc50 drifted"
            ):
                gate.build_fulltext_route_gate_artifact(**kwargs)

    def test_all_query_oracle_pairing_is_required(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            gate, "EXPECTED_MANIFEST_N", self._expected_n()
        ):
            kwargs, route_payload = self._fixture(Path(temporary))
            path = kwargs["routed_record_paths"][gate.VAL_SPLITS[0]]
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["all_query_best_iou"] = 0.8
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with mock.patch.object(
                gate,
                "load_and_verify_caption_route_artifact",
                return_value=route_payload,
            ), self.assertRaisesRegex(
                gate.FullTextRouteGateArtifactError, "changed all_query_best_iou"
            ):
                gate.build_fulltext_route_gate_artifact(**kwargs)

    def test_token_contract_and_unconditional_candidate_are_explicit(self):
        self.assertEqual(gate.full_expression_token_count(" Man's red-shirt, #2. "), 5)
        self.assertEqual(gate.THRESHOLD_CANDIDATES[-1], 64)


if __name__ == "__main__":
    unittest.main()
