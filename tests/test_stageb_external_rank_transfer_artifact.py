import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tests.test_merge_stageb_gdino_adapter_eval import MergedEvalFixture
from tools import stageb_external_rank_transfer_artifact as artifact_tool
from tools import stageb_canonical_caption_route_artifact as route_tool
from tools import stageb_fulltext_route_gate_artifact as fulltext_gate_tool
from tools.merge_stageb_gdino_adapter_eval import (
    DEFAULT_EVAL_CONFIG,
    MergedEvalCheckpointError,
    create_merged_eval_checkpoint,
    verify_merged_eval_checkpoint,
)
from tools.stageb_external_rank_transfer_artifact import (
    ExternalRankTransferArtifactError,
    build_formal_fulltext_routed_transfer_artifact,
    build_formal_transfer_artifact,
    build_formal_routed_transfer_artifact,
    create_formal_fulltext_routed_transfer_artifact,
    create_formal_transfer_artifact,
    create_formal_routed_transfer_artifact,
    evaluator_settings_from_artifact,
    load_and_verify_formal_transfer_artifact,
)
from util.stageb_exact_topk_contract import canonical_sha256


class FormalTransferArtifactFixture:
    def __init__(self) -> None:
        self.merged = MergedEvalFixture()
        create_merged_eval_checkpoint(**self.merged.kwargs())
        self.patch_config = self.merged.config_dir / "cfg_stageb_v18_patch_fixture.py"
        self.patch_config.write_text(
            "from config.ablations.cfg_stageb_v18_strong_fpr_tail import *\n",
            encoding="utf-8",
        )
        self.patch_checkpoint = self.merged.work / "patch.pth"
        self._write_patch_checkpoint()
        self.artifact = self.merged.work / "formal-transfer.json"

    def close(self) -> None:
        self.merged.close()

    def _write_patch_checkpoint(self, *, topk: int = 50) -> None:
        torch.save(
            {
                "model": {
                    "backbone.weight": torch.arange(4, dtype=torch.float32),
                    "stage_b_fixed_text_scorer._score_contract_candidate_topk": torch.tensor(
                        topk
                    ),
                },
                "args": {"config_file": str(self.patch_config.resolve())},
            },
            self.patch_checkpoint,
        )

    def kwargs(self, **overrides):
        values = {
            "patch_config": self.patch_config,
            "patch_checkpoint": self.patch_checkpoint,
            "merged_external_config": DEFAULT_EVAL_CONFIG,
            "merged_external_checkpoint": self.merged.output,
            "source_rank_checkpoint": self.merged.rank_checkpoint,
            "source_confidence_checkpoint": self.merged.confidence_checkpoint,
            "base_baseline_checkpoint": self.merged.baseline_checkpoint,
            "mode": "max_score_iou_power",
            "iou_power": 0.5,
            "patch_weight": 0.046875,
            "text_weight": 1.0,
            "base_seed": 42,
        }
        values.update(overrides)
        return values


class StageBExternalRankTransferArtifactTest(unittest.TestCase):
    def setUp(self):
        self.fixture = FormalTransferArtifactFixture()

    def tearDown(self):
        self.fixture.close()

    def _route_selection_payload(self, path: Path, overrides=None):
        path.write_text('{"fixture": true}\n', encoding="utf-8")
        canonical_classes = path.with_name("canonical-classes.json")
        canonical_classes.write_text("[]\n", encoding="utf-8")
        patch_config = artifact_tool.file_record(self.fixture.patch_config)
        patch_checkpoint = artifact_tool.file_record(self.fixture.patch_checkpoint)
        external_config = artifact_tool.file_record(
            Path(DEFAULT_EVAL_CONFIG).resolve()
        )
        external_checkpoint = artifact_tool.file_record(self.fixture.merged.output)
        overrides = dict(overrides or {})
        return {
            "schema": route_tool.SCHEMA,
            "kind": route_tool.KIND,
            "source_identity": {
                "checkpoint": patch_checkpoint["path"],
                "patch_checkpoint_sha256": patch_checkpoint["sha256"],
                "patch_config": patch_config["path"],
                "patch_config_sha256": patch_config["sha256"],
                "external_gdino_checkpoint": external_checkpoint["path"],
                "external_gdino_checkpoint_sha256": external_checkpoint[
                    "sha256"
                ],
                "external_gdino_config": external_config["path"],
                "external_gdino_config_sha256": external_config["sha256"],
                "external_gdino_query_count": 900,
                "batch_size": 32,
                "num_workers": 4,
            },
            "source_component_files": {
                "patch_config": patch_config,
                "patch_checkpoint": patch_checkpoint,
                "external_config": external_config,
                "external_checkpoint": external_checkpoint,
            },
            "caption_contract": dict(route_tool.CAPTION_CONTRACT),
            "canonical_classes": artifact_tool.file_record(canonical_classes),
            "manifest_caption_audit_contract": dict(
                route_tool.MANIFEST_CAPTION_AUDIT_CONTRACT
            ),
            "descriptor_registry": dict(route_tool.DESCRIPTOR_REGISTRY),
            "descriptor_registry_sha256": route_tool.canonical_sha256(
                route_tool.DESCRIPTOR_REGISTRY
            ),
            "selection_contract": dict(route_tool.SELECTION_CONTRACT),
            "selection_contract_sha256": route_tool.canonical_sha256(
                route_tool.SELECTION_CONTRACT
            ),
            "route": {
                "default_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
                "overrides": overrides,
                "override_count": len(overrides),
                "caption_count": max(1, len(overrides)),
            },
            "artifact_identity": {
                "algorithm": route_tool.IDENTITY_ALGORITHM,
                "sha256": "9" * 64,
            },
        }

    def _fulltext_gate_payload(
        self,
        path: Path,
        *,
        routed_v2_path: Path,
        routed_v2_payload,
        max_tokens: int = 7,
    ):
        path.write_text('{"fixture": "fulltext-gate"}\n', encoding="utf-8")
        routing = routed_v2_payload["inference_contract"]["routing"]
        overrides = dict(routing["policy"]["overrides"])
        person_descriptor = overrides.pop("person")
        return {
            "schema": fulltext_gate_tool.SCHEMA,
            "kind": fulltext_gate_tool.KIND,
            "artifact_identity": {
                "algorithm": fulltext_gate_tool.IDENTITY_ALGORITHM,
                "sha256": "8" * 64,
            },
            "selection_contract": dict(fulltext_gate_tool.SELECTION_CONTRACT),
            "selection_contract_sha256": canonical_sha256(
                fulltext_gate_tool.SELECTION_CONTRACT
            ),
            "formal_routed_artifact": artifact_tool.file_record(routed_v2_path),
            "formal_routed_artifact_identity": dict(
                routed_v2_payload["artifact_identity"]
            ),
            "caption_route_artifact": dict(routing["artifact"]),
            "caption_route_artifact_identity": dict(
                routing["artifact_identity"]
            ),
            "baseline_checkpoint": dict(
                routed_v2_payload["components"]["merged_external"][
                    "base_baseline_checkpoint"
                ]
            ),
            "route": {
                "default_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
                "unconditional_overrides": overrides,
                "conditional_overrides": {
                    "person": {
                        "descriptor_id": person_descriptor,
                        "fallback_descriptor_id": route_tool.DEFAULT_DESCRIPTOR_ID,
                        "predicate": {
                            "kind": "full_expression_lexical_token_count_lte",
                            "max_tokens": max_tokens,
                            "token_count_contract": dict(
                                fulltext_gate_tool.TOKEN_COUNT_CONTRACT
                            ),
                        },
                    }
                },
            },
            "validation_route_summary": {},
        }

    def _routed_v3_inputs(self):
        route_path = self.fixture.merged.work / "caption-route-v3.json"
        selection = self._route_selection_payload(
            route_path,
            {
                "bowl": "max_external_p05_w0046875_v1",
                "person": "max_patch_p025_w003125_v1",
            },
        )
        routed_v2_path = self.fixture.merged.work / "formal-routed-v2-for-v3.json"
        kwargs = self.fixture.kwargs()
        for key in ("mode", "iou_power", "patch_weight", "text_weight"):
            kwargs.pop(key)
        kwargs["caption_route_artifact"] = route_path
        with mock.patch.object(
            artifact_tool,
            "load_and_verify_caption_route_artifact",
            return_value=selection,
        ):
            routed_v2_payload = create_formal_routed_transfer_artifact(
                routed_v2_path, **kwargs
            )
        gate_path = self.fixture.merged.work / "fulltext-gate-v2.json"
        gate = self._fulltext_gate_payload(
            gate_path,
            routed_v2_path=routed_v2_path,
            routed_v2_payload=routed_v2_payload,
        )
        return selection, routed_v2_path, routed_v2_payload, gate_path, gate

    def test_create_verify_and_evaluator_settings_bind_single_point(self):
        payload = create_formal_transfer_artifact(
            self.fixture.artifact, **self.fixture.kwargs()
        )
        self.assertEqual(
            payload["artifact_identity"]["sha256"],
            canonical_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "artifact_identity"
                }
            ),
        )
        verified = load_and_verify_formal_transfer_artifact(self.fixture.artifact)
        self.assertEqual(verified, payload)
        settings = evaluator_settings_from_artifact(self.fixture.artifact)
        self.assertEqual(len(settings["fixed_grid"]), 1)
        self.assertEqual(settings["candidate_topk"], 50)
        self.assertEqual(settings["external_query_count"], 900)
        self.assertEqual(
            settings["fixed_grid"][0],
            {
                "transfer_mode": "max_score_iou_power",
                "iou_power": 0.5,
                "patch_weight": 0.046875,
                "text_weight": 1.0,
            },
        )
        self.assertEqual(
            settings["evaluation_protocol"]["split_seeds"]["refcocog_val"],
            600042,
        )

    def test_artifact_contract_tamper_fails_identity_check(self):
        create_formal_transfer_artifact(
            self.fixture.artifact, **self.fixture.kwargs()
        )
        payload = json.loads(self.fixture.artifact.read_text(encoding="utf-8"))
        payload["inference_contract"]["transfer"]["patch_weight"] = 0.0
        self.fixture.artifact.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(
            ExternalRankTransferArtifactError, "identity mismatch"
        ):
            load_and_verify_formal_transfer_artifact(self.fixture.artifact)

    def test_artifact_file_drift_during_verification_fails_closed(self):
        payload = create_formal_transfer_artifact(
            self.fixture.artifact, **self.fixture.kwargs()
        )

        def mutate_artifact(**_kwargs):
            with self.fixture.artifact.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            return payload

        with (
            mock.patch.object(
                artifact_tool,
                "build_formal_transfer_artifact",
                side_effect=mutate_artifact,
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError, "changed during validation"
            ),
        ):
            load_and_verify_formal_transfer_artifact(self.fixture.artifact)

    def test_patch_file_and_merged_tensor_tamper_fail_closed(self):
        create_formal_transfer_artifact(
            self.fixture.artifact, **self.fixture.kwargs()
        )
        patch = torch.load(
            self.fixture.patch_checkpoint, map_location="cpu", weights_only=False
        )
        patch["model"]["backbone.weight"][0] += 1
        torch.save(patch, self.fixture.patch_checkpoint)
        with self.assertRaises(ExternalRankTransferArtifactError):
            load_and_verify_formal_transfer_artifact(self.fixture.artifact)

        self.fixture.artifact.unlink()
        self.fixture._write_patch_checkpoint()
        merged = torch.load(
            self.fixture.merged.output, map_location="cpu", weights_only=False
        )
        merged["model"]["backbone.weight"][0, 0] += 1
        torch.save(merged, self.fixture.merged.output)
        with self.assertRaisesRegex(
            ExternalRankTransferArtifactError,
            "canonical merged checkpoint verification failed|tensor_sha256",
        ):
            build_formal_transfer_artifact(**self.fixture.kwargs())

    def test_canonical_contract_tamper_is_rejected_by_artifact_builder(self):
        for field, value in (
            (("contract", "branch_selection", "rank"), "confidence"),
            (("contract", "synthetic_input_contract", "seed"), 123456),
        ):
            with self.subTest(field=field):
                merged = torch.load(
                    self.fixture.merged.output,
                    map_location="cpu",
                    weights_only=False,
                )
                target = merged
                for key in field[:-1]:
                    target = target[key]
                target[field[-1]] = value
                tampered = self.fixture.merged.work / (
                    "tampered-" + "-".join(field) + ".pth"
                )
                torch.save(merged, tampered)

                with self.assertRaises(MergedEvalCheckpointError):
                    verify_merged_eval_checkpoint(tampered)
                with self.assertRaisesRegex(
                    ExternalRankTransferArtifactError,
                    "canonical merged checkpoint verification failed",
                ):
                    build_formal_transfer_artifact(
                        **self.fixture.kwargs(merged_external_checkpoint=tampered)
                    )

    def test_source_and_config_tamper_fail_closed(self):
        create_formal_transfer_artifact(
            self.fixture.artifact, **self.fixture.kwargs()
        )
        original_config = self.fixture.patch_config.read_text(encoding="utf-8")
        self.fixture.patch_config.write_text(
            original_config + "\n", encoding="utf-8"
        )
        with self.assertRaises(ExternalRankTransferArtifactError):
            load_and_verify_formal_transfer_artifact(self.fixture.artifact)
        self.fixture.patch_config.write_text(original_config, encoding="utf-8")

        rank = torch.load(
            self.fixture.merged.rank_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        rank["model"]["stage_b_gdino_score_adapter.rank_output.bias"] += 1
        torch.save(rank, self.fixture.merged.rank_checkpoint)
        with self.assertRaises(ExternalRankTransferArtifactError):
            load_and_verify_formal_transfer_artifact(self.fixture.artifact)

    def test_invalid_transfer_mode_power_and_topk_are_rejected(self):
        with self.assertRaisesRegex(
            ExternalRankTransferArtifactError, "does not accept an IoU power"
        ):
            build_formal_transfer_artifact(
                **self.fixture.kwargs(
                    mode="top_query_nearest_candidate", iou_power=0.5
                )
            )
        top_query = build_formal_transfer_artifact(
            **self.fixture.kwargs(
                mode="top_query_nearest_candidate", iou_power=None
            )
        )
        self.assertIsNone(
            top_query["inference_contract"]["transfer"]["iou_power"]
        )
        self.fixture._write_patch_checkpoint(topk=49)
        with self.assertRaisesRegex(ExternalRankTransferArtifactError, "Top50"):
            build_formal_transfer_artifact(**self.fixture.kwargs())

    def test_create_verify_routed_v2_binds_selection_and_exact_registry(self):
        route_path = self.fixture.merged.work / "caption-route.json"
        selection = self._route_selection_payload(
            route_path,
            {"person": "max_patch_p025_w003125_v1"},
        )
        output = self.fixture.merged.work / "formal-routed-v2.json"
        kwargs = self.fixture.kwargs()
        for key in ("mode", "iou_power", "patch_weight", "text_weight"):
            kwargs.pop(key)
        kwargs["caption_route_artifact"] = route_path
        with mock.patch.object(
            artifact_tool,
            "load_and_verify_caption_route_artifact",
            return_value=selection,
        ) as verify_selection:
            payload = create_formal_routed_transfer_artifact(output, **kwargs)
            verified = load_and_verify_formal_transfer_artifact(output)
            settings = evaluator_settings_from_artifact(output)
        self.assertEqual(verified, payload)
        self.assertGreaterEqual(verify_selection.call_count, 3)
        self.assertEqual(payload["schema"], artifact_tool.ROUTED_V2_SCHEMA)
        self.assertEqual(settings["formal_artifact_version"], 2)
        self.assertEqual(
            settings["route_default_descriptor_id"],
            route_tool.DEFAULT_DESCRIPTOR_ID,
        )
        self.assertEqual(
            settings["route_overrides"],
            {"person": "max_patch_p025_w003125_v1"},
        )
        self.assertEqual(
            set(settings["descriptor_grid_by_id"]),
            set(route_tool.CANDIDATE_DESCRIPTOR_IDS),
        )
        self.assertEqual(len(settings["fixed_grid"]), 4)
        with self.assertRaisesRegex(
            ExternalRankTransferArtifactError, "frozen base seed 42"
        ):
            build_formal_routed_transfer_artifact(
                **{**kwargs, "base_seed": 43}
            )

    def test_routed_v2_rehashed_mapping_and_selection_rebuild_tamper_fail(self):
        route_path = self.fixture.merged.work / "caption-route.json"
        selection = self._route_selection_payload(
            route_path,
            {"person": "max_patch_p025_w003125_v1"},
        )
        output = self.fixture.merged.work / "formal-routed-v2.json"
        kwargs = self.fixture.kwargs()
        for key in ("mode", "iou_power", "patch_weight", "text_weight"):
            kwargs.pop(key)
        kwargs["caption_route_artifact"] = route_path
        with mock.patch.object(
            artifact_tool,
            "load_and_verify_caption_route_artifact",
            return_value=selection,
        ):
            create_formal_routed_transfer_artifact(output, **kwargs)

        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["inference_contract"]["routing"]["policy"]["overrides"] = {
            "person": "top_query_patch_w0_v1"
        }
        unsigned = {
            key: value for key, value in payload.items() if key != "artifact_identity"
        }
        payload["artifact_identity"]["sha256"] = canonical_sha256(unsigned)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError, "bound inputs"
            ),
        ):
            load_and_verify_formal_transfer_artifact(output)

        output.unlink()
        with mock.patch.object(
            artifact_tool,
            "load_and_verify_caption_route_artifact",
            return_value=selection,
        ):
            create_formal_routed_transfer_artifact(output, **kwargs)
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["inference_contract"]["routing"]["descriptor_registry"][
            "max_patch_p025_w003125_v1"
        ]["patch_weight"] = 0.5
        unsigned = {
            key: value for key, value in payload.items() if key != "artifact_identity"
        }
        payload["artifact_identity"]["sha256"] = canonical_sha256(unsigned)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError, "bound inputs"
            ),
        ):
            load_and_verify_formal_transfer_artifact(output)

        output.unlink()
        with mock.patch.object(
            artifact_tool,
            "load_and_verify_caption_route_artifact",
            return_value=selection,
        ):
            create_formal_routed_transfer_artifact(output, **kwargs)
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                side_effect=route_tool.CaptionRouteArtifactError(
                    "bound validation evidence drifted"
                ),
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError,
                "selection verification failed",
            ),
        ):
            load_and_verify_formal_transfer_artifact(output)

    def test_create_verify_routed_v3_binds_gate_and_exposes_conditional_policy(self):
        selection, routed_v2_path, routed_v2_payload, gate_path, gate = (
            self._routed_v3_inputs()
        )
        output = self.fixture.merged.work / "formal-routed-v3.json"
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            mock.patch.object(
                artifact_tool,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=gate,
            ),
        ):
            payload = create_formal_fulltext_routed_transfer_artifact(
                output,
                routed_v2_artifact=routed_v2_path,
                fulltext_route_gate_artifact=gate_path,
            )
            verified = load_and_verify_formal_transfer_artifact(output)
            settings = evaluator_settings_from_artifact(output)

        self.assertEqual(verified, payload)
        self.assertEqual(payload["schema"], artifact_tool.ROUTED_V3_SCHEMA)
        self.assertEqual(settings["formal_artifact_version"], 3)
        self.assertEqual(
            settings["evaluation_protocol"]["batch_size"], 32
        )
        self.assertEqual(
            settings["evaluation_protocol"]["num_workers"], 4
        )
        self.assertIs(settings["evaluation_protocol"]["amp"], True)
        self.assertEqual(
            settings["route_unconditional_overrides"],
            {"bowl": "max_external_p05_w0046875_v1"},
        )
        person = settings["route_conditional_overrides"]["person"]
        self.assertEqual(person["descriptor_id"], "max_patch_p025_w003125_v1")
        self.assertEqual(person["predicate"]["max_tokens"], 7)
        self.assertEqual(
            person["predicate"]["token_count_contract"],
            fulltext_gate_tool.TOKEN_COUNT_CONTRACT,
        )
        self.assertEqual(
            settings["canonical_route_selection"],
            routed_v2_payload["inference_contract"]["routing"],
        )
        self.assertEqual(
            settings["routed_v2_artifact_identity"],
            routed_v2_payload["artifact_identity"],
        )
        self.assertEqual(
            settings["fulltext_route_gate_artifact_identity"],
            gate["artifact_identity"],
        )

    def test_routed_v3_rehashed_policy_and_bound_gate_drift_fail_closed(self):
        selection, routed_v2_path, _, gate_path, gate = self._routed_v3_inputs()
        output = self.fixture.merged.work / "formal-routed-v3.json"
        patches = (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            mock.patch.object(
                artifact_tool,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=gate,
            ),
        )
        with patches[0], patches[1]:
            create_formal_fulltext_routed_transfer_artifact(
                output,
                routed_v2_artifact=routed_v2_path,
                fulltext_route_gate_artifact=gate_path,
            )

        payload = json.loads(output.read_text(encoding="utf-8"))
        policy = payload["inference_contract"]["routing"]["policy"]
        policy["conditional_overrides"]["person"]["predicate"]["max_tokens"] = 8
        payload["inference_contract"]["routing"]["policy_sha256"] = canonical_sha256(
            policy
        )
        unsigned = {
            key: value for key, value in payload.items() if key != "artifact_identity"
        }
        payload["artifact_identity"]["sha256"] = canonical_sha256(unsigned)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            mock.patch.object(
                artifact_tool,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=gate,
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError, "bound inputs"
            ),
        ):
            load_and_verify_formal_transfer_artifact(output)

        output.unlink()
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            mock.patch.object(
                artifact_tool,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=gate,
            ),
        ):
            create_formal_fulltext_routed_transfer_artifact(
                output,
                routed_v2_artifact=routed_v2_path,
                fulltext_route_gate_artifact=gate_path,
            )
        gate_path.write_text('{"fixture": "drifted"}\n', encoding="utf-8")
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            mock.patch.object(
                artifact_tool,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=gate,
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError, "bound inputs"
            ),
        ):
            load_and_verify_formal_transfer_artifact(output)

    def test_routed_v3_rejects_gate_binding_and_token_contract_drift(self):
        selection, routed_v2_path, _, gate_path, gate = self._routed_v3_inputs()
        bad_binding = json.loads(json.dumps(gate))
        bad_binding["formal_routed_artifact_identity"]["sha256"] = "7" * 64
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            mock.patch.object(
                artifact_tool,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=bad_binding,
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError, "not bound"
            ),
        ):
            build_formal_fulltext_routed_transfer_artifact(
                routed_v2_artifact=routed_v2_path,
                fulltext_route_gate_artifact=gate_path,
            )

        bad_token_contract = json.loads(json.dumps(gate))
        bad_token_contract["route"]["conditional_overrides"]["person"][
            "predicate"
        ]["token_count_contract"]["normalization"] = "drifted"
        with (
            mock.patch.object(
                artifact_tool,
                "load_and_verify_caption_route_artifact",
                return_value=selection,
            ),
            mock.patch.object(
                artifact_tool,
                "load_and_verify_fulltext_route_gate_artifact",
                return_value=bad_token_contract,
            ),
            self.assertRaisesRegex(
                ExternalRankTransferArtifactError, "predicate contract drifted"
            ),
        ):
            build_formal_fulltext_routed_transfer_artifact(
                routed_v2_artifact=routed_v2_path,
                fulltext_route_gate_artifact=gate_path,
            )


if __name__ == "__main__":
    unittest.main()
