import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.GroundingDINO.stage_b_gdino_score_adapter import (
    StageBGDINOScoreAdapterCriterion,
)
from tools.make_stageb_gdino_adapter_p0 import build_adapter_from_config
from tools.stageb_composite_artifact import (
    CompositeArtifactError,
    REPO_ROOT,
    _confidence_expected_config,
    build_composite_artifact,
    create_composite_artifact,
    verify_composite_artifact,
)
from util.slconfig import SLConfig
from util.stageb_exact_topk_contract import canonical_sha256, sha256_file


class CompositeFixture:
    def __init__(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.work_root = Path(self.work.name)
        self.config_dir_context = tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "config" / "ablations"
        )
        self.config_dir = Path(self.config_dir_context.name)
        self.rank_config = self.config_dir / "cfg_stageb_v18_fixture.py"
        self.rank_config.write_text(
            "from config.ablations.cfg_stageb_v18_strong_fpr_tail import *\n",
            encoding="utf-8",
        )
        self.confidence_config = (
            self.config_dir / "cfg_stageb_gdino_score_adapter_semantic_verified.py"
        )
        self.confidence_config.write_text(
            "from config.ablations.cfg_stageb_gdino_score_adapter_semantic_verified import *\n",
            encoding="utf-8",
        )

        self.rank_checkpoint = self.work_root / "rank.pth"
        self.stagea_checkpoint = self.work_root / "stagea.pth"
        self.warm_source = self.work_root / "warm_source.pth"
        self.confidence_checkpoint = self.work_root / "confidence.pth"
        self.baseline_checkpoint = self.work_root / "baseline.pth"
        self.artifact = self.work_root / "composite.json"
        self.rank_base_state = {
            "backbone.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "patch_encoder.stub.weight": torch.tensor([[1.0, 2.0]]),
            "query_proj_for_patch.weight": torch.eye(2),
            "query_proj_for_patch.bias": torch.zeros(2),
            "patch_logit_scale": torch.tensor(2.5),
        }
        self._write_rank_inputs()
        self._write_confidence_inputs()

    def close(self) -> None:
        self.config_dir_context.cleanup()
        self.work.cleanup()

    def _rank_scorer_state(self):
        prefix = "stage_b_fixed_text_scorer."
        state = {}
        for layer in range(3):
            suffix = f"layers.{layer}.stub.weight"
            state[prefix + "decoder." + suffix] = torch.full((2, 2), layer + 1.0)
            state[prefix + "confidence_decoder." + suffix] = torch.full(
                (2, 2), layer + 0.5
            )
        state[prefix + "decoder.ref_point_head.stub.weight"] = torch.ones(2, 2)
        state[prefix + "confidence_decoder.ref_point_head.stub.weight"] = torch.ones(
            2, 2
        )
        state.update(
            {
                prefix + "validity_head.0.weight": torch.ones(2, 2),
                prefix + "validity_head.0.bias": torch.zeros(2),
                prefix + "validity_head.2.weight": torch.ones(1, 2),
                prefix + "validity_head.2.bias": torch.ones(1),
                prefix + "_score_contract_version": torch.tensor(4),
                prefix
                + "_score_contract_decoupled_confidence": torch.tensor(True),
                prefix
                + "_score_contract_validity_pool_temperature": torch.tensor(0.2),
                prefix + "_score_contract_patch_rank_fusion": torch.tensor(True),
                prefix + "_score_contract_patch_rank_weight": torch.tensor(1.0),
                prefix + "_score_contract_exclude_canonical": torch.tensor(False),
                prefix + "_score_contract_candidate_topk": torch.tensor(50),
                prefix
                + "_score_contract_confidence_output_mode": torch.tensor(1),
            }
        )
        return state

    def _write_rank_inputs(self) -> None:
        torch.save({"model": copy.deepcopy(self.rank_base_state)}, self.stagea_checkpoint)
        torch.save({"model": {"decoder.stub": torch.ones(1)}}, self.warm_source)
        warm_audit = {
            "schema": "stage_b_v15_scorer_init/v1",
            "status": "applied",
            "requested_source_path": str(self.warm_source),
            "resolved_source_path": str(self.warm_source.resolve()),
            "source_sha256": sha256_file(self.warm_source),
            "source_size_bytes": self.warm_source.stat().st_size,
            "source_decoder_num_layers": 6,
            "selected_source_layer_indices": [3, 4, 5],
            "loaded_num_layers": 3,
            "loaded_tensor_count": 8,
            "loaded_components": [
                "decoder.layers[-N:]",
                "decoder.ref_point_head",
                "decoder.norm",
            ],
        }
        model = copy.deepcopy(self.rank_base_state)
        model.update(self._rank_scorer_state())
        torch.save(
            {
                "model": model,
                "args": {
                    "config_file": str(self.rank_config.resolve()),
                    "stage_b_v15_scorer_init_audit": warm_audit,
                },
                "epoch": 0,
                "iteration": 500,
                "epoch_finished": False,
                "checkpoint_reason": "max_train_iters",
            },
            self.rank_checkpoint,
        )

    def _confidence_args(self):
        cfg = SLConfig.fromfile(str(self.confidence_config))
        args = {
            key: getattr(cfg, key) for key in _confidence_expected_config()
        }
        args["config_file"] = str(self.confidence_config.resolve())
        return args

    def _confidence_criterion_state(self, *, count=512, pointer=0):
        criterion = StageBGDINOScoreAdapterCriterion(
            tn_scope="image_global_topk_verified",
            train_mode="confidence_only",
            confidence_objective="detached_recent_q05_trust",
            positive_iou_threshold=0.5,
            negative_iou_threshold=0.5,
            rank_weight=0.0,
            confidence_weight=1.0,
            paired_margin_weight=0.25,
            paired_margin=0.05,
            positive_trust_margin=0.02,
            positive_trust_weight=1.0,
            queue_size=512,
            queue_min_count=256,
        )
        criterion.fpr_positive_queue.copy_(torch.linspace(0.1, 0.9, 512))
        criterion.fpr_negative_queue.copy_(torch.linspace(0.0, 0.8, 512))
        criterion.fpr_queue_count.fill_(count)
        criterion.fpr_queue_ptr.fill_(pointer)
        return criterion.state_dict()

    def _write_confidence_inputs(
        self,
        *,
        drop_adapter_key=None,
        drift_base=False,
        criterion_state="default",
    ) -> None:
        baseline_state = {"backbone.weight": torch.arange(6.0).reshape(2, 3)}
        torch.save({"model": copy.deepcopy(baseline_state)}, self.baseline_checkpoint)
        adapter = build_adapter_from_config(self.confidence_config, seed=7)
        with torch.no_grad():
            adapter.confidence_gate[-1].bias.fill_(0.25)
        model = copy.deepcopy(baseline_state)
        model.update(
            {
                "stage_b_gdino_score_adapter." + key: value.detach().clone()
                for key, value in adapter.state_dict().items()
            }
        )
        if drop_adapter_key is not None:
            model.pop(drop_adapter_key)
        if drift_base:
            model["backbone.weight"] = model["backbone.weight"] + 1.0
        if criterion_state == "default":
            criterion_state = self._confidence_criterion_state()
        torch.save(
            {
                "model": model,
                "criterion": criterion_state,
                "args": self._confidence_args(),
                "epoch": 0,
                "iteration": 100,
                "epoch_finished": False,
                "checkpoint_reason": "max_train_iters",
            },
            self.confidence_checkpoint,
        )

    def build_kwargs(self):
        return {
            "rank_checkpoint": self.rank_checkpoint,
            "rank_config": self.rank_config,
            "rank_stagea_checkpoint": self.stagea_checkpoint,
            "confidence_checkpoint": self.confidence_checkpoint,
            "confidence_config": self.confidence_config,
            "confidence_baseline_checkpoint": self.baseline_checkpoint,
        }


class StageBCompositeArtifactTest(unittest.TestCase):
    def setUp(self):
        self.fixture = CompositeFixture()

    def tearDown(self):
        self.fixture.close()

    def test_create_and_verify_bind_every_component(self):
        payload = create_composite_artifact(
            self.fixture.artifact, **self.fixture.build_kwargs()
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
        self.assertTrue(Path(payload["rank"]["checkpoint"]["path"]).is_absolute())
        self.assertGreater(len(payload["rank"]["config"]["import_chain"]), 1)
        self.assertGreater(len(payload["confidence"]["config"]["import_chain"]), 1)
        self.assertEqual(
            payload["confidence"]["role_contract"]["confidence_query_reduction"],
            "max_over_all_900_queries",
        )
        self.assertEqual(payload["rank"]["role_contract"]["candidate_topk"], 50)
        receipt = verify_composite_artifact(self.fixture.artifact)
        self.assertEqual(
            receipt["artifact_identity"], payload["artifact_identity"]
        )

    def test_role_contract_tamper_is_rejected_even_if_identity_is_recomputed(self):
        create_composite_artifact(self.fixture.artifact, **self.fixture.build_kwargs())
        payload = json.loads(self.fixture.artifact.read_text(encoding="utf-8"))
        payload["confidence"]["role_contract"][
            "confidence_query_reduction"
        ] = "max_over_top3_queries"
        payload["artifact_identity"]["sha256"] = canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "artifact_identity"
            }
        )
        self.fixture.artifact.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(CompositeArtifactError, "no longer exactly matches"):
            verify_composite_artifact(self.fixture.artifact)

    def test_changed_config_or_warmstart_source_is_rejected(self):
        create_composite_artifact(self.fixture.artifact, **self.fixture.build_kwargs())
        self.fixture.rank_config.write_text(
            self.fixture.rank_config.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(CompositeArtifactError):
            verify_composite_artifact(self.fixture.artifact)

        self.fixture.artifact.unlink()
        self.fixture._write_rank_inputs()
        create_composite_artifact(self.fixture.artifact, **self.fixture.build_kwargs())
        with self.fixture.warm_source.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(CompositeArtifactError, "warm-start source"):
            verify_composite_artifact(self.fixture.artifact)

    def test_same_checkpoint_and_swapped_roles_are_rejected(self):
        kwargs = self.fixture.build_kwargs()
        kwargs["confidence_checkpoint"] = self.fixture.rank_checkpoint
        with self.assertRaisesRegex(CompositeArtifactError, "same checkpoint"):
            build_composite_artifact(**kwargs)

        kwargs = self.fixture.build_kwargs()
        kwargs.update(
            {
                "rank_checkpoint": self.fixture.confidence_checkpoint,
                "rank_config": self.fixture.confidence_config,
                "confidence_checkpoint": self.fixture.rank_checkpoint,
                "confidence_config": self.fixture.rank_config,
            }
        )
        with self.assertRaises(CompositeArtifactError):
            build_composite_artifact(**kwargs)

    def test_incomplete_adapter_is_rejected_by_existing_validator(self):
        key = "stage_b_gdino_score_adapter.confidence_gate.4.bias"
        self.fixture._write_confidence_inputs(drop_adapter_key=key)
        with self.assertRaisesRegex(CompositeArtifactError, "adapter state is incomplete"):
            build_composite_artifact(**self.fixture.build_kwargs())

    def test_confidence_base_tensor_drift_is_rejected(self):
        self.fixture._write_confidence_inputs(drift_base=True)
        with self.assertRaisesRegex(CompositeArtifactError, "drifted"):
            build_composite_artifact(**self.fixture.build_kwargs())

    def test_missing_cold_and_nonfinite_criterion_states_are_rejected(self):
        self.fixture._write_confidence_inputs(criterion_state=None)
        with self.assertRaisesRegex(CompositeArtifactError, "missing required"):
            build_composite_artifact(**self.fixture.build_kwargs())

        cold = self.fixture._confidence_criterion_state(count=128, pointer=128)
        self.fixture._write_confidence_inputs(criterion_state=cold)
        with self.assertRaisesRegex(CompositeArtifactError, "not warm"):
            build_composite_artifact(**self.fixture.build_kwargs())

        invalid = self.fixture._confidence_criterion_state()
        invalid["fpr_negative_queue"][0] = torch.nan
        self.fixture._write_confidence_inputs(criterion_state=invalid)
        with self.assertRaisesRegex(CompositeArtifactError, "finite"):
            build_composite_artifact(**self.fixture.build_kwargs())

    def test_wrong_pooling_and_rank_patch_contracts_are_rejected(self):
        self.fixture.confidence_config.write_text(
            self.fixture.confidence_config.read_text(encoding="utf-8")
            + "stage_b_gdino_gate_pool_temperature = 0.1\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CompositeArtifactError, "pool_temperature"):
            build_composite_artifact(**self.fixture.build_kwargs())

        self.fixture.confidence_config.write_text(
            "from config.ablations.cfg_stageb_gdino_score_adapter_semantic_verified import *\n",
            encoding="utf-8",
        )
        self.fixture.rank_config.write_text(
            self.fixture.rank_config.read_text(encoding="utf-8")
            + "stage_b_v15_patch_rank_weight = 0.5\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(CompositeArtifactError, "patch_rank_weight"):
            build_composite_artifact(**self.fixture.build_kwargs())

    def test_stagea_tensor_drift_and_incomplete_rank_scorer_are_rejected(self):
        stagea = torch.load(self.fixture.stagea_checkpoint, weights_only=False)
        stagea["model"]["backbone.weight"] += 1.0
        torch.save(stagea, self.fixture.stagea_checkpoint)
        with self.assertRaisesRegex(CompositeArtifactError, "changed frozen Stage-A"):
            build_composite_artifact(**self.fixture.build_kwargs())

        self.fixture._write_rank_inputs()
        rank = torch.load(self.fixture.rank_checkpoint, weights_only=False)
        rank["model"].pop(
            "stage_b_fixed_text_scorer.confidence_decoder.layers.2.stub.weight"
        )
        torch.save(rank, self.fixture.rank_checkpoint)
        with self.assertRaisesRegex(CompositeArtifactError, "structurally identical"):
            build_composite_artifact(**self.fixture.build_kwargs())


if __name__ == "__main__":
    unittest.main()
