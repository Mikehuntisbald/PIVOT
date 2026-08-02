import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools import eval_text_groundingdino_refcoco_tn as text_eval
from models.GroundingDINO.stage_b_gdino_score_adapter import (
    StageBGDINOScoreAdapterCriterion,
)
from tools.make_stageb_gdino_adapter_p0 import build_adapter_from_config
from tools.merge_stageb_gdino_adapter_eval import (
    ADAPTER_PREFIX,
    ARCHITECTURE_FIELDS,
    DEFAULT_EVAL_CONFIG,
    ROLE_CONFIG,
    MergedEvalCheckpointError,
    create_merged_eval_checkpoint,
    verify_merged_eval_checkpoint,
)
from tools.stageb_gdino_adapter_probe_audit import sha256_file
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


class MergedEvalFixture:
    def __init__(self) -> None:
        self.work_context = tempfile.TemporaryDirectory()
        self.work = Path(self.work_context.name)
        self.config_context = tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "config" / "ablations"
        )
        self.config_dir = Path(self.config_context.name)
        self.rank_config = self.config_dir / "cfg_rank_fixture.py"
        self.rank_config.write_text(
            "from config.ablations.cfg_stageb_gdino_score_adapter_rank_three_ref import *\n",
            encoding="utf-8",
        )
        self.confidence_config = self.config_dir / "cfg_confidence_fixture.py"
        self.confidence_config.write_text(
            "from config.ablations.cfg_stageb_gdino_score_adapter_semantic_verified import *\n",
            encoding="utf-8",
        )
        self.rank_checkpoint = self.work / "rank.pth"
        self.confidence_checkpoint = self.work / "confidence.pth"
        self.baseline_checkpoint = self.work / "baseline.pth"
        self.output = self.work / "merged.pth"
        self.base_state = {
            "backbone.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)
        }
        self._write_baseline()
        self._write_source("rank")
        self._write_source("confidence")

    def close(self) -> None:
        self.config_context.cleanup()
        self.work_context.cleanup()

    def _write_baseline(self, *, dtype=torch.float32) -> None:
        state = {
            key: value.to(dtype=dtype).clone() for key, value in self.base_state.items()
        }
        torch.save({"model": state}, self.baseline_checkpoint)

    def _criterion(self, role: str):
        if role == "rank":
            criterion = StageBGDINOScoreAdapterCriterion(
                tn_scope="",
                train_mode="rank_only",
                rank_weight=1.0,
                confidence_weight=0.0,
                queue_size=0,
                queue_min_count=0,
            )
        else:
            criterion = StageBGDINOScoreAdapterCriterion(
                tn_scope="image_global_topk_verified",
                train_mode="confidence_only",
                confidence_objective="detached_recent_q05_trust",
                rank_weight=0.0,
                confidence_weight=1.0,
                paired_margin_weight=0.25,
                paired_margin=0.05,
                positive_trust_margin=0.02,
                positive_trust_weight=1.0,
                queue_size=512,
                queue_min_count=256,
            )
            criterion.fpr_positive_queue.copy_(torch.linspace(0.1, 1.0, 512))
            criterion.fpr_negative_queue.copy_(torch.linspace(-0.2, 0.7, 512))
            criterion.fpr_queue_count.fill_(512)
            criterion.fpr_queue_ptr.zero_()
        return criterion.state_dict()

    def _args(self, role: str, config: Path):
        cfg = SLConfig.fromfile(str(config))
        keys = tuple(ARCHITECTURE_FIELDS) + tuple(ROLE_CONFIG[role])
        result = {key: getattr(cfg, key) for key in keys}
        result["config_file"] = str(config.resolve())
        return result

    def _write_source(
        self,
        role: str,
        *,
        base_delta: float = 0.0,
        extra_adapter_key: bool = False,
        adapter_dtype=None,
        recorded_config: Path | None = None,
    ) -> None:
        config = self.rank_config if role == "rank" else self.confidence_config
        checkpoint = (
            self.rank_checkpoint if role == "rank" else self.confidence_checkpoint
        )
        adapter = build_adapter_from_config(config, seed=7 if role == "rank" else 11)
        with torch.no_grad():
            if role == "rank":
                adapter.rank_output.bias.fill_(0.125)
            else:
                adapter.confidence_gate[-1].bias.fill_(-0.25)
        state = {
            key: value.clone() + base_delta for key, value in self.base_state.items()
        }
        for key, value in adapter.state_dict().items():
            tensor = value.detach().clone()
            if adapter_dtype is not None and key == "rank_norm.weight":
                tensor = tensor.to(dtype=adapter_dtype)
            state[ADAPTER_PREFIX + key] = tensor
        if extra_adapter_key:
            state[ADAPTER_PREFIX + "unknown.weight"] = torch.ones(1)
        args = self._args(role, config)
        if recorded_config is not None:
            args["config_file"] = str(recorded_config.resolve())
        torch.save(
            {
                "model": state,
                "criterion": self._criterion(role),
                "optimizer": {
                    "state": {},
                    "param_groups": [{"stage_b_gdino_branch": role, "params": []}],
                },
                "args": args,
                "epoch": 0,
                "iteration": 100 if role == "rank" else 50,
                "epoch_finished": False,
                "checkpoint_reason": "max_train_iters",
            },
            checkpoint,
        )

    def kwargs(self):
        return {
            "output": self.output,
            "rank_checkpoint": self.rank_checkpoint,
            "rank_checkpoint_sha256": sha256_file(self.rank_checkpoint),
            "rank_config": self.rank_config,
            "confidence_checkpoint": self.confidence_checkpoint,
            "confidence_checkpoint_sha256": sha256_file(
                self.confidence_checkpoint
            ),
            "confidence_config": self.confidence_config,
            "baseline_checkpoint": self.baseline_checkpoint,
            "baseline_checkpoint_sha256": sha256_file(self.baseline_checkpoint),
            "eval_config": DEFAULT_EVAL_CONFIG,
        }


class MergeStageBGDINOAdapterEvalTest(unittest.TestCase):
    def setUp(self):
        self.fixture = MergedEvalFixture()

    def tearDown(self):
        self.fixture.close()

    def test_create_is_minimal_and_branch_exact(self):
        receipt = create_merged_eval_checkpoint(**self.fixture.kwargs())
        self.assertEqual(receipt["status"], "verified")
        payload = torch.load(self.fixture.output, map_location="cpu", weights_only=False)
        self.assertEqual(set(payload), {"model", "lineage", "contract"})
        self.assertIs(payload["contract"]["eval_only"], True)
        self.assertIs(payload["contract"]["resumable"], False)
        self.assertEqual(len(payload["contract"]["adapter_key_whitelist"]), 20)
        self.assertEqual(len(payload["contract"]["rank_key_whitelist"]), 8)
        self.assertEqual(len(payload["contract"]["confidence_key_whitelist"]), 12)
        self.assertTrue(all(payload["contract"]["functional_bitwise"].values()))
        for forbidden in (
            "args",
            "criterion",
            "optimizer",
            "lr_scheduler",
            "scaler",
            "rng_state",
            "epoch_rng_state",
        ):
            self.assertNotIn(forbidden, payload)

        rank = torch.load(
            self.fixture.rank_checkpoint, map_location="cpu", weights_only=False
        )["model"]
        confidence = torch.load(
            self.fixture.confidence_checkpoint,
            map_location="cpu",
            weights_only=False,
        )["model"]
        for key in payload["contract"]["rank_key_whitelist"]:
            self.assertTrue(torch.equal(payload["model"][key], rank[key]))
        for key in payload["contract"]["confidence_key_whitelist"]:
            self.assertTrue(torch.equal(payload["model"][key], confidence[key]))

    def test_verify_rejects_serialized_contract_tamper(self):
        create_merged_eval_checkpoint(**self.fixture.kwargs())
        payload = torch.load(self.fixture.output, map_location="cpu", weights_only=False)
        payload["contract"]["rank_tensor_sha256"] = "0" * 64
        torch.save(payload, self.fixture.output)
        with self.assertRaisesRegex(MergedEvalCheckpointError, "contract"):
            verify_merged_eval_checkpoint(self.fixture.output)

    def test_expected_source_hash_and_base_drift_are_rejected(self):
        kwargs = self.fixture.kwargs()
        kwargs["rank_checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(MergedEvalCheckpointError, "hash mismatch"):
            create_merged_eval_checkpoint(**kwargs)

        self.fixture._write_source("rank", base_delta=1.0)
        kwargs = self.fixture.kwargs()
        with self.assertRaisesRegex(MergedEvalCheckpointError, "base"):
            create_merged_eval_checkpoint(**kwargs)

    def test_adapter_whitelist_dtype_and_recorded_config_are_rejected(self):
        self.fixture._write_source("rank", extra_adapter_key=True)
        with self.assertRaisesRegex(MergedEvalCheckpointError, "unexpected|whitelist"):
            create_merged_eval_checkpoint(**self.fixture.kwargs())

        self.fixture._write_source("rank", adapter_dtype=torch.float64)
        with self.assertRaisesRegex(MergedEvalCheckpointError, "shape/dtype"):
            create_merged_eval_checkpoint(**self.fixture.kwargs())

        self.fixture._write_source(
            "rank", recorded_config=self.fixture.confidence_config
        )
        with self.assertRaisesRegex(MergedEvalCheckpointError, "config_file"):
            create_merged_eval_checkpoint(**self.fixture.kwargs())

    def test_strict_tn_loader_verifies_and_binds_compact_provenance(self):
        create_merged_eval_checkpoint(**self.fixture.kwargs())
        cfg = SLConfig.fromfile(str(DEFAULT_EVAL_CONFIG))
        sentinel = object()
        with patch.object(text_eval, "_load_model", return_value=sentinel) as loader:
            model, fields = text_eval._load_model_with_checkpoint_contract(
                cfg, self.fixture.output, torch.device("cpu")
            )
        self.assertIs(model, sentinel)
        loader.assert_called_once_with(
            cfg, str(self.fixture.output), torch.device("cpu")
        )
        self.assertEqual(set(fields), set(text_eval.MERGED_EVAL_SUMMARY_FIELDS))
        self.assertEqual(
            fields["merged_eval_contract_schema"],
            "stageb-gdino-adapter-merged-eval-contract-v1",
        )
        self.assertEqual(
            fields["merged_eval_rank_source_checkpoint_sha256"],
            sha256_file(self.fixture.rank_checkpoint),
        )
        self.assertEqual(
            fields["merged_eval_confidence_source_checkpoint_sha256"],
            sha256_file(self.fixture.confidence_checkpoint),
        )
        row = {"acc50": 0.5}
        text_eval._bind_checkpoint_summary_fields(row, fields)
        self.assertEqual(row["merged_eval_full_model_tensor_sha256"], fields["merged_eval_full_model_tensor_sha256"])

    def test_strict_tn_loader_rejects_tamper_before_model_load(self):
        create_merged_eval_checkpoint(**self.fixture.kwargs())
        payload = torch.load(self.fixture.output, map_location="cpu", weights_only=False)
        payload["contract"]["full_model_tensor_sha256"] = "0" * 64
        torch.save(payload, self.fixture.output)
        cfg = SLConfig.fromfile(str(DEFAULT_EVAL_CONFIG))
        with patch.object(text_eval, "_load_model") as loader:
            with self.assertRaisesRegex(MergedEvalCheckpointError, "contract"):
                text_eval._load_model_with_checkpoint_contract(
                    cfg, self.fixture.output, torch.device("cpu")
                )
        loader.assert_not_called()

    def test_strict_tn_loader_keeps_legacy_checkpoint_path_unchanged(self):
        cfg = SimpleNamespace(
            stage_b_gdino_adapter_merged_eval_only=False,
            stage_b_gdino_score_adapter=True,
        )
        sentinel = object()
        with (
            patch.object(
                text_eval, "verify_merged_eval_checkpoint"
            ) as verifier,
            patch.object(text_eval, "_load_model", return_value=sentinel) as loader,
        ):
            model, fields = text_eval._load_model_with_checkpoint_contract(
                cfg, self.fixture.rank_checkpoint, torch.device("cpu")
            )
        self.assertIs(model, sentinel)
        self.assertEqual(fields, {})
        verifier.assert_not_called()
        loader.assert_called_once_with(
            cfg, str(self.fixture.rank_checkpoint), torch.device("cpu")
        )
        row = {"acc50": 0.5}
        text_eval._bind_checkpoint_summary_fields(row, fields)
        self.assertEqual(row, {"acc50": 0.5})


if __name__ == "__main__":
    unittest.main()
