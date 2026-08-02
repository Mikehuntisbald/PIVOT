import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from models.GroundingDINO.stage_b_u0_patch_rank import (
    U0_PATCH_SOURCE_KEYS,
    U0_SEALED_TEACHER_ARCHITECTURE_FIELDS,
    U0_TEACHER_FUNCTIONAL_FIELDS,
    stage_b_u0_tensor_state_sha256,
)
from tools import build_stageb_u0_training_receipt as receipt
from tools.audit_stageb_u0_transition import (
    FROZEN_PATCH_KEYS,
    audit_u0_transition,
)
from tools.build_stageb_u0_initializer import (
    SCHEMA as INITIALIZER_SCHEMA,
    compose_u0_model_state,
)
from tools.stageb_gdino_adapter_probe_audit import file_record


class _UnsafeCheckpointValue:
    pass


class U0TrainingReceiptFixture:
    def __init__(self):
        self.context = tempfile.TemporaryDirectory()
        self.temp = Path(self.context.name)
        self.root = self.temp / "formal_u0"
        self.initializer_dir = self.root / "initializer"
        self.milestones = self.root / "milestones"
        self.audits = self.root / "audits"
        self.initializer_dir.mkdir(parents=True)
        self.milestones.mkdir()
        self.audits.mkdir()
        self.initializer = self.initializer_dir / "checkpoint_u0_init.pth"
        self.u50 = self.milestones / "checkpoint_iter_000050.pth"
        self.u100 = self.milestones / "checkpoint_iter_000100.pth"
        self.output = self.root / "training_receipt.json"
        self.config = (
            receipt.REPO_ROOT
            / "config/ablations/cfg_stageb_u0_r100p50_patch_rank.py"
        )
        self.data_root = self.temp / "data"
        self.data_root.mkdir()
        self.datasets = self.temp / "datasets.json"
        self._write_data()
        self.initializer_payload = self._make_initializer()
        torch.save(self.initializer_payload, self.initializer)
        self._write_milestone(self.u50, 50)
        self._write_milestone(self.u100, 100)
        self._write_transition(self.u50, 50)
        self._write_transition(self.u100, 100)

    def close(self):
        self.context.cleanup()

    def _write_data(self):
        canonical = self.data_root / "canonical.json"
        canonical.write_text(
            json.dumps([{"id": 0, "raw_name": "person"}]), encoding="utf-8"
        )
        tsv = self.data_root / "patches.tsv"
        tsv.write_text(
            "path\tclass\tbucket\n/a.jpg\tperson\tclean\n", encoding="utf-8"
        )
        train = []
        for index, name in enumerate(("refcoco", "refcocoplus", "refcocog")):
            annotation = self.data_root / f"{name}.jsonl"
            annotation.write_text(
                json.dumps(
                    {
                        "filename": f"/{index}.jpg",
                        "instances": [{"bbox": [0, 0, 1, 1], "class_id": 0}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train.append(
                {
                    "dataset_mode": "patch_episode",
                    "root": "/",
                    "anno": str(annotation),
                    "canonical_classes_json": str(canonical),
                    "support_patch_tsv": str(tsv),
                    "support_patch_bucket": "clean",
                    "keep_only_support_gt": True,
                    "neg_episode_prob": 0.0,
                    "lvis_neg_category_only": False,
                    "mix_weight": 2.0,
                }
            )
        self.datasets.write_text(
            json.dumps({"train": train, "val": []}), encoding="utf-8"
        )

    def _make_initializer(self):
        merged = OrderedDict(
            {
                "backbone.block.weight": torch.tensor([1.0, 2.0]),
                "transformer.weight": torch.tensor([3.0]),
                "stage_b_gdino_score_adapter.rank_output.weight": torch.tensor(
                    [[4.0]]
                ),
            }
        )
        stagea = {
            key: torch.full((1,), float(index + 10))
            for index, key in enumerate(sorted(U0_PATCH_SOURCE_KEYS))
        }
        template = OrderedDict(
            (key, torch.zeros_like(value)) for key, value in merged.items()
        )
        template["patch_encoder.backbone.block.weight"] = torch.zeros(2)
        for key in sorted(U0_PATCH_SOURCE_KEYS):
            template[key] = torch.zeros_like(stagea[key])
        template["stage_b_u0_patch_rank_adapter.output.weight"] = torch.zeros(1)
        template["stage_b_u0_patch_rank_adapter.output.bias"] = torch.zeros(1)
        state, roles = compose_u0_model_state(template, merged, stagea)
        teacher_architecture = {
            key: False if key == "enable_patch_branch" else 1
            for key in U0_SEALED_TEACHER_ARCHITECTURE_FIELDS
        }
        u0_architecture = dict(teacher_architecture)
        u0_architecture["enable_patch_branch"] = True
        contract = {
            "schema": INITIALIZER_SCHEMA,
            "model_state_keys": len(state),
            "role_key_counts": {key: len(value) for key, value in roles.items()},
            "sealed_teacher_architecture": teacher_architecture,
            "u0_architecture": u0_architecture,
            "teacher_functional_bitwise": {
                key: True for key in U0_TEACHER_FUNCTIONAL_FIELDS
            },
            "role_keys": roles,
            "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, state.keys()
            ),
            "merged_teacher_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, roles["merged"]
            ),
            "stagea_patch_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, roles["stagea_patch"]
            ),
            "shared_backbone_alias_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, roles["shared_backbone_alias"]
            ),
            "u0_zero_tensor_sha256": stage_b_u0_tensor_state_sha256(
                state, roles["u0_zero"]
            ),
            "invariants": {
                "merged_teacher_copied_bitwise": True,
                "stagea_patch_specific_keys_only": True,
                "stagea_patch_backbone_imported": False,
                "patch_backbone_aliases_source_b58": True,
                "u0_output_exactly_zero": True,
                "u0_rank_equals_r100_at_initialization": True,
                "p50_confidence_unchanged": True,
            },
        }
        return {"model": state, "u0_initializer": contract}

    def _args(self, target):
        return {
            "config_file": str(self.config),
            "datasets": str(self.datasets),
            "output_dir": str(self.root),
            "batch_size": 56,
            "seed": 42,
            "amp": True,
            "max_train_iters": target,
            "gradient_accumulation_steps": 1,
            "world_size": 1,
            "distributed": False,
            "stage_b_u0_patch_rank": True,
            "enable_patch_branch": True,
            "stage_b_gdino_score_adapter": True,
            "stage_b_gdino_adapter_train_mode": "rank_only",
            "resume": "" if target == 50 else str(self.u50),
            "pretrain_model_path": str(self.initializer) if target == 50 else None,
        }

    def _trained_model(self, target):
        state = OrderedDict(
            (key, value.clone())
            for key, value in self.initializer_payload["model"].items()
        )
        delta = float(target) / 100.0
        for key in sorted(set(U0_PATCH_SOURCE_KEYS).difference(FROZEN_PATCH_KEYS)):
            state[key].add_(delta)
        state["stage_b_u0_patch_rank_adapter.output.weight"].fill_(delta)
        state["stage_b_u0_patch_rank_adapter.output.bias"].fill_(-delta)
        return state

    def _write_milestone(self, path, target):
        torch.save(
            {
                "model": self._trained_model(target),
                "epoch": 0,
                "iteration": target,
                "optimizer_updates": target,
                "epoch_finished": False,
                "checkpoint_reason": "max_train_iters",
                "optimizer": {"state": {}, "param_groups": []},
                "lr_scheduler": {},
                "scaler": {"scale": 65536.0},
                "criterion": {},
                "rng_state": np.random.RandomState(target).get_state(),
                "args": self._args(target),
            },
            path,
        )

    def _write_transition(self, checkpoint, target):
        with torch.serialization.safe_globals(list(receipt._NUMPY_SAFE_GLOBALS)):
            trained = torch.load(
                checkpoint,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        result = audit_u0_transition(self.initializer_payload, trained)
        result["initializer"] = file_record(self.initializer.resolve())
        result["checkpoint"] = file_record(checkpoint.resolve())
        path = self.audits / f"{checkpoint.stem}.transition.json"
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )

    def build(self):
        with mock.patch.object(
            receipt,
            "CORE_SOURCE_PATHS",
            ("tools/build_stageb_u0_training_receipt.py",),
        ):
            return receipt.build_receipt(
                output=self.output,
                initializer=self.initializer,
                u50=self.u50,
                u100=self.u100,
                config=self.config,
                datasets=self.datasets,
                data_root=self.data_root,
            )


class BuildStageBU0TrainingReceiptTest(unittest.TestCase):
    def setUp(self):
        self.fixture = U0TrainingReceiptFixture()

    def tearDown(self):
        self.fixture.close()

    def test_build_binds_lineage_data_sources_and_frozen_hashes(self):
        value = self.fixture.build()

        self.assertEqual(value["schema"], receipt.SCHEMA)
        self.assertEqual(value["invariants"]["batch_size"], 56)
        self.assertEqual(value["invariants"]["optimizer_updates"], [50, 100])
        self.assertTrue(value["invariants"]["u100_resumes_u50"])
        self.assertEqual(
            value["initializer"]["frozen_tensor_sha256"],
            value["checkpoints"]["u50"]["frozen_tensor_sha256"],
        )
        self.assertEqual(
            value["checkpoints"]["u50"]["frozen_tensor_sha256"],
            value["checkpoints"]["u100"]["frozen_tensor_sha256"],
        )
        self.assertEqual(value["datasets"]["train_entry_count"], 3)
        self.assertEqual(value["datasets"]["support_patch_tsv"]["parsed_rows"], 1)
        self.assertEqual(value["datasets"]["canonical_classes_json"]["parsed_entries"], 1)
        self.assertTrue(
            value["checkpoints"]["u100"]["transition_audit"]["recomputed_equal"]
        )
        unsealed = {
            key: item for key, item in value.items() if key != "receipt_sha256"
        }
        self.assertEqual(
            value["receipt_sha256"], receipt.canonical_json_sha256(unsealed)
        )
        self.assertEqual(
            json.loads(self.fixture.output.read_text(encoding="ascii")), value
        )

    def test_refuses_overwrite_before_reauditing_inputs(self):
        self.fixture.build()
        with mock.patch.object(receipt, "build_receipt_payload") as payload:
            with self.assertRaisesRegex(
                receipt.U0TrainingReceiptError, "refusing to overwrite"
            ):
                self.fixture.build()
        payload.assert_not_called()
        self.assertFalse(list(self.fixture.root.glob(".training_receipt.json.tmp.*")))

    def test_rejects_u100_resume_lineage_drift(self):
        with torch.serialization.safe_globals(list(receipt._NUMPY_SAFE_GLOBALS)):
            value = torch.load(
                self.fixture.u100,
                map_location="cpu",
                weights_only=True,
            )
        value["args"]["resume"] = str(self.fixture.initializer)
        torch.save(value, self.fixture.u100)

        with self.assertRaisesRegex(receipt.U0TrainingReceiptError, "U100 resume"):
            self.fixture.build()
        self.assertFalse(self.fixture.output.exists())

    def test_rejects_frozen_tensor_drift_even_with_stale_audit(self):
        with torch.serialization.safe_globals(list(receipt._NUMPY_SAFE_GLOBALS)):
            value = torch.load(
                self.fixture.u100,
                map_location="cpu",
                weights_only=True,
            )
        value["model"]["backbone.block.weight"].add_(1.0)
        torch.save(value, self.fixture.u100)

        with self.assertRaisesRegex(
            receipt.U0TrainingReceiptError, "changed frozen tensors"
        ):
            self.fixture.build()
        self.assertFalse(self.fixture.output.exists())

    def test_safe_loader_rejects_unapproved_pickle_global(self):
        unsafe = self.fixture.temp / "unsafe.pth"
        torch.save({"value": _UnsafeCheckpointValue()}, unsafe)
        with self.assertRaisesRegex(
            receipt.U0TrainingReceiptError, "unsafe or unknown pickle globals"
        ):
            receipt._safe_load_checkpoint(unsafe, label="unsafe fixture")

    def test_rejects_unparseable_annotation_jsonl(self):
        dataset = json.loads(self.fixture.datasets.read_text(encoding="utf-8"))
        annotation = Path(dataset["train"][1]["anno"])
        annotation.write_text("{not-json}\n", encoding="utf-8")

        with self.assertRaisesRegex(receipt.U0TrainingReceiptError, "invalid JSON"):
            self.fixture.build()
        self.assertFalse(self.fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
