import copy
import json
import types
import unittest
from pathlib import Path

import main
from util.slconfig import SLConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_data_driven_dd1_role_routed_patch_residual_clean_20260727.py"
)
CENTERED_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_data_driven_dd1_role_routed_patch_residual_"
    "raw_centered_clean_20260727.py"
)
TOPK_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_data_driven_dd1_role_routed_patch_topk_semantic_"
    "clean_20260727.py"
)


class RoleRoutedMainContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = SLConfig.fromfile(str(CONFIG))
        cls.values = dict(config._cfg_dict)
        cls.values.update(
            config_file=str(CONFIG),
            datasets=cls.values[
                "stage_b_data_driven_assignment_dataset_config_path"
            ],
            amp=True,
            eval=False,
            gradient_accumulation_steps=1,
            iter_checkpoint_interval=1000,
            max_train_iters=1000,
            num_workers=8,
            prefetch_factor=1,
            pretrain_model_path=cls.values[
                "stage_b_data_driven_base_initializer_path"
            ],
            resume="",
            seed=42,
        )
        centered = SLConfig.fromfile(str(CENTERED_CONFIG))
        cls.centered_values = dict(centered._cfg_dict)
        cls.centered_values.update(
            config_file=str(CENTERED_CONFIG),
            datasets=cls.centered_values[
                "stage_b_data_driven_assignment_dataset_config_path"
            ],
            amp=True,
            eval=False,
            gradient_accumulation_steps=1,
            iter_checkpoint_interval=1000,
            max_train_iters=1000,
            num_workers=8,
            prefetch_factor=1,
            pretrain_model_path=cls.centered_values[
                "stage_b_data_driven_base_initializer_path"
            ],
            resume="",
            seed=42,
        )
        topk = SLConfig.fromfile(str(TOPK_CONFIG))
        cls.topk_values = dict(topk._cfg_dict)
        cls.topk_values.update(
            config_file=str(TOPK_CONFIG),
            datasets=cls.topk_values[
                "stage_b_data_driven_assignment_dataset_config_path"
            ],
            amp=True,
            eval=False,
            gradient_accumulation_steps=1,
            iter_checkpoint_interval=1000,
            max_train_iters=1000,
            num_workers=cls.topk_values[
                "stage_b_data_driven_role_expected_num_workers"
            ],
            pin_memory=cls.topk_values[
                "stage_b_data_driven_role_expected_pin_memory"
            ],
            prefetch_factor=1,
            pretrain_model_path=cls.topk_values[
                "stage_b_data_driven_base_initializer_path"
            ],
            resume="",
            seed=42,
        )

    def _args(self, **updates):
        values = copy.deepcopy(self.values)
        values.update(updates)
        return types.SimpleNamespace(**values)

    @staticmethod
    def _validate(args):
        main._validate_stage_b_data_driven_role_routed_training_contract(
            args,
            base_path=Path(args.stage_b_data_driven_base_initializer_path).resolve(
                strict=True
            ),
            dataset_path=Path(args.datasets).resolve(strict=True),
        )

    def test_exact_clean_role_routed_contract_passes(self):
        self._validate(self._args())
        self._validate(types.SimpleNamespace(**copy.deepcopy(self.centered_values)))
        self._validate(types.SimpleNamespace(**copy.deepcopy(self.topk_values)))

    def test_topk_context_shape_is_bound(self):
        for key, value in (
            ("stage_b_data_driven_patch_residual_context_dim", 8),
            ("stage_b_data_driven_patch_residual_context_topk", 5),
        ):
            mutated = copy.deepcopy(self.topk_values)
            mutated[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                RuntimeError, "architecture drifted"
            ):
                self._validate(types.SimpleNamespace(**mutated))

    def test_topk_worker_count_is_resource_bound(self):
        self.assertEqual(
            self.topk_values["stage_b_data_driven_role_expected_num_workers"], 0
        )
        mutated = copy.deepcopy(self.topk_values)
        mutated["num_workers"] = 2
        with self.assertRaisesRegex(RuntimeError, "CLI runtime drifted"):
            self._validate(types.SimpleNamespace(**mutated))

        mutated = copy.deepcopy(self.topk_values)
        mutated["pin_memory"] = True
        with self.assertRaisesRegex(RuntimeError, "pin-memory runtime drifted"):
            self._validate(types.SimpleNamespace(**mutated))

    def test_formal_dataset_rows_require_lazy_jsonl(self):
        dataset_path = Path(
            self.topk_values[
                "stage_b_data_driven_assignment_dataset_config_path"
            ]
        )
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["train"]), 3)
        self.assertTrue(
            all(row.get("lazy_jsonl") is True for row in payload["train"])
        )

    def test_centering_flag_and_architecture_are_bound(self):
        centered = copy.deepcopy(self.centered_values)
        centered["stage_b_data_driven_patch_residual_center_raw"] = False
        with self.assertRaisesRegex(RuntimeError, "architecture drifted"):
            self._validate(types.SimpleNamespace(**centered))

    def test_config_does_not_define_argparse_owned_runtime_keys(self):
        parser_keys = {action.dest for action in main.get_args_parser()._actions}
        config_keys = set(SLConfig.fromfile(str(CONFIG))._cfg_dict)
        allow_cfg_override = {"fix_size", "persistent_workers"}
        forbidden_collisions = (config_keys & parser_keys) - allow_cfg_override
        self.assertEqual(forbidden_collisions, set())

    def test_old_supervision_architecture_and_teacher_route_fail_closed(self):
        mutations = {
            "old supervision": {
                "stage_b_data_driven_rank_supervision": (
                    "official_same_image_same_category_assignment_v1"
                )
            },
            "relational architecture": {
                "stage_b_data_driven_rank_architecture": "relational_v1"
            },
            "unbalanced patch rows": {
                "stage_b_data_driven_patch_row_balance_contract": "legacy"
            },
            "mismatched patch gradient": {
                "stage_b_data_driven_patch_drop_positive_anchor_gradient_policy": (
                    "global_max_positive_v1"
                )
            },
            "residual width": {
                "stage_b_data_driven_patch_residual_hidden_dim": 64
            },
            "teacher adapter": {"stage_b_gdino_score_adapter": True},
        }
        for label, updates in mutations.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                RuntimeError,
                "contract drifted|architecture drifted|routes are not disabled",
            ):
                self._validate(self._args(**updates))

    def test_resume_and_initializer_lineage_drift_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "fresh optimizer"):
            self._validate(self._args(resume="checkpoint.pth"))
        with self.assertRaisesRegex(RuntimeError, "initializer receipt contract"):
            self._validate(
                self._args(
                    stage_b_data_driven_role_initializer_source_checkpoint_sha256=(
                        "0" * 64
                    )
                )
            )

    def test_cli_runtime_and_expected_runtime_drift_fail_closed(self):
        mutations = {
            "max train iters": {"max_train_iters": 999},
            "checkpoint interval": {"iter_checkpoint_interval": 999},
            "expected max train iters": {
                "stage_b_data_driven_role_expected_max_train_iters": 999
            },
            "expected checkpoint interval": {
                "stage_b_data_driven_role_expected_iter_checkpoint_interval": 999
            },
            "amp disabled": {"amp": False},
            "seed": {"seed": 43},
            "num workers": {"num_workers": 4},
            "prefetch factor": {"prefetch_factor": 2},
            "gradient accumulation": {"gradient_accumulation_steps": 2},
        }
        for label, updates in mutations.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                RuntimeError, "contract drifted|CLI runtime drifted"
            ):
                self._validate(self._args(**updates))


if __name__ == "__main__":
    unittest.main()
