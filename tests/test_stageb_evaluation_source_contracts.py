import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_stageb_paper_evaluations as evaluator
from tools import stageb_evaluation_source_contracts as contracts
from tools.stageb_profile_dependency_audit import (
    recursive_local_python_dependencies,
)


class StageBEvaluationSourceContractsTest(unittest.TestCase):
    def test_registry_is_stdlib_only_and_source_families_are_exact(self):
        registry = (
            evaluator.REPO_ROOT / "tools/stageb_evaluation_source_contracts.py"
        ).resolve()
        self.assertEqual(
            recursive_local_python_dependencies(
                [registry], repository_root=evaluator.REPO_ROOT
            ),
            [registry],
        )
        self.assertEqual(
            contracts.SOURCE_FAMILIES,
            ("token", "paper", "historical_baseline"),
        )
        expected = {
            "pivot_token_ablation_training_run": "token",
            "pivot_paper_training_run": "paper",
            "pivot_paper_training_run_rank_diagnostic": "paper",
            "historical_pure_gdino_explicit": "historical_baseline",
        }
        for kind, family in expected.items():
            self.assertEqual(contracts.source_family_for_kind(kind), family)
        with self.assertRaisesRegex(
            contracts.EvaluationSourceContractError, "unknown evaluation source"
        ):
            contracts.source_family_for_kind("historical_pure_gdino")

    def test_leaf_preserves_existing_headline_identity_constants(self):
        from tools import stageb_headline_release_contract as headline

        self.assertEqual(
            dict(contracts.FIXED_BASELINE),
            {
                key: headline.FIXED_BASELINE[key]
                for key in contracts.FIXED_BASELINE
            },
        )
        self.assertEqual(
            dict(contracts.CANONICAL_RUNTIME), dict(headline.CANONICAL_RUNTIME)
        )
        self.assertEqual(
            contracts.canonical_validation_root("baseline", 42),
            headline.canonical_validation_root("baseline", 42),
        )
        m0 = contracts.M0_CONTRACT
        self.assertEqual(
            (
                m0.id,
                m0.architecture_objective,
                m0.seeds,
                m0.batch_size,
                m0.optimizer_updates,
                m0.successful_update_batch_slots,
            ),
            (
                headline.CANDIDATE_ID,
                headline.CANDIDATE_ARCHITECTURE_OBJECTIVE,
                headline.CANDIDATE_SEEDS,
                headline.CANDIDATE_BATCH_SIZE,
                headline.CANDIDATE_UPDATES,
                headline.CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS,
            ),
        )

    def test_m0_and_predeclared_m0n_contracts_are_exact(self):
        m0 = contracts.M0_CONTRACT
        self.assertEqual(m0.id, "M0")
        self.assertEqual(
            m0.config,
            "config/ablations/cfg_stageb_v25_m0_compute_matched.py",
        )
        self.assertEqual(m0.architecture_objective, "S2F")
        self.assertIsNone(m0.token_objective_scope)
        self.assertNotIn("token_objective_scope", m0.expected_row())
        self.assertFalse(m0.matrix_validation_only)
        self.assertTrue(m0.headline)

        m0n = contracts.M0N_CONTRACT
        self.assertEqual(m0n.id, "M0N")
        self.assertEqual(
            m0n.config,
            "config/ablations/cfg_stageb_v25_m0n_compute_matched_allneg_bce.py",
        )
        self.assertEqual(m0n.runner, "tools/run_stageb_headline_m0.py")
        self.assertEqual(
            m0n.dataset,
            "config/datasets_stageb_v21_single_edit_train.json",
        )
        self.assertEqual(m0n.architecture_objective, "S2F")
        self.assertEqual(m0n.phase_ids, ("joint",))
        self.assertEqual(m0n.batch_size, 40)
        self.assertEqual(m0n.optimizer_updates, 23532)
        self.assertEqual(m0n.final_phase_updates, 23532)
        self.assertEqual(m0n.successful_update_batch_slots, 941280)
        self.assertEqual(m0n.iter_checkpoint_interval, 500)
        self.assertEqual(m0n.seeds, (17, 42, 73))
        self.assertEqual(m0n.dedicated_queue_order, (17, 42, 73))
        self.assertEqual(
            m0n.dedicated_queue_run_ids, ("M0N:17", "M0N:42", "M0N:73")
        )
        self.assertEqual(m0n.token_objective, "targetlocal_allneg_bce")
        self.assertEqual(
            m0n.token_objective_scope,
            "target_local_positive_and_all_negative_token_logits",
        )
        self.assertEqual(m0n.predicate_pair_rank_weight, 1.0)
        self.assertTrue(m0n.matrix_validation_only)
        self.assertFalse(m0n.headline)
        self.assertEqual(
            m0n.canonical_training_root(42),
            (
                evaluator.REPO_ROOT
                / "outputs/paper_cvpr_v1/headline_main_compute_matched/M0N/seed42"
            ).resolve(strict=False),
        )
        self.assertEqual(
            m0n.expected_row()["token_objective_scope"],
            "target_local_positive_and_all_negative_token_logits",
        )

    def test_m0n_formal_validation_uses_predeclared_contract(self):
        contract = contracts.M0N_CONTRACT
        observed = evaluator._validate_paper_formal_run_contract(
            sequence={
                "training_seeds_contract": list(contract.seeds),
                "equal_budget_contract": contract.expected_budget(),
            },
            run_root=contract.canonical_training_root(17),
            row=contract.expected_row(),
            row_id=contract.id,
            seed=17,
        )
        self.assertIs(observed, contract)
        self.assertEqual(evaluator._expected_phase_ids("M0N"), ("joint",))

        drifted = contract.expected_row()
        drifted["token_objective"] = "edit_bce"
        with self.assertRaisesRegex(
            evaluator.PaperEvaluationError, "architecture/objective contract"
        ):
            evaluator._validate_paper_formal_run_contract(
                sequence={
                    "training_seeds_contract": list(contract.seeds),
                    "equal_budget_contract": contract.expected_budget(),
                },
                run_root=contract.canonical_training_root(17),
                row=drifted,
                row_id=contract.id,
                seed=17,
            )

    def test_dependency_profiles_split_common_and_source_specific_code(self):
        common = evaluator.evaluation_common_code_paths()
        self.assertEqual(len(common), 72)
        expected_provenance_counts = {
            "token": 3,
            "paper": 4,
            "historical_baseline": 1,
        }
        unions = {}
        for family, expected_count in expected_provenance_counts.items():
            provenance = evaluator.evaluation_source_provenance_paths(family)
            self.assertEqual(len(provenance), expected_count)
            self.assertFalse(set(common).intersection(provenance))
            unions[family] = set(common).union(provenance)
        self.assertEqual(len(unions["token"]), 75)

        forbidden = {
            "tools/stageb_headline_release_contract.py",
            "tools/build_stageb_paper_ablation_completion_receipt.py",
            "tools/build_stageb_b58_exposure_receipt.py",
            "tools/build_stageb_paper_results_manifest.py",
            "tools/aggregate_stageb_table_d_diagnostics.py",
            "tools/aggregate_stageb_matrix_validation.py",
            "tools/aggregate_stageb_paper_results.py",
            "tools/audit_stageb_table_c_dependency_closure.py",
        }
        for family, paths in unions.items():
            labels = {
                path.relative_to(evaluator.REPO_ROOT).as_posix() for path in paths
            }
            self.assertFalse(
                labels.intersection(forbidden),
                f"{family} profile contains downstream consumers",
            )

    def test_plan_tags_provenance_and_does_not_require_headline_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("stage_b = False\n", encoding="utf-8")
            checkpoint.write_bytes(b"baseline")
            source = evaluator._resolve_baseline_source(
                config, checkpoint, "baseline", evaluator.HashCache()
            )
            runtime = evaluator.Runtime(
                python=Path("/usr/bin/python3"),
                data_root=root,
                device="cpu",
                batch_size=1,
                num_workers=0,
                amp=False,
                log_every=1,
            )
            registry = (
                evaluator.REPO_ROOT
                / "tools/stageb_evaluation_source_contracts.py"
            ).resolve()
            strict = {
                label: {
                    **evaluator._file_record(
                        checkpoint, evaluator.HashCache(), roles=(label,)
                    ),
                    "rows": spec["rows"],
                    "source_counts": spec["source_counts"],
                }
                for label, spec in evaluator.STRICT_SPECS.items()
            }
            with (
                patch.object(
                    evaluator,
                    "_strict_manifest_record",
                    side_effect=lambda label, cache: strict[label],
                ),
                patch.object(evaluator, "_config_paths", return_value=[config]),
                patch.object(
                    evaluator,
                    "_evaluation_code_paths",
                    return_value=[evaluator.EVALUATOR],
                ),
                patch.object(
                    evaluator,
                    "_evaluation_source_provenance_paths",
                    return_value=[registry],
                ),
                patch.object(evaluator, "_data_input_paths", return_value=[]),
            ):
                plan = evaluator.build_plan(
                    runtime,
                    source,
                    root / "fresh-output",
                    evaluator.HashCache(),
                )
            by_path = {
                Path(record["path"]): set(record["roles"])
                for record in plan["inputs"]["records"]
            }
            self.assertIn(
                "evaluation_code_dependency", by_path[evaluator.EVALUATOR]
            )
            self.assertIn("source_provenance_dependency", by_path[registry])
            self.assertNotIn("headline_release", plan)

    def test_matrix_only_formal_source_is_rejected_by_screen_and_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.py"
            checkpoint = root / "checkpoint.pth"
            config.write_text("stage_b = True\n", encoding="utf-8")
            checkpoint.write_bytes(b"m0n")
            source = evaluator.EvaluationSource(
                kind="pivot_paper_training_run",
                evaluation_id="M0N_seed17",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=evaluator.HashCache().digest(checkpoint),
                training_run_id="M0N:17",
                training_seed=17,
                formal_contract_id="M0N",
                matrix_validation_only=True,
            )
            runtime = evaluator.Runtime(
                python=Path("/usr/bin/python3"),
                data_root=root,
                device="cpu",
                batch_size=1,
                num_workers=0,
                amp=False,
                log_every=1,
            )
            for profile in (evaluator.SCREEN_PROFILE, evaluator.FINAL_PROFILE):
                with self.assertRaisesRegex(
                    evaluator.PaperEvaluationError,
                    "matrix-validation-only.*matrix_validation",
                ):
                    evaluator.build_plan(
                        runtime,
                        source,
                        root / f"{profile}-output",
                        evaluator.HashCache(),
                        profile=profile,
                    )

            stripped = evaluator.EvaluationSource(
                kind="pivot_paper_training_run",
                evaluation_id="M0N_seed17",
                config=config.resolve(),
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=evaluator.HashCache().digest(checkpoint),
                training_run_id="M0N:17",
                training_seed=17,
            )
            with self.assertRaisesRegex(
                evaluator.PaperEvaluationError,
                "matrix-validation-only.*matrix_validation",
            ):
                evaluator.build_plan(
                    runtime,
                    stripped,
                    root / "stripped-final-output",
                    evaluator.HashCache(),
                    profile=evaluator.FINAL_PROFILE,
                )


if __name__ == "__main__":
    unittest.main()
