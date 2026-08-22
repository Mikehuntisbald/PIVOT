import unittest

from tools.aggregate_mmgdino_e6_ownership_2x2 import _holm, _one_sided
from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
)
from tools.mmgdino_e6_ownership_2x2 import (
    FORMAL_SEEDS,
    OWNERS,
    REF_INPUTS,
    TRUNK_SPECS,
    eval_cache_path,
    owner_output_dir,
    training_cache_path,
)
from tools.run_mmgdino_e6_ownership_2x2 import (
    _eval_extract_command,
    _training_extract_command,
)


class MMGDinoE6Ownership2x2Tests(unittest.TestCase):
    def test_matrix_is_exactly_two_by_two_with_three_seeds(self):
        self.assertEqual(tuple(TRUNK_SPECS), ("e6_posctrl", "e6_tn10"))
        self.assertEqual(
            OWNERS, (OWNERSHIP_SHARED_WIDE, OWNERSHIP_ISOLATED_128)
        )
        self.assertNotIn(OWNERSHIP_SHARED_128, OWNERS)
        self.assertEqual(FORMAL_SEEDS, (17, 42, 73))
        targets = {
            owner_output_dir(trunk, owner, seed)
            for trunk in TRUNK_SPECS
            for owner in OWNERS
            for seed in FORMAL_SEEDS
        }
        self.assertEqual(len(targets), 12)

    def test_shared_wide_is_capacity_matched_to_isolated(self):
        shared = MMGDinoE5ResponsibilityOwners(
            ownership=OWNERSHIP_SHARED_WIDE
        ).architecture_report()
        isolated = MMGDinoE5ResponsibilityOwners(
            ownership=OWNERSHIP_ISOLATED_128
        ).architecture_report()
        self.assertEqual(shared.trainable_parameters, 100362)
        self.assertEqual(isolated.trainable_parameters, 100358)
        self.assertEqual(shared.macs_per_query_both_outputs, 99424)
        self.assertEqual(isolated.macs_per_query_both_outputs, 98816)
        self.assertGreater(shared.rank_representation_dim, isolated.rank_representation_dim)

    def test_training_extract_command_binds_selected_trunk_and_old_schedule(self):
        command = _training_extract_command(
            "e6_tn10",
            42,
            output=training_cache_path("e6_tn10", 42),
            receipt=training_cache_path("e6_tn10", 42).with_suffix(".receipt.json"),
        )
        joined = " ".join(command)
        self.assertIn("epoch_6_tn10.pth", joined)
        self.assertIn(TRUNK_SPECS["e6_tn10"].checkpoint_sha256, command)
        self.assertIn("rank_seed42.jsonl", joined)
        self.assertIn("d3_seed42.jsonl", joined)
        self.assertIn("float32", command)

    def test_eval_commands_bind_correct_surface_and_unique_cache(self):
        paths = {
            eval_cache_path(trunk, surface)
            for trunk in TRUNK_SPECS for surface in REF_INPUTS
        }
        self.assertEqual(len(paths), 6)
        command = _eval_extract_command("e6_posctrl", "strict2031")
        joined = " ".join(command)
        self.assertIn("epoch_6_postctrl.pth", joined)
        self.assertIn("strict2031", joined)
        self.assertIn("--mode tn", joined)

    def test_holm_and_one_sided_are_deterministic(self):
        self.assertEqual(_holm({"a": 0.01, "b": 0.04}), {"a": 0.02, "b": 0.04})
        self.assertAlmostEqual(_one_sided([-1.0, 1.0, 2.0], 0.0), 0.5)


if __name__ == "__main__":
    unittest.main()
