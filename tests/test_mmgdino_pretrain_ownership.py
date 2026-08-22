import unittest

import tools.run_mmgdino_e6_ownership_2x2 as mature
from tools.aggregate_mmgdino_pretrain_ownership import _ci, _one_sided
from tools.mmgdino_e5_ownership import (
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
)
from tools.mmgdino_pretrain_ownership import (
    CHECKPOINT_SHA256,
    FORMAL_SEEDS,
    OWNERS,
    REF_INPUTS,
    TEST5_SURFACES,
    TESTAB_SURFACES,
    TRUNK_ID,
    TRUNK_SPECS,
    eval_cache_path,
    training_cache_path,
)
from tools.run_mmgdino_pretrain_ownership import (
    _eval_extract_command,
    _training_extract_command,
)


class MMGDinoPretrainOwnershipTests(unittest.TestCase):
    def test_matrix_is_exactly_shared_wide_and_isolated(self):
        self.assertEqual(tuple(TRUNK_SPECS), (TRUNK_ID,))
        self.assertEqual(
            OWNERS, (OWNERSHIP_SHARED_WIDE, OWNERSHIP_ISOLATED_128)
        )
        self.assertNotIn(OWNERSHIP_SHARED_128, OWNERS)
        self.assertEqual(FORMAL_SEEDS, (17, 42, 73))
        self.assertEqual(len(OWNERS) * len(FORMAL_SEEDS), 6)

    def test_capacity_and_macs_match_mature_heads(self):
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

    def test_surfaces_include_test5_testab_and_strict(self):
        self.assertEqual(
            TESTAB_SURFACES, ("refcoco_testA", "refcoco_testB")
        )
        self.assertEqual(len(TEST5_SURFACES), 5)
        self.assertEqual(sum(REF_INPUTS[s]["rows"] for s in TEST5_SURFACES), 30969)
        self.assertIn("strict2031", REF_INPUTS)
        self.assertEqual(REF_INPUTS["strict2031"]["rows"], 2031)

    def test_commands_bind_pretrained_checkpoint_and_mature_schedule(self):
        before = tuple(mature.TRUNK_SPECS)
        command = _training_extract_command(
            TRUNK_ID,
            42,
            output=training_cache_path(TRUNK_ID, 42),
            receipt=training_cache_path(TRUNK_ID, 42).with_suffix(".receipt.json"),
        )
        joined = " ".join(command)
        self.assertIn("pretrain_obj365_goldg_grit9m_v3det", joined)
        self.assertIn(CHECKPOINT_SHA256, command)
        self.assertIn("rank_seed42.jsonl", joined)
        self.assertIn("d3_seed42.jsonl", joined)
        self.assertIn("--allow-rank-rows-without-positive", command)
        self.assertEqual(tuple(mature.TRUNK_SPECS), before)

    def test_eval_command_binds_test5_surface_without_mutating_e6_runner(self):
        before = tuple(mature.REF_INPUTS)
        command = _eval_extract_command(TRUNK_ID, "refcocog_test")
        joined = " ".join(command)
        self.assertIn("refcocog_umd_test.jsonl", joined)
        self.assertIn(str(eval_cache_path(TRUNK_ID, "refcocog_test")), joined)
        self.assertEqual(tuple(mature.REF_INPUTS), before)

    def test_bootstrap_helpers_are_deterministic(self):
        self.assertEqual(_ci([0.0, 1.0, 2.0]), [0.05, 1.95])
        self.assertAlmostEqual(_one_sided([-1.0, 1.0, 2.0], 0.0), 0.5)


if __name__ == "__main__":
    unittest.main()
