import unittest

import torch

from tools import build_stageb_u2_training_receipt as receipt


def _optimizer_payload():
    state = {
        parameter_id: {
            "step": torch.tensor(100.0),
            "exp_avg": torch.zeros(1),
            "exp_avg_sq": torch.zeros(1),
        }
        for parameter_id in range(16)
    }

    def group(branch, parameters):
        return {
            "params": list(parameters),
            "stage_b_u0_branch": branch,
            "lr": 3e-4,
            "initial_lr": 3e-4,
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }

    return {
        "optimizer": {
            "state": state,
            "param_groups": [
                group("patch_rank_residual", range(8)),
                group("patch_projection", range(8, 16)),
            ],
        }
    }


def _amp_payload():
    return {
        "iteration": 100,
        "optimizer_updates": 100,
        "scaler": {
            "scale": 8192.0,
            "_growth_tracker": 100,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
    }


class BuildStageBU2TrainingReceiptTest(unittest.TestCase):
    def test_optimizer_binding_requires_two_disjoint_complete_u100_groups(self):
        value = receipt._optimizer_binding(_optimizer_payload())

        self.assertEqual(value["group_count"], 2)
        self.assertEqual(
            [group["branch"] for group in value["groups"]],
            ["patch_rank_residual", "patch_projection"],
        )
        self.assertEqual(value["state_count"], 16)
        self.assertEqual(value["state_step_values"], [100])
        self.assertTrue(value["groups_disjoint"])

    def test_optimizer_binding_rejects_overlap(self):
        payload = _optimizer_payload()
        payload["optimizer"]["param_groups"][1]["params"][0] = 0

        with self.assertRaisesRegex(receipt.U2TrainingReceiptError, "overlap"):
            receipt._optimizer_binding(payload)

    def test_amp_binding_seals_zero_skip_derivation(self):
        value = receipt._amp_binding(_amp_payload())

        self.assertEqual(value["amp_skipped_steps"], 0)
        self.assertEqual(value["initial_scale"], 8192.0)
        self.assertEqual(value["final_scale"], 8192.0)
        self.assertEqual(value["growth_tracker"], 100)
        self.assertTrue(
            value["zero_skip_derivation"][
                "iteration_equals_successful_optimizer_updates"
            ]
        )

    def test_amp_binding_rejects_backoff(self):
        payload = _amp_payload()
        payload["scaler"]["scale"] = 4096.0

        with self.assertRaisesRegex(receipt.U2TrainingReceiptError, "scaler.scale"):
            receipt._amp_binding(payload)


if __name__ == "__main__":
    unittest.main()
