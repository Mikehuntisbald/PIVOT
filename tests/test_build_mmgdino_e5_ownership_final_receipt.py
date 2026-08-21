import unittest

from tools.build_mmgdino_e5_ownership_final_receipt import _mean_sd


class MMGDinoE5OwnershipFinalReceiptTests(unittest.TestCase):
    def test_empty_optional_metric_is_explicit(self):
        self.assertEqual(
            _mean_sd([]),
            {"mean": None, "sample_sd": None, "by_seed": None},
        )

    def test_three_seed_metric_binds_seed_order_and_sample_sd(self):
        summary = _mean_sd([1.0, 2.0, 3.0])
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["sample_sd"], 1.0)
        self.assertEqual(summary["by_seed"], {"17": 1.0, "42": 2.0, "73": 3.0})


if __name__ == "__main__":
    unittest.main()
