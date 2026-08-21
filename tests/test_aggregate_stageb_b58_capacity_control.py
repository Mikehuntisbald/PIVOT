import unittest

import numpy as np

from tools.aggregate_stageb_b58_capacity_control import _binary_metrics, _summary


class B58CapacityAggregationTests(unittest.TestCase):
    def test_binary_metrics_are_exact_for_perfect_separation(self):
        metrics = _binary_metrics(
            np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2])
        )
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["positive_aupr"], 1.0)

    def test_binary_metrics_give_half_credit_to_ties(self):
        metrics = _binary_metrics(np.asarray([1.0]), np.asarray([1.0]))
        self.assertEqual(metrics["auroc"], 0.5)

    def test_summary_uses_finite_sample_one_sided_p(self):
        summary = _summary(
            np.asarray([1.0, 1.0, 1.0, 1.0]), 1.0, margin=0.0
        )
        self.assertEqual(summary["one_sided_p_gain_le_margin"], 0.2)
        self.assertGreater(summary["ci95_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
