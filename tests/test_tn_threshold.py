import unittest

import numpy as np

from tools.eval_stageb_tn_val import _threshold_for_tpr as stageb_threshold
from tools.eval_text_groundingdino_refcoco_tn import (
    _threshold_for_tpr as gdino_threshold,
)


class TnThresholdTest(unittest.TestCase):
    def test_exact_order_statistic_never_undershoots_requested_tpr(self):
        scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        for threshold_fn in (stageb_threshold, gdino_threshold):
            threshold = threshold_fn(scores, 0.75)
            self.assertAlmostEqual(threshold, 0.2, places=6)
            self.assertGreaterEqual(float(np.mean(scores >= threshold)), 0.75)

            threshold = threshold_fn(scores, 0.95)
            self.assertAlmostEqual(threshold, 0.1, places=6)
            self.assertGreaterEqual(float(np.mean(scores >= threshold)), 0.95)

    def test_zero_tpr_rejects_every_finite_positive(self):
        scores = np.asarray([0.1, 0.2], dtype=np.float32)
        for threshold_fn in (stageb_threshold, gdino_threshold):
            threshold = threshold_fn(scores, 0.0)
            self.assertEqual(int(np.sum(scores >= threshold)), 0)


if __name__ == "__main__":
    unittest.main()
