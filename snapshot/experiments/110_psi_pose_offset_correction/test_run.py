import unittest
import numpy as np
from run import fit_offsets, cues


class OffsetTests(unittest.TestCase):
    def test_known_bias_and_scale(self):
        heights = np.array([100., 200., 300.])
        est = np.ones((3, 17, 2)) * 10
        gt = est + heights[:, None, None] * np.array([.01, -.02])
        offsets, support = fit_offsets(gt, est, heights, np.ones((3, 17), bool), np.array(['a', 'b', 'c']))
        np.testing.assert_allclose(offsets, np.tile([.01, -.02], (17, 1)))
        self.assertEqual(support, [3] * 17)

    def test_non_keyframes_not_used(self):
        est = np.zeros((4, 17, 2))
        gt = np.zeros_like(est)
        gt[-1] = 10000
        flags = np.ones((4, 17), bool)
        flags[-1] = False
        offsets, _ = fit_offsets(gt, est, np.ones(4), flags, np.array(['a', 'b', 'c', 'd']))
        np.testing.assert_array_equal(offsets, 0)

    def test_missing_support_identity(self):
        est = np.ones((2, 17, 2))
        offsets, _ = fit_offsets(est + 1, est, np.ones(2), np.ones((2, 17), bool), np.array(['a', 'b']))
        np.testing.assert_array_equal(offsets, 0)

    def test_no_input_mutation(self):
        points = np.arange(34., dtype=float).reshape(1, 17, 2)
        saved = points.copy()
        cues(points)
        np.testing.assert_array_equal(points, saved)


if __name__ == '__main__':
    unittest.main()
