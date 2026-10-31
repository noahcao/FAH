"""MOW contact builder: the camera conventions and the label encoding (CPU only, no data)."""
import unittest

import numpy as np

from train.build_mow_contact import fah_camera, fit_camera, project


class MOWConventionTests(unittest.TestCase):
    def test_cameras(self):
        K = fit_camera(720, 1280)  # MOW's fits: f = L = max(H, W), principal point (L / 2, L / 2) of the padded square
        np.testing.assert_allclose(K, [[1280, 0, 640], [0, 1280, 640], [0, 0, 1]])
        np.testing.assert_allclose(fah_camera(720, 1280), [[1280, 0, 640], [0, 1280, 360], [0, 0, 1]])  # FAH: image centre

    def test_hand_scale_is_projection_invariant(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-0.05, 0.05, (778, 3)) + [0, 0, 0.6]
        K = fit_camera(480, 640)
        np.testing.assert_allclose(project(X, K), project(1.85 * X, K), atol=1e-9)  # hand_s only moves the hand along the rays


if __name__ == "__main__":
    unittest.main()
