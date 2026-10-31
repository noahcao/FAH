"""MANO mesh fit (fah/fast_mano.py MeshFitter.fit) with centre (residuals relative to the target wrist) + a larger
finite-difference step: the same fit as in camera coordinates, and stable under 1e-6 relative input changes for a far
hand (CPU, jac fdb, eager; needs MANO_DIR with MANO_RIGHT.pkl)."""
import os
import unittest

import numpy as np
import torch

MANO_DIR = os.environ.get("MANO_DIR", "data/mano")


@unittest.skipUnless(os.path.exists(os.path.join(MANO_DIR, "MANO_RIGHT.pkl")), "MANO_DIR (MANO_RIGHT.pkl) not found")
class MeshFitCentreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fah.fast_mano import FastMano, rodrigues

        torch.manual_seed(0)
        m = FastMano(MANO_DIR, "cpu")
        pose, betas = 0.3 * torch.randn(2, 45), 0.5 * torch.randn(2, 10)
        R = rodrigues(torch.tensor([[0.3, -1.2, 0.4], [2.0, 0.5, -0.3]]))
        t = torch.tensor([[0.4, -0.3, 2.5], [-0.6, 0.2, 3.0]])  # far hands: metres from the camera
        with torch.no_grad():
            v, j = m(pose, betas)
        cls.tV = (v @ R.mT + t[:, None]).numpy().astype(np.float64)
        cls.tJ = (j @ R.mT + t[:, None]).numpy().astype(np.float64)
        cls.R0 = R.numpy()
        cls.pose0 = (pose + 0.05 * torch.randn(2, 45)).numpy()

    def fit(self, tV, tJ, **kw):
        from fah.fast_mano import MeshFitter

        f = MeshFitter(MANO_DIR, "cpu", iters=3, solver="gn", wv=0.3, jac="fdb", compile=False, **kw)
        return f.fit(tV, tJ, self.R0, self.pose0, np.zeros((2, 10)))

    def test_centre_is_the_same_fit(self):
        a, b = self.fit(self.tV, self.tJ), self.fit(self.tV, self.tJ, centre=True, fd_eps=1e-2)
        self.assertLess(np.abs(a - b).max(), 5e-4)
        self.assertLess(np.abs(b - self.tV).max(), 2e-3)

    def test_centre_is_stable_under_tiny_input_changes(self):
        rng = np.random.RandomState(1)
        tV, tJ = self.tV * (1 + 1e-6 * rng.randn(*self.tV.shape)), self.tJ * (1 + 1e-6 * rng.randn(*self.tJ.shape))
        a, b = self.fit(self.tV, self.tJ, centre=True, fd_eps=1e-2), self.fit(tV, tJ, centre=True, fd_eps=1e-2)
        self.assertLess(np.abs(a - b).max(), 5e-5)  # 0.05 mm


if __name__ == "__main__":
    unittest.main()
