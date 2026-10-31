"""scripts/render_teaser.py (the paper's Fig. 1): crop window, contact colours, the contact-side tile and the figure grid
(CPU, synthetic)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.render_teaser import contact_colors_linear, contact_tile, crop_window, grid  # noqa: E402


def blob(n=778, seed=0):
    """A closed synthetic surface (n points on an ellipsoid, faces from a convex hull-like fan) standing in for a hand."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0.1, np.pi - 0.1, n)
    x = np.stack([0.04 * np.cos(u) * np.sin(v), 0.08 * np.sin(u) * np.sin(v), 0.02 * np.cos(v) + 0.5], 1)
    order = np.argsort(u)
    faces = np.array([[order[i], order[(i + 1) % n], order[(i + 2) % n]] for i in range(n)])
    return x, faces


class RenderTeaserTests(unittest.TestCase):
    def test_crop_window(self):
        x0, y0, x1, y1 = crop_window([[100, 100, 200, 150], [300, 120, 360, 220]], 1000, 800, 0.04, 4 / 3)
        self.assertLessEqual(x0, 100 - 0.04 * 260 + 1)
        self.assertGreaterEqual(x1, 360 + 0.04 * 260 - 1)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 4 / 3, delta=0.02)
        x0, y0, x1, y1 = crop_window([[0, 0, 990, 790]], 1000, 800, 0.1, 4 / 3)  # larger than the image: clipped inside it
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, 1000)
        self.assertLessEqual(y1, 800)

    def test_contact_colours(self):
        c = contact_colors_linear(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(c[0], [0.72, 0.74, 0.78])
        np.testing.assert_allclose(c[1], np.array([211, 0, 148]) / 255.0)
        np.testing.assert_allclose(c[2], c[1])  # clipped

    def test_contact_tile(self):
        v, f = blob()
        faces = {True: f, False: f}
        hands = [dict(verts3d=v + [0.1, 0, 0], contact=np.linspace(0, 1, len(v)), is_right=False, bbox=[500, 0, 600, 100]),
                 dict(verts3d=v, contact=np.zeros(len(v)), is_right=True, bbox=[0, 0, 100, 100])]
        t = contact_tile(hands, faces, 120, 90, ss=1)
        self.assertEqual(t.shape, (90, 120, 3))
        self.assertLess(t[:, :60].min(), 250)  # a hand drawn in each slot
        self.assertLess(t[:, 60:].min(), 250)
        violet = (t[..., 0].astype(int) - t[..., 1]) > 60  # contact only on the right slot's hand (the left hand by image x)
        self.assertGreater(violet[:, 60:].sum(), violet[:, :60].sum())
        self.assertTrue((contact_tile([], faces, 120, 90) == 255).all())

    def test_grid(self):
        cells = [np.zeros((229, 1520, 3), np.uint8) for _ in range(3)]
        g = grid(cells, 2, 300)
        self.assertEqual(g.shape, (34 + 2 * (229 + 6), 2 * 1520 + 8, 3))
        self.assertTrue((g[34:34 + 229, 1520:1528] == 255).all())  # column gap
        self.assertTrue((g[34 + 229 + 6:, 1528:] == 255).all())  # missing last cell stays white
        self.assertLess(g[:34].min(), 255)  # panel labels in the header


if __name__ == "__main__":
    unittest.main()
