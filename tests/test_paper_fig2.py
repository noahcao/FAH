"""scripts/paper_fig2.py (the paper's Fig. 2 tiles): crop windows, the part map placement, contact colours and the mesh
rasteriser (CPU, synthetic)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.paper_fig2 import contact_colors_linear, crop_box, full_label_map, hand_crop, raster  # noqa: E402


class PaperFig2Tests(unittest.TestCase):
    def test_crop_box_aspect_inside_and_covering(self):
        boxes = [(300, 700, 600, 1000), (700, 800, 1000, 1200)]
        x0, y0, x1, y1 = crop_box(boxes, 1408, 1408, 1.3, 0.10)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 1.3, delta=0.01)
        self.assertTrue(0 <= x0 <= 300 and 0 <= y0 <= 700 and 1000 <= x1 <= 1408 and 1200 <= y1 <= 1408)
        x0, y0, x1, y1 = crop_box([(0, 0, 1400, 100)], 1408, 1408, 1.6, 0.45)  # wider than the frame: shrunk to fit
        self.assertTrue(0 <= x0 and x1 <= 1408 and 0 <= y0 and y1 <= 1408)

    def test_hand_crop_square_with_context(self):
        (x0, y0, x1, y1), s = hand_crop(dict(bbox=(100, 200, 180, 300)))
        self.assertEqual(s, 125)
        self.assertAlmostEqual(x1 - x0, y1 - y0, delta=1)
        self.assertAlmostEqual((x0 + x1) / 2, 140, delta=1)

    def test_part_map_is_placed_and_clipped(self):
        roi = np.arange(1, 17, dtype=np.uint8).reshape(4, 4)
        lab = full_label_map(dict(parts_roi=(8, 6, roi)), 10, 10)
        self.assertEqual(lab.shape, (10, 10))
        np.testing.assert_array_equal(lab[6:10, 8:10], roi[:, :2])
        self.assertEqual(int(lab[:6].sum() + lab[:, :8].sum()), 0)

    def test_contact_colours_linear_grey_to_violet(self):
        c = contact_colors_linear([-1.0, 0.0, 0.5, 1.0, 2.0])
        np.testing.assert_allclose(c[0], c[1])
        np.testing.assert_allclose(c[1], [0.72, 0.74, 0.78])
        np.testing.assert_allclose(c[3], np.array([211, 0, 148]) / 255.0)
        np.testing.assert_allclose(c[4], c[3])
        np.testing.assert_allclose(c[2], (c[1] + c[3]) / 2)

    def test_raster_covers_the_projected_triangle(self):
        v = np.array([[-0.1, -0.1, 1.0], [0.1, -0.1, 1.0], [0.0, 0.1, 1.0]])
        K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])
        img, mask = raster(v, np.array([[0, 1, 2]]), K, (64, 64), (0.5, 0.5, 0.5))
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertGreater(mask[32, 32], 0)
        self.assertEqual(mask[2, 2], 0)
        np.testing.assert_array_equal(img[2, 2], [255, 255, 255])
        self.assertTrue(0.5 * 20 * 20 <= (mask > 0).sum() <= 0.5 * 20 * 20 + 66)  # the triangle's area + its boundary pixels
        self.assertTrue((img[32, 32] < 255).all())


if __name__ == "__main__":
    unittest.main()
