"""Supplementary samples figure (scripts/supp_samples.py): crop window, keypoint tile, seeded low-light order and the
recorded-check selection, on synthetic data (CPU)."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.supp_samples import PICKS, T, dark_order, dark_picks, square, tile  # noqa: E402


def hand(x0, y0, side, vis=21):
    kp = np.stack([np.linspace(x0, x0 + side, 21), np.linspace(y0, y0 + side, 21)], 1)
    return dict(bbox=[x0, y0, x0 + side, y0 + side], kp2d=kp.tolist(), vis=[1] * vis + [0] * (21 - vis))


class SuppSamplesTests(unittest.TestCase):
    def test_square_crop_is_clipped_and_scaled(self):
        self.assertEqual(square([10, 10, 60, 40], 100, 200, 1.8), (0, 0, 90))  # 50 * 1.8 = 90, clipped at the corner
        self.assertEqual(square([0, 0, 400, 50], 100, 200, 1.8), (100, 0, 100))  # side capped by the image, centred
        x0, y0, side = square([100, 40, 120, 60], 100, 200, 2.0)
        self.assertEqual((x0, y0, side), (90, 30, 40))

    def test_tile_draws_keypoints_inside_the_crop_only(self):
        img = np.zeros((100, 200, 3), np.uint8)
        t = tile(img, [hand(20, 20, 40)], (0, 0, 100))
        self.assertEqual(t.shape, (T, T, 3))
        self.assertGreater(int(t.sum()), 0)
        far = tile(img, [hand(150, 20, 40)], (0, 0, 100))  # mostly right of the crop: not drawn
        self.assertEqual(int(far.sum()), 0)

    def test_dark_order_filters_and_is_seeded(self):
        ims = [dict(id=f"coco/{i}", subset="coco", hands=[hand(0, 0, 100 if i % 3 else 50)]) for i in range(10)]
        ims.append(dict(id="hot3d/x", subset="hot3d", hands=[hand(0, 0, 100)]))
        lic = {i: (4 if i % 2 else 1) for i in range(10)}
        o1 = dark_order(ims, lic, [4, 5, 7, 8], 80, 15, 0)
        o2 = dark_order(ims, lic, [4, 5, 7, 8], 80, 15, 0)
        self.assertEqual(o1, o2)
        ids = sorted(int(k.split("/")[1]) for k, _ in o1)
        self.assertEqual(ids, [1, 5, 7])  # odd (licence 4), box side >= 80 (i % 3 != 0)

    def test_picks_follow_the_recorded_checks(self):
        order = [("a", [0]), ("b", [0]), ("c", [0]), ("d", [0]), ("e", [0]), ("f", [0])]
        dec = [{"rank": i, "id": k, "pass": ok} for i, ((k, _), ok) in enumerate(zip(order, [False, True, False, True, True, True]))]
        self.assertEqual(dark_picks(order, dec), ["b", "d", "e", "f"])
        with self.assertRaises(AssertionError):
            dark_picks(order[::-1], dec)

    def test_shipped_selection_file(self):
        cfg = json.load(open(PICKS))
        dk = cfg["dark"]
        self.assertEqual([d["rank"] for d in dk["decisions"]], list(range(len(dk["decisions"]))))
        self.assertEqual(sorted(d["id"] for d in dk["decisions"] if d["pass"]), sorted(dk["tile_order"]))
        self.assertEqual(len(cfg["shadow"]["picks"]), 4)
        self.assertEqual(len(cfg["credits"]), 8)


if __name__ == "__main__":
    unittest.main()
