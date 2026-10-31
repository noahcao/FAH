"""scripts/compare_figure.py: hand descriptors, the seeded stratum orders and the selection walk, on synthetic data."""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.compare_figure import blur_score, n_curled, palm_side, seeded, stratum_order, walk  # noqa: E402


def cand(key, subset="hint_nd", seq=None, pa=0.5, pb=0.5, **kw):
    return dict(dict(data="bench2d_v2", key=key, gi=0, subset=subset, seq=seq or key, rel=0.3, nvis=21, ov=0.0, partner=-1, palm=0.5,
                     curled=0, ita=30.0, img_L=50.0, side=100.0, pa=pa, pb=pb, file="", bbox=[0, 0, 10, 10]), **kw)


class CompareFigureTests(unittest.TestCase):
    def test_palm_side_flips_with_handedness(self):
        j = np.zeros((21, 2))
        j[5], j[17] = [1, 0], [0, 1]
        self.assertGreater(palm_side(j, False), 0)
        self.assertLess(palm_side(j, True), 0)

    def test_curled_fingers(self):
        j = np.zeros((21, 2))
        for f in range(1, 5):
            j[2 + 4 * f], j[4 + 4 * f] = [0, 2], [0, 3]  # straight
        j[4 + 4 * 2] = [0, 1]  # middle tip closer to the wrist than its PIP
        self.assertEqual(n_curled(j), 1)

    def test_seeded_order_is_fixed(self):
        rows = list(range(20))
        self.assertEqual(seeded(rows), [rows[i] for i in np.random.default_rng(0).permutation(20)])
        self.assertEqual(seeded(rows), seeded(rows))

    def test_walk_visual_check_and_gain(self):
        order = [cand("a", pa=0.5, pb=0.5), cand("b", pa=0.5, pb=0.55), cand("c", pa=0.5, pb=0.7)]
        checks = [dict(key="a", gi=0, visual=False), dict(key="b", gi=0, visual=True), dict(key="c", gi=0, visual=True)]
        self.assertEqual(walk(order, checks, lambda r: True)[0], 1)
        self.assertEqual(walk(order, checks, lambda r: r["pb"] - r["pa"] >= 0.1 - 1e-9)[0], 2)
        with self.assertRaises(RuntimeError):  # the recorded checks must belong to this order
            walk(order, [dict(key="x", gi=0, visual=True)], lambda r: True)

    def test_filter_modes_draw_different_orders(self):
        C = [cand(f"k{i}", seq="used" if i % 3 == 0 else f"s{i}") for i in range(30)]
        have = {(r["key"], r["gi"]) for r in C}
        spec = dict(exclude_seqs=["used"])
        before = stratum_order("s3_third_object", dict(spec, mode="filter_then_order"), C, have, lambda r: True, None)
        after = stratum_order("s3_third_object", dict(spec, mode="order_then_filter"), C, have, lambda r: True, None)
        self.assertEqual(sorted(r["key"] for r in before), sorted(r["key"] for r in after))
        self.assertTrue(all(r["seq"] != "used" for r in before + after))
        self.assertNotEqual([r["key"] for r in before], [r["key"] for r in after])

    def test_blur_score_lower_when_blurred(self):
        rng = np.random.default_rng(1)
        img = (rng.random((200, 200)) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img, (0, 0), 4)
        box = [20, 20, 180, 180]
        self.assertLess(blur_score(blurred, box), blur_score(img, box))


if __name__ == "__main__":
    unittest.main()
