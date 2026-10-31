"""Two hands that share one detector box (crossed / praying hands): the box's instance mask is split by the hands' mesh
silhouettes instead of going wholly to the first hand (bench.methods.fah.split_shared_masks)."""
import unittest

import numpy as np

from bench.methods.fah import split_shared_masks

K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]], np.float32)
QUAD = np.array([[0, 1, 2], [0, 2, 3]])


def quad(x0, x1, z=1.0):
    """A square mesh spanning image columns ~x0..x1 and rows ~30..70 at depth z (K above)."""
    X = lambda u: (u - 50) / 100 * z
    return np.array([[X(x0), X(30), z], [X(x1), X(30), z], [X(x1), X(70), z], [X(x0), X(70), z]])


def box_mask(x0=20, x1=80):
    m = np.zeros((100, 100), bool)
    m[30:70, x0:x1] = True
    return m


class MaskSplitTests(unittest.TestCase):
    def test_shared_box_mask_is_split(self):
        hands = [dict(vertices_mano=quad(20, 50), cam_int=K), dict(vertices_mano=quad(50, 80, z=1.2), cam_int=K)]
        full = box_mask()
        dm = split_shared_masks([full.copy(), np.zeros_like(full)], [0, 0], hands, QUAD, (100, 100))  # the tie: hand 2 empty
        self.assertTrue(dm[0].any() and dm[1].any())
        self.assertFalse((dm[0] & dm[1]).any())
        np.testing.assert_array_equal(dm[0] | dm[1], full)
        self.assertGreater(dm[0][:, 20:50].mean(), dm[0][:, 50:80].mean())  # each hand over its own silhouette
        self.assertGreater(dm[1][:, 50:80].mean(), dm[1][:, 20:50].mean())

    def test_uncovered_pixels_stay_with_the_first_hand(self):
        hands = [dict(vertices_mano=quad(25, 45), cam_int=K), dict(vertices_mano=quad(55, 75), cam_int=K)]
        full = box_mask()
        dm = split_shared_masks([full.copy(), np.zeros_like(full)], [0, 0], hands, QUAD, (100, 100))
        self.assertTrue(dm[0][:, 46:54].any())  # the gap between the meshes
        np.testing.assert_array_equal(dm[0] | dm[1], full)

    def test_hands_with_their_own_boxes_are_unchanged(self):
        a, b = box_mask(20, 50), box_mask(50, 80)
        hands = [dict(vertices_mano=quad(20, 50), cam_int=K), dict(vertices_mano=quad(50, 80), cam_int=K)]
        dm = split_shared_masks([a.copy(), b.copy(), None], [0, 1, None], hands + [dict(cam_int=K)], QUAD, (100, 100))
        np.testing.assert_array_equal(dm[0], a)
        np.testing.assert_array_equal(dm[1], b)
        self.assertIsNone(dm[2])


if __name__ == "__main__":
    unittest.main()
