"""Perspective-crop outputs must share the original image/camera coordinate system."""
import copy
import unittest

import cv2
import numpy as np

from fah.estimator import perspective_view, unwarp_hand


class PerspectiveOutputTests(unittest.TestCase):
    def test_dense_and_parametric_meshes_return_to_the_same_camera(self):
        K = np.array([[40., 0, 32], [0, 42., 24], [0, 0, 1]])
        R, _, H = perspective_view(K, np.array([2., 5., 24., 30.]))
        points = np.array([[.1, -.02, .5], [.12, .03, .55]], np.float32)
        virtual = points @ R.T
        hand = dict(vertices_mano=virtual.copy(), vertices_dense=virtual.copy(),
                    surface_3d_head_rel=virtual.copy())
        result = unwarp_hand(hand, R, H, K, 48, 64)
        for key in hand:
            if key != "cam_int":
                np.testing.assert_allclose(result[key], points, atol=1e-7)

    def test_part_maps_and_masks_agree_after_perspective_unwarping_on_both_sides(self):
        height, width = 48, 64
        K = np.array([[40., 0, 32], [0, 42., 24], [0, 0, 1]])
        R, _, H = perspective_view(K, np.array([2., 5., 24., 30.]))
        F = np.diag([-1., 1., 1.])
        mirror = np.array([[-1., 0, width - 1], [0, 1, 0], [0, 0, 1]])
        for left in (False, True):
            with self.subTest(left=left):
                r, hom = (F @ R @ F, mirror @ H @ mirror) if left else (R, H)
                x0, y0 = (39, 6) if left else (3, 6)
                fg = np.full((22, 20), 2, np.uint8)
                fg[:, 10:] = 7
                labels = fg.copy()
                labels[3:7, 8:12] = 0
                canvas = np.zeros((height, width), np.uint8)
                canvas[y0:y0 + 22, x0:x0 + 20] = labels
                bbox = np.array([0., 0., width, height])
                hand = dict(parts_roi=(x0, y0, labels), parts_fg_roi=(x0, y0, fg), bbox=bbox.copy(),
                            mask=canvas > 0)
                result = unwarp_hand(hand, r, hom, K, height, width)
                expected = cv2.warpPerspective(canvas, np.linalg.inv(hom), (width, height), flags=cv2.INTER_NEAREST)
                actual = np.zeros_like(expected)
                x, y, roi = result["parts_roi"]
                actual[y:y + roi.shape[0], x:x + roi.shape[1]] = roi
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual > 0, result["mask"])
                # Foreground candidates must cover every detector-box pixel for subsequent mask conditioning.
                x, y, roi = result["parts_fg_roi"]
                self.assertEqual((x, y, *roi.shape), (0, 0, height, width))
                self.assertTrue(np.all((roi >= 1) & (roi <= 16)))
                np.testing.assert_array_equal(roi[expected > 0], expected[expected > 0])
                np.testing.assert_array_equal(result["bbox"], bbox)
                np.testing.assert_array_equal(labels, canvas[y0:y0 + 22, x0:x0 + 20])

    def test_identity_transform_preserves_labels_and_clips_to_frame(self):
        roi = np.array([[2, 3, 4], [5, 6, 7]], np.uint8)
        hand = dict(parts_roi=(-1, 2, roi), parts_fg_roi=(-1, 2, roi), bbox=np.array([0., 2., 2., 4.]))
        original = copy.deepcopy(hand)
        result = unwarp_hand(hand, np.eye(3), np.eye(3), np.eye(3), 4, 4)
        canvas = np.zeros((4, 4), np.uint8)
        x, y, labels = result["parts_roi"]
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + labels.shape[1], 4)
        self.assertLessEqual(y + labels.shape[0], 4)
        canvas[y:y + labels.shape[0], x:x + labels.shape[1]] = labels
        np.testing.assert_array_equal(canvas[2:4, :2], roi[:, 1:])
        np.testing.assert_array_equal(original["parts_roi"][2], roi)


if __name__ == "__main__":
    unittest.main()
