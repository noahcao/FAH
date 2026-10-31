"""Contact data builders and the HACO adapter: CPU-only checks on synthetic data (no datasets, no weights)."""
import importlib.util
import json
import os
import pickle
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from bench.datasets.contactclip import ContactClip
from bench.methods.haco import crop_box, haco_patch, mask_box

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cube(side=0.1):
    """A closed cube mesh (metres, centred at the origin): 8 vertices, 12 outward-facing triangles."""
    h = side / 2
    V = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], np.float64)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    return V, F


@unittest.skipUnless(importlib.util.find_spec("open3d") and importlib.util.find_spec("igl"), "needs open3d + libigl (pip install .[contact])")
class SignedDistanceTests(unittest.TestCase):
    def test_inside_negative_outside_positive_clipped(self):
        from train.contact_sdf import CLIP_MM, MeshSDF

        V, F = cube()
        sdf = MeshSDF(V, F)
        d = sdf(np.array([[0, 0, 0], [0.07, 0, 0], [0.045, 0, 0], [0, 0, 2.0]]))
        np.testing.assert_allclose(d[:3], [-50, 20, -5], atol=1e-3)  # centre 50 mm inside, 20 mm outside a face, 5 mm inside
        self.assertEqual(d[3], CLIP_MM)
        T = np.eye(4)
        T[:3, 3] = [0.0, 0.0, 1.0]  # object 1 m in front of the camera
        np.testing.assert_allclose(sdf.posed(np.array([[0, 0, 1.0]]), T), [-50], atol=1e-3)


class HACOCropTests(unittest.TestCase):
    def test_crop_rules(self):
        np.testing.assert_allclose(crop_box([10, 20, 30, 60], "hand2x", 640, 480), (0, 0, 40, 80))
        np.testing.assert_allclose(crop_box([10, 20, 30, 60], "box13", 640, 480), (7, 14, 26, 52))
        np.testing.assert_allclose(crop_box([10, 20, 30, 60], "mask13", 640, 480), (0, 0, 44, 87))  # hi = (43, 86), clipped lo = (0, 0)

    def test_left_hand_is_the_mirrored_right_crop(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (120, 160, 3), np.uint8)
        box = (20.0, 30.0, 50.0, 40.0)
        left = haco_patch(img, box, False)
        right = haco_patch(img[:, ::-1], (160 - (box[0] + box[2]), box[1], box[2], box[3]), True)
        self.assertEqual(left.shape, (256, 256, 3))
        np.testing.assert_array_equal(left, right)

    def test_mask_box(self):
        m = np.zeros((100, 200), bool)
        m[40:60, 50:90] = True  # 40 x 20 px
        self.assertEqual(mask_box(m, 1.3), (24, 27, 92, 46))


class ContactClipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_hands_and_ignore(self):
        d = os.path.join(self.tmp, "clips", "c0")
        os.makedirs(os.path.join(d, "frames"))
        K = [[500.0, 0, 160], [0, 500.0, 120], [0, 0, 1]]
        json.dump(dict(K=K, hw=[240, 320], first=0, last=1, frames="frames/%06d.jpg", dataset="toy", sequence="s/0"), open(os.path.join(d, "meta.json"), "w"))
        for f in (0, 1):
            cv2.imwrite(os.path.join(d, "frames", f"{f:06d}.jpg"), np.zeros((240, 320, 3), np.uint8))
        rng = np.random.default_rng(0)
        inside = np.c_[rng.uniform(-0.05, 0.05, (778, 2)), np.full(778, 0.5)]  # projects inside the image
        half = inside.copy()
        half[:500, 0] += 0.5  # 500 of 778 vertices leave the image -> visible 36 % -> ignore
        np.savez(os.path.join(d, "gt.npz"), frame=np.array([0, 1]), is_right=np.array([True, False]), verts=np.stack([inside, half]).astype(np.float32),
                 joints=np.zeros((2, 21, 3), np.float32) + [0, 0, 0.5], sdf=np.zeros((2, 778), np.float16), valid=np.array([True, False]),
                 T_obj=np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)))
        clip = ContactClip(d)
        self.assertEqual(len(clip), 2)
        f0, f1 = clip.frame(0), clip.frame(1, image=False)
        self.assertEqual((len(f0["hands"]), len(f0["ignore"])), (1, 0))
        self.assertEqual((len(f1["hands"]), len(f1["ignore"])), (0, 1))
        self.assertTrue(f0["hands"][0]["is_right"] and f0["hands"][0]["valid"])
        self.assertEqual(f0["image"].shape, (240, 320, 3))


class HOI4DSplitTests(unittest.TestCase):
    def test_split_and_no_chair(self):
        s = json.load(open(os.path.join(REPO, "bench", "splits", "hoi4d_contact.json")))
        nc = json.load(open(os.path.join(REPO, "bench", "splits", "hoi4d_contact_test_nochair.json")))
        self.assertFalse(set(s["train"]) & set(s["test"]))
        self.assertTrue(all(x.split("/")[1] == "H3" for x in s["test"]))
        self.assertEqual(nc["test"], [x for x in s["test"] if x.split("/")[2] != "C20"])
        self.assertEqual((len(s["test"]), len(nc["test"])), (176, 149))
        self.assertEqual(nc["clips"], [x.replace("/", "_") for x in nc["test"]])

    def test_numpy_only_unpickler(self):
        import io

        from train.build_hoi4d_contact import NumpyUnpickler

        ok = pickle.dumps(dict(poseCoeff=np.zeros(48), beta=np.zeros(10)))
        self.assertEqual(NumpyUnpickler(io.BytesIO(ok)).load()["poseCoeff"].shape, (48,))
        with self.assertRaises(pickle.UnpicklingError):
            NumpyUnpickler(io.BytesIO(pickle.dumps(os.getcwd))).load()


class MOWMetricTests(unittest.TestCase):
    def test_per_image_f1(self):
        import sys

        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from contact_mow_pairs import per_image

        lab = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], bool)
        prob = np.array([[0.9, 0.1, 0.9, 0.1], [0.1, 0.1, 0.1, 0.1]])
        f1, (tp, fp, fn) = per_image(prob, lab, 0.5)
        np.testing.assert_allclose(f1, [0.5, 0.0])
        np.testing.assert_array_equal(np.c_[tp, fp, fn], [[1, 1, 1], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
