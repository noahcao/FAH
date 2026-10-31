"""CPU-only checks of the paper analyses (bench.analysis): fingertip extent, output agreement and the dense-vs-parameters
alignment measures on synthetic hands; no weights, no MANO files."""
import unittest

import numpy as np

from bench.analysis import agreement, alignment, tips


def hand2d(x0=100.0, y0=200.0):
    """21 2D joints: wrist + five straight fingers of four 10 px segments, pointing up."""
    j = np.zeros((21, 2))
    j[0] = (x0, y0)
    for f in range(5):
        for k in range(4):
            j[1 + 4 * f + k] = (x0 - 40 + 20 * f, y0 - 30 - 10 * k)
    return j


def frame2d(pred):
    g = hand2d()
    gt = dict(joints2d=g, vis=np.ones(21), bbox=np.array([50.0, 150.0, 150.0, 210.0]))
    return dict(key="cmu/x", image_size=(480, 640), gt=[gt], pred=pred)


def camera_hand(z=0.5):
    """(21, 3) camera-frame joints of a hand about z m from the camera, and an intrinsics matrix."""
    j2 = hand2d(320.0, 300.0)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    zs = z + 0.002 * np.arange(21)
    X = np.stack([(j2[:, 0] - 320) * zs / 600, (j2[:, 1] - 240) * zs / 600, zs], 1)
    return X, K


class TipExtentTests(unittest.TestCase):
    def test_signed_error_and_length(self):
        g = hand2d()
        p = g.copy()
        for f in range(5):
            dip, tip = 3 + 4 * f, 4 + 4 * f
            p[tip] = g[tip] + 0.25 * (g[tip] - g[dip])  # every tip 0.25 phalanx beyond the label
        res = tips.summarize(tips.score_frames([frame2d([dict(joints2d=p)])]))
        self.assertAlmostEqual(res["all"]["tip"], 0.25, places=6)
        self.assertEqual(res["all"]["short"], 0.0)
        self.assertEqual(res["all"]["n"], 5)
        self.assertAlmostEqual(res["all"]["length"], 32.5 / 30.0, places=6)
        self.assertIn("cmu", res)

    def test_furthest_region_point(self):
        g = hand2d()
        surf = np.tile(g[:1], (778, 1)) + 0.5  # every surface point near the wrist ...
        region = {f: np.array([10 * i, 10 * i + 1]) for i, f in enumerate(tips.FINGERS)}
        for i, (f, (_, _, dip, tip)) in enumerate(tips.FINGERS.items()):
            surf[region[f][1]] = g[tip] - 0.5 * (g[tip] - g[dip])  # ... except one region point per finger, 0.5 short
        res = tips.summarize(tips.score_frames([frame2d([dict(joints2d=g.copy(), surface2d_head=surf)])], "surf2d", region=region))
        self.assertAlmostEqual(res["all"]["tip"], -0.5, places=6)
        self.assertEqual(res["all"]["short"], 1.0)

    def test_default_cameras(self):
        self.assertAlmostEqual(tips.default_K(480, 640)[0, 0], 800.0)
        self.assertAlmostEqual(tips.default_K(480, 640, "wilor")[0, 0], 5000 / 256 * 640)
        np.testing.assert_allclose(tips.default_K(480, 640)[:2, 2], [320, 240])


class AlignmentTests(unittest.TestCase):
    def test_rigid_and_translation(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(30, 3))
        c, s = np.cos(0.7), np.sin(0.7)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        B = A @ R.T + [0.1, -0.2, 0.3]
        np.testing.assert_allclose(alignment.rigid(A, B), B, atol=1e-9)
        X, K = camera_hand()
        np.testing.assert_allclose(alignment.best_translation(X - X[:1] + 5.0, alignment.proj(X, K), K), X, atol=1e-6)

    def test_identical_outputs_agree(self):
        X, K = camera_hand()
        uv = alignment.proj(X, K)
        box = np.r_[uv.min(0), uv.max(0)]
        verts = np.zeros((778, 3)) + X.mean(0)
        pred = dict(joints3d=X, joints2d=uv, verts3d=verts, verts3d_dense=verts, bbox=box)
        fr = dict(K=K, key="x/0", gt=[dict(joints3d=X, bbox=box)], pred=[pred])
        sc = alignment.Scorer(lambda v: X, np.arange(400), "dense")
        sc.add_frame(fr)
        out = sc.summary()
        for k in ("A_own", "A_fit", "B_joints", "D_3d", "D_2d", "C_3d", "e2d", "emesh"):
            self.assertLess(out[k]["median"], 1e-6, k)
        self.assertLess(out["pa3d"]["median"], 1e-6)
        sc = alignment.Scorer(lambda v: X, np.arange(400), "param")
        sc.add_frame(fr)
        self.assertNotIn("D_3d", sc.summary())


class AgreementTests(unittest.TestCase):
    def test_pck_and_disagreement(self):
        X, K = camera_hand()
        uv = alignment.proj(X, K)
        box = np.r_[uv.min(0), uv.max(0)]
        S = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
        off = np.array([0.08 * S, 0.0])  # direct 2D off by 0.08 hand sizes: inside PCK@0.1, outside PCK@0.05
        pred = dict(joints3d=X, joints2d=uv + off, verts3d=np.zeros((778, 3)) + X.mean(0), bbox=box)
        frames = [dict(K=K, gt=[dict(joints3d=X, bbox=box)], pred=[pred])] * 4
        out = agreement.summarize(agreement.score_frames(frames, lambda v: X, every=2))
        self.assertEqual(out["e2d"]["hands"], 2)
        self.assertEqual(out["e2d"]["pck05"], 0.0)
        self.assertEqual(out["e2d"]["pck10"], 1.0)
        self.assertEqual(out["e3d"]["pck05"], 1.0)
        self.assertLess(out["pa3d"], 1e-6)
        self.assertAlmostEqual(out["d_2d_3d"]["median"], 0.08, places=6)


if __name__ == "__main__":
    unittest.main()
