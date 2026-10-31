"""bench.world --protocol wham (WHAM's W-MPJPE100 / WA-MPJPE100 / RTE): CPU-only checks on synthetic tracks."""
import os
import pickle
import tempfile
import unittest

import numpy as np

from bench import world


def rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return (np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]) @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            @ np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]]))


def track(T, seed=0):
    rng = np.random.default_rng(seed)
    hand = rng.normal(0, 0.03, (21, 3))
    path = np.cumsum(rng.normal(0, 0.004, (T, 3)), 0)
    return hand[None] + path[:, None]  # (T, 21, 3) GT world joints (m)


class WhamErrorTests(unittest.TestCase):
    def test_a_similarity_transformed_run_has_zero_error(self):
        G = track(230)
        P = 0.7 * G @ rot(0.3, -0.2, 0.5).T + np.array([0.1, -0.4, 2.0])
        w, wa, rte = world.wham_errors(P, G, 100)
        self.assertEqual((len(w), len(wa)), (230, 230))  # chunks of 100, 100 and the 30-frame remainder
        self.assertLess(max(w.max(), wa.max()), 1e-6)
        self.assertGreater(rte.max(), 1.0)  # RTE: rigid fit only, so the 0.7 scale stays an error

    def test_chunks_are_aligned_independently(self):
        G = track(200, 1)
        P = G.copy()
        P[100:] = P[100:] @ rot(0.2, 0, 0).T + np.array([0.05, 0, 0])  # a rigid jump at the chunk boundary
        w, wa, _ = world.wham_errors(P, G, 100)
        self.assertLess(max(w.max(), wa.max()), 1e-6)  # each chunk has its own Sim(3) fit
        w0, wa0, _ = world.wham_errors(P, G, 0)  # chunk 0 = the whole run: one fit, the jump is an error
        self.assertGreater(wa0.mean(), 1.0)


class WhamProtocolTests(unittest.TestCase):
    def test_missing_frames_split_runs_and_short_runs_are_dropped(self):
        T, gaps = 140, {20, 21, 130}  # runs 0-19, 22-129, 131-139 (9 frames < min_run 16)
        G = track(T, 2)
        P = G + np.random.default_rng(3).normal(0, 0.005, G.shape)
        frames = []
        for i in range(T):
            fr = dict(index=i, key=f"{i:06d}", T_world_cam=np.eye(4), pred=[], gt=[])
            fr["gt"].append(dict(bbox=[100, 100, 200, 200], is_right=True, joints3d=G[i]))
            if i not in gaps:
                fr["pred"].append(dict(bbox=[100, 100, 200, 200], joints3d=P[i]))
            frames.append(fr)
        with tempfile.TemporaryDirectory() as d:
            pickle.dump(dict(clip="synth", method="synth", frames=frames), open(os.path.join(d, "synth.pkl"), "wb"))
            res = world.evaluate_world(d, "gt")
            rigid = world.evaluate_world(d, "gt", protocol="rigid")
        self.assertEqual(res["protocol"], "wham")  # the default
        self.assertEqual((res["num_runs"], res["num_short_runs"]), (2, 1))
        self.assertEqual((res["num_frames"], res["num_tp_frames"]), (20 + 108, T - len(gaps)))
        runs = [world.wham_errors(P[a:b], G[a:b], 100) for a, b in ((0, 20), (22, 130))]
        self.assertAlmostEqual(res["w_mpjpe"], float(np.concatenate([r[0] for r in runs]).mean()), places=9)
        self.assertAlmostEqual(res["wa_mpjpe"], float(np.concatenate([r[1] for r in runs]).mean()), places=9)
        self.assertAlmostEqual(res["rte"], float(np.concatenate([r[2] for r in runs]).mean()), places=9)
        self.assertEqual(res["w_mpjpe_right"], res["w_mpjpe"])
        self.assertIsNone(res["w_mpjpe_left"])
        self.assertNotIn("protocol", rigid)
        self.assertEqual(rigid["num_frames"], T - len(gaps))  # rigid: one track, gaps allowed


if __name__ == "__main__":
    unittest.main()
