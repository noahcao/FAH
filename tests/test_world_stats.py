"""World-space statistics and diagnostics (bench.world_reps, bench.world_bootstrap, bench.world_offline, bench.world_paper,
bench.world_camdiag, bench.datasets.egodex_world merge): CPU-only checks on synthetic clips."""
import os
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from bench import world, world_bootstrap, world_camdiag, world_paper, world_reps
from fah.world import fuse_offline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return (np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]) @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            @ np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]]))


def write_clips(d, n_clips=4, T=150, noise=0.005, seed=0):
    """Clips with one right hand moving along a random walk, GT camera poses in the pickles; predictions = GT + noise,
    with a gap (frames 60-61) in clip 0 and a short last run in clip 1."""
    rng = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    gt_T = {}
    for c in range(n_clips):
        hand = rng.normal(0, 0.03, (21, 3))
        path = np.cumsum(rng.normal(0, 0.004, (T, 3)), 0) + [0, 0, 0.5]
        cam = np.cumsum(rng.normal(0, 0.01, (T, 3)), 0)
        frames, Ts = [], []
        for i in range(T):
            Tw = np.eye(4)
            Tw[:3, 3] = cam[i]
            Ts.append(Tw)
            G = hand + path[i]  # camera frame
            fr = dict(index=i, key=f"{i:06d}", T_world_cam=Tw, pred=[], gt=[dict(bbox=[100, 100, 200, 200], is_right=True, joints3d=G)])
            if not (c == 0 and i in (60, 61)) and not (c == 1 and i == T - 10):
                fr["pred"].append(dict(bbox=[100, 100, 200, 200], joints3d=G + rng.normal(0, noise, G.shape)))
            frames.append(fr)
        name = f"clip-{c:03d}"
        pickle.dump(dict(clip=name, method="synth", frames=frames), open(os.path.join(d, f"{name}.pkl"), "wb"))
        gt_T[name] = np.stack(Ts)
    return gt_T


def write_cams(d, gt_T, scale, seed):
    """A camera realisation: GT poses with the translation scaled and jittered."""
    rng = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    for name, T in gt_T.items():
        T = T.copy()
        T[:, :3, 3] = T[:, :3, 3] * scale + np.cumsum(rng.normal(0, 0.002, (len(T), 3)), 0)
        np.savez(os.path.join(d, f"{name}.npz"), index=np.arange(len(T)), T_world_cam=T)


class WorldStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        r = cls.tmp.name
        cls.hands, cls.hands2 = os.path.join(r, "hands"), os.path.join(r, "hands2")
        gt_T = write_clips(cls.hands)
        write_clips(cls.hands2, noise=0.01, seed=0)
        cls.cams = [os.path.join(r, f"cams_{i}") for i in range(3)]
        for i, (c, s) in enumerate(zip(cls.cams, (0.9, 1.0, 1.15))):
            write_cams(c, gt_T, s, i)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_reps_mean_and_sample_sd_over_separately_scored_realisations(self):
        res = world_reps.score_rows({"a": (self.hands, self.cams)}, (100, 0), workers=2)
        for ch in (100, 0):
            per = [world.evaluate_world(self.hands, c, protocol="wham", chunk=ch) for c in self.cams]
            for k in world_reps.KEYS:
                v = [p[k] for p in per]
                self.assertAlmostEqual(res["a"][ch][k]["mean"], np.mean(v), places=9)
                self.assertAlmostEqual(res["a"][ch][k]["sd"], np.std(v, ddof=1), places=9)
                self.assertEqual([p[k] for p in res["a"][ch]["per_realisation"]], v)
        self.assertIsNone(world_reps.aggregate([dict(w_mpjpe=1.0, wa_mpjpe=1.0, rte=1.0)])["w_mpjpe"]["sd"])

    def test_bootstrap_full_sample_matches_evaluate_world_and_paired_diffs(self):
        runs = {"a0": self.hands, "a1": self.hands, "a2": self.hands, "b": self.hands2, "same": self.hands}
        cams = {"a0": self.cams[0], "a1": self.cams[1], "a2": self.cams[2], "b": self.cams[1], "same": self.cams[1]}
        out = world_bootstrap.bootstrap(runs, cams, "a1", "egostandard", {"a": ["a0", "a1", "a2"]}, boot=200, workers=2)
        self.assertEqual((out["num_clips"], out["num_sequences"]), (4, 4))  # egostandard: one sequence per clip
        for r in ("a0", "b"):
            ref = world.evaluate_world(runs[r], cams[r], protocol="wham", chunk=100)
            for k in world_bootstrap.KEYS:
                self.assertAlmostEqual(out["point"][r][k], ref[k], places=9)
        for k in world_bootstrap.KEYS:  # avg = mean of the members' metrics, not of their predictions
            self.assertAlmostEqual(out["point"]["a"][k], np.mean([out["point"][m][k] for m in ("a0", "a1", "a2")]), places=12)
            d = out["diff"]["same-a1"][k]  # identical run and camera: every paired resample differs by 0
            self.assertEqual((d["point"], d["lo"], d["hi"]), (0.0, 0.0, 0.0))
            d = out["diff"]["b-a1"][k]
            self.assertLessEqual(d["lo"], d["point"])
            self.assertLessEqual(d["point"], d["hi"])
        st = out["stats"]["a0"]
        self.assertEqual(st["runs"], 4 + 1)  # clip 0 split at the gap; clip 1's 9-frame tail run (< 16) dropped
        self.assertEqual(st["frames"], 4 * 150 - 2 - 10)

    def test_bootstrap_common_support_restricts_both_runs(self):
        common = world.common_support([self.hands, self.hands2], 0.3)
        out = world_bootstrap.bootstrap({"a": self.hands, "b": self.hands2}, {"a": "gt", "b": "gt"}, "a", "egostandard",
                                        common=common, boot=50, workers=2)
        self.assertEqual(out["stats"]["a"]["frames"], out["stats"]["b"]["frames"])

    def test_track_stats(self):
        runs = [(n, float(n)) for n in range(16, 36)]  # 20 runs of 16..35 frames, W error sum = length
        st = world_bootstrap.track_stats(runs)
        self.assertEqual((st["runs"], st["frames"], st["len_median"]), (20, sum(range(16, 36)), 25.5))
        self.assertAlmostEqual(st["top10_w_share"], sum(range(26, 36)) / sum(range(16, 36)))

    def test_offline_cli_fuses_scale_halfway(self):
        with tempfile.TemporaryDirectory() as d:
            T = np.tile(np.eye(4), (50, 1, 1))
            T[:, :3, 3] = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (50, 3)), 0)
            ours = T.copy()
            ours[:, :3, 3] *= 4.0  # the real-time camera's scale is 4x SLAM's
            for name, x in (("slam", T), ("ours", ours)):
                os.makedirs(os.path.join(d, name))
                np.savez(os.path.join(d, name, "clip.npz"), index=np.arange(50), T_world_cam=x)
            subprocess.run([sys.executable, "-m", "bench.world_offline", "--slam", os.path.join(d, "slam"), "--ours",
                            os.path.join(d, "ours"), "--out", os.path.join(d, "off")], cwd=REPO, check=True, capture_output=True)
            z = np.load(os.path.join(d, "off", "clip.npz"))
            self.assertAlmostEqual(float(z["k_ours_over_slam"]), 4.0, places=6)
            self.assertFalse(bool(z["fallback"]))
            want, _, _ = fuse_offline(T, ours)
            np.testing.assert_allclose(z["T_world_cam"], want)
            c0 = T[0, :3, 3]
            np.testing.assert_allclose(z["T_world_cam"][:, :3, 3], c0 + (T[:, :3, 3] - c0) * 2.0, atol=1e-12)  # 4 ** 0.5

    def test_camdiag_hybrids_replace_one_part_with_gt(self):
        rng = np.random.default_rng(1)
        n = 200
        gt = np.tile(np.eye(4), (n, 1, 1))
        gt[:, :3, 3] = np.cumsum(rng.normal(0, 0.01, (n, 3)), 0)
        for i in range(n):
            gt[i, :3, :3] = rot(0.002 * i, -0.001 * i, 0.003 * i)
        h = world_camdiag.hybrids(gt.copy(), gt, seg=81)  # a perfect camera: every hybrid is the GT camera
        for k, T in h.items():
            np.testing.assert_allclose(T, gt, atol=1e-9, err_msg=k)
        T = gt.copy()
        T[:, :3, 3] *= 0.7  # scale error only
        T[:, :3, :3] = T[:, :3, :3] @ rot(0.05, 0, 0)  # and a constant orientation offset
        h = world_camdiag.hybrids(T, gt, seg=81)
        np.testing.assert_allclose(h["gt_rotation"][:, :3, :3], gt[:, :3, :3], atol=1e-12)
        np.testing.assert_allclose(h["gt_centres"][:, :3, 3], gt[:, :3, 3], atol=1e-12)
        d = h["gt_clipscale"][:, :3, 3] - h["gt_clipscale"][:1, :3, 3]  # clip scale restores the GT path
        np.testing.assert_allclose(d, gt[:, :3, 3] - gt[:1, :3, 3], atol=1e-9)

    def test_egodex_merge_puts_predictions_next_to_gt(self):
        with tempfile.TemporaryDirectory() as d:
            for sub in ("gt", "pred"):
                os.makedirs(os.path.join(d, sub))
            frames = [dict(index=i, key=f"{i:05d}", K=np.eye(3), T_world_cam=np.eye(4), gt=[dict(is_right=True)], pred=[]) for i in range(3)]
            pickle.dump(dict(clip="ep", dataset="egodex", frames=frames), open(os.path.join(d, "gt", "ep.pkl"), "wb"))
            pred = [dict(index=i, pred=[dict(joints3d=np.full((21, 3), i))]) for i in range(3)]
            pickle.dump(dict(clip="ep", method="m", frames=pred), open(os.path.join(d, "pred", "ep.pkl"), "wb"))
            subprocess.run([sys.executable, "-m", "bench.datasets.egodex_world", "merge", "--gt", os.path.join(d, "gt"), "--pred",
                            os.path.join(d, "pred"), "--out", os.path.join(d, "out")], cwd=REPO, check=True, capture_output=True)
            out = pickle.load(open(os.path.join(d, "out", "ep.pkl"), "rb"))
            self.assertEqual(out["method"], "m")
            self.assertEqual([float(fr["pred"][0]["joints3d"][0, 0]) for fr in out["frames"]], [0.0, 1.0, 2.0])
            self.assertEqual([len(fr["gt"]) for fr in out["frames"]], [1, 1, 1])

    def test_stage_fps_drops_the_warm_up_clip(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, w = (os.path.join(d, f) for f in ("a.log", "b.log", "w.log"))
            open(a, "w").write("100.0 [stamp] start\n150.0 [hawor] c0: 900 frames\n250.0 [hawor] c1: 900 frames\n"
                               "330.0 noise\n350.0 [hawor] c2: 900 frames, 3 ms\n")
            open(b, "w").write("10.0 [world_slam] c0: 900 frames, 20 keyframes\n60.0 [world_slam] c1: 900 frames\n"
                               "110.0 [world_slam] c2: 900 frames\n")
            open(w, "w").write("1.0 [stamp] start\n2.0 done\n2.0 [stamp] wall 10.000\n")
            out = subprocess.run([sys.executable, "scripts/world_stage_fps.py", "fps", "--stages", a, b, "--whole", f"{w}:0.5"],
                                 cwd=REPO, check=True, capture_output=True, text=True).stdout
        self.assertIn("1800 frames in 200.0 s (2 clips) = 9.00 fps", out)  # c0 (warm-up, model loading) dropped
        self.assertIn("pipeline: 1800 frames in 305.0 s = 5.90 fps", out)  # 200 + 100 + 10 x 0.5

    def test_paper_table_is_complete(self):
        for ds in ("hot3d", "egostandard"):
            self.assertEqual(set(world_paper.PAPER[ds]), set(world_paper.ROWS) | set(world_paper.GT_ROWS))
            for r, (means, sds) in world_paper.PAPER[ds].items():
                self.assertEqual(len(means), len(world_paper.CELLS))
                self.assertEqual(sds is None, r in world_paper.GT_ROWS)
            for r in world_paper.TRACKS[ds]:
                self.assertIn(r, world_paper.ROWS)
        for iv in world_paper.INTERVALS:
            self.assertIn(iv[3], world_paper.ROWS)
            self.assertIn(iv[4], world_paper.ROWS)
            self.assertLessEqual(iv[6][1], iv[6][0])
            self.assertLessEqual(iv[6][0], iv[6][2])
        self.assertTrue(world_paper.same(-0.004, -0.00) and world_paper.same(22.6849, 22.68) and not world_paper.same(22.686, 22.68))


if __name__ == "__main__":
    unittest.main()
