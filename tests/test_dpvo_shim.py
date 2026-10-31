"""DPVO runs with FAH's torch_scatter shim (fah/_vendor) first on its path: the shim's scatter ops and the camera stage's
subprocess environment."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from fah import world

SHIMS = world.VENDOR
sys.path.insert(0, SHIMS)
import torch_scatter  # noqa: E402  (the shim, not PyG's)


class ShimTests(unittest.TestCase):
    def test_the_stand_in_is_imported(self):
        self.assertEqual(os.path.dirname(os.path.abspath(torch_scatter.__file__)), os.path.abspath(SHIMS))

    def test_scatter_ops(self):
        src = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        idx = torch.tensor([0, 1, 0, 1])
        torch.testing.assert_close(torch_scatter.scatter_sum(src, idx, dim=1), torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
        torch.testing.assert_close(torch_scatter.scatter_mean(src, idx, dim=1), torch.tensor([[2.0, 3.0], [6.0, 7.0]]))
        sm = torch_scatter.scatter_softmax(src, idx, dim=1)
        torch.testing.assert_close(torch_scatter.scatter_sum(sm, idx, dim=1), torch.ones(2, 2))
        torch.testing.assert_close(sm[:, [0, 2]], torch.softmax(src[:, [0, 2]], 1))


class CameraStageTests(unittest.TestCase):
    def test_dpvo_subprocess_gets_the_shim_on_its_path(self):
        seen = []

        def run(cmd, cwd=None, check=None, env=None):
            seen.append((cmd, cwd, env))
            raise RuntimeError("stop after the DPVO call")

        with patch.object(world.subprocess, "run", run), self.assertRaises(RuntimeError):
            world.estimate_cameras(tempfile.mkdtemp(), "clip", "camera_motion.pt", "/dpvo", "/dpvo/python", "/moge/python")
        cmd, cwd, env = seen[0]
        self.assertEqual((cmd[0], cwd), ("/dpvo/python", "/dpvo"))
        self.assertEqual(os.path.abspath(env["PYTHONPATH"].split(os.pathsep)[0]), os.path.abspath(SHIMS))  # always first

    def test_camera_configuration(self):
        dpvo, fuse = world.DPVO_ARGS, world.FUSE_ARGS
        arg = lambda a, k: a[a.index(k) + 1]
        self.assertEqual((arg(dpvo, "--scale_calls"), arg(dpvo, "--scale_every"), "--scale_rmed" in dpvo), ("15", "30", True))
        self.assertEqual((arg(fuse, "--ev_lo"), arg(fuse, "--ev_hi")), ("0.5", "1.5"))
        self.assertEqual((arg(fuse, "--t_sigma"), arg(fuse, "--t_sigma_low"), arg(fuse, "--t_sigma_cov")), ("10", "3", "0.7"))
        seen = []

        def run(cmd, cwd=None, check=None, env=None):
            seen.append(cmd)
            raise RuntimeError("stop after the DPVO call")

        with patch.object(world.subprocess, "run", run), self.assertRaises(RuntimeError):
            world.estimate_cameras(tempfile.mkdtemp(), "clip", "camera_motion.pt", "/dpvo")
        self.assertEqual((arg(seen[0], "--scale_calls"), arg(seen[0], "--scale_every")), ("15", "30"))
        from fah import cli

        self.assertEqual(cli.parse_args(["--vid", "x.mp4", "--mode", "world"]).camera_version, "realtime")


class CoverageRuleTests(unittest.TestCase):
    def test_low_coverage_steps_take_the_head_translation_where_they_disagree(self):
        import numpy as np

        from bench.world_fuse import fuse

        n = 6
        P = np.tile(np.eye(4), (n, 1, 1))
        P[:, 0, 3] = np.cumsum([0, 0.01, 0.01, 0.5, 0.01, 0.01])  # one 0.5 m DPVO jump at step 3
        rel = np.tile(np.eye(4), (n, 1, 1))
        rel[1:, 0, 3] = 0.01
        lv = np.full((n, 2), np.log(0.01))  # head errors: 0.01 deg, 0.01 cm
        kw = dict(win=30, thr=0, k_sigma=0, w_head=0.0, t_sigma=10.0)
        everywhere_10 = np.full(n, 10.0)
        T_hi, _, _ = fuse(P, rel, lv, ts_step=everywhere_10, **kw)  # 0.5 m vs 1 cm: beyond 10 x 3 x 0.01 cm -> replaced
        self.assertAlmostEqual(T_hi[3, 0, 3] - T_hi[2, 0, 3], 0.01)
        lv[:, 1] = np.log(5.0)  # head translation error 5 cm: 49 cm apart is within 10 x 3 x 5 cm but beyond 3 x 3 x 5 cm
        T_hi, _, _ = fuse(P, rel, lv, ts_step=everywhere_10, **kw)
        T_lo, _, _ = fuse(P, rel, lv, ts_step=np.where(np.arange(n) == 3, 3.0, 10.0), **kw)
        self.assertAlmostEqual(T_hi[3, 0, 3] - T_hi[2, 0, 3], 0.5)
        self.assertAlmostEqual(T_lo[3, 0, 3] - T_lo[2, 0, 3], 0.01)


class OfflineCameraTests(unittest.TestCase):
    def test_slam_scale_moves_halfway_and_disagreeing_trajectories_fall_back(self):
        import numpy as np

        rng = np.random.default_rng(0)
        n = 50
        ours = np.tile(np.eye(4), (n, 1, 1))
        ours[:, :3, 3] = np.cumsum(rng.normal(0, 0.01, (n, 3)), 0) + np.outer(np.arange(n) / n, [0.1, 0, 0])
        slam = ours.copy()
        slam[:, :3, 3] = (ours[:, :3, 3] - ours[0, :3, 3]) * 2.0 + ours[0, :3, 3]  # SLAM at twice our scale
        T, k, fb = world.fuse_offline(slam, ours)
        self.assertFalse(fb)
        self.assertAlmostEqual(k, 0.5, places=6)
        np.testing.assert_allclose(T[:, :3, 3] - T[0, :3, 3], (slam[:, :3, 3] - slam[0, :3, 3]) * 0.5 ** 0.5, atol=1e-9)
        noise = slam.copy()
        noise[:, :3, 3] = rng.normal(0, 1, (n, 3))  # unrelated trajectory: our camera
        T, k, fb = world.fuse_offline(noise, ours)
        self.assertTrue(fb)
        np.testing.assert_array_equal(T, ours)
        static = np.tile(np.eye(4), (n, 1, 1))  # our camera does not move: SLAM kept at its own scale
        T, k, fb = world.fuse_offline(slam, static)
        self.assertEqual((k, fb), (1.0, False))
        np.testing.assert_array_equal(T, slam)

    def test_offline_camera_needs_the_fitted_meshes(self):
        from fah import cli

        self.assertEqual(cli.parse_args(["--vid", "x.mp4", "--mode", "world", "--camera_version", "offline"]).camera_version, "offline")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli.parse_args(["--vid", "x.mp4", "--mode", "world", "--camera_version", "offline", "--mesh", "none"])


if __name__ == "__main__":
    unittest.main()
