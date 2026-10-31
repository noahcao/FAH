"""Opt-in video identity rules (bench.methods.fah side_track / edge_score + edge_track): off by default and in the settings,
--identity switches them on; with weights (FAH_TEST_VIDEO, a short public HOT3D clip): defaults are bit-identical to the flags
given explicitly off, and the flags on run."""
import inspect
import os
import unittest

import numpy as np

from bench.methods.fah import FAHMethod
from fah import api, cli, video_demo


class DefaultsTests(unittest.TestCase):
    def test_off_by_default_and_in_the_settings(self):
        p = inspect.signature(FAHMethod.__init__).parameters
        self.assertEqual((p["side_track"].default, p["edge_score"].default, p["edge_track"].default), (False, 0.0, False))
        for mode in ("image", "video"):  # the settings on top of config.json set none of them
            self.assertFalse(set(api.IDENTITY) & set(api.SETTINGS[mode]), mode)

    def test_identity_flag(self):
        a = cli.parse_args(["--vid", "x.mp4", "--identity"])
        self.assertEqual({k: cli.model_kwargs(a)[k] for k in api.IDENTITY}, api.IDENTITY)
        self.assertTrue(cli.meta(a)["identity"])
        b = cli.parse_args(["--vid", "x.mp4"])
        self.assertFalse(set(api.IDENTITY) & set(cli.model_kwargs(b)))
        self.assertTrue(video_demo.parser().parse_args(["x.mp4", "--identity"]).identity)


@unittest.skipUnless(os.path.exists(os.environ.get("FAH_TEST_VIDEO", "")) and os.path.isdir(os.path.join(api.DEFAULT_WEIGHTS, "fah")) and
                     os.path.isdir(api.DEFAULT_SAM3D), "needs FAH_TEST_VIDEO and the weights")
class ClipTests(unittest.TestCase):
    def run_clip(self, **kw):
        import cv2

        m = api.FAH("fah", "video", trt_dir=os.environ.get("FAH_TEST_TRT", ""), det_masks=True, **kw)
        m.reset()
        cap, out = cv2.VideoCapture(os.environ["FAH_TEST_VIDEO"]), []
        for _ in range(16):
            ok, img = cap.read()
            if not ok:
                break
            out.append(m(img, None))
        return out

    def test_defaults_equal_explicit_off_and_on_runs(self):
        a = self.run_clip()
        b = self.run_clip(side_track=False, edge_score=0.0, edge_track=False)
        self.assertEqual([len(f) for f in a], [len(f) for f in b])
        for fa, fb in zip(a, b):
            for x, y in zip(fa, fb):
                self.assertEqual(x["is_right"], y["is_right"])
                for k in ("bbox", "score", "joints3d"):
                    np.testing.assert_array_equal(np.asarray(x[k]), np.asarray(y[k]))
        c = self.run_clip(**api.IDENTITY)
        self.assertEqual(len(c), len(a))
        self.assertTrue(all(np.isfinite(np.asarray(h["joints3d"])).all() for f in c for h in f))


if __name__ == "__main__":
    unittest.main()
