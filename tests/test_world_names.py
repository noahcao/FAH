"""World camera stages keep clips whose names contain dots (e.g. video_12.5_20.0.mp4) and skip only the stages'
temporary outputs (<clip>.npz.<pid>.npz)."""
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from bench.world_camera import calibrate, gate, partial


class DottedNameTests(unittest.TestCase):
    def test_calibrate_and_gate_keep_dotted_clip_names(self):
        src, cal, out = tempfile.mkdtemp(), tempfile.mkdtemp(), tempfile.mkdtemp()
        T = np.tile(np.eye(4), (6, 1, 1))
        T[:, 0, 3] = np.arange(6)
        rec = dict(T_world_cam_d0=T, n_fed=3, sig_res=np.ones((3, 2)), scale_causal=2.0)
        for name in ("-0Rhey_474.8_487.3.npz", "plain.npz", "-0Rhey_474.8_487.3.npz.123.npz"):
            np.savez(os.path.join(src, name), **rec)
        calibrate(SimpleNamespace(src=src, dst=cal, cal=0.5))
        self.assertEqual(sorted(os.listdir(cal)), ["-0Rhey_474.8_487.3.npz", "plain.npz"])
        self.assertEqual(float(np.load(os.path.join(cal, "-0Rhey_474.8_487.3.npz"))["scale_cal"]), 1.0)
        gate(SimpleNamespace(src=cal, dst=out, g=0.5))
        self.assertEqual(sorted(os.listdir(out)), ["-0Rhey_474.8_487.3.npz", "plain.npz"])

    def test_partial(self):
        self.assertTrue(partial("clip.npz.4242.npz"))
        self.assertFalse(partial("clip_12.5_20.0.npz"))


if __name__ == "__main__":
    unittest.main()
