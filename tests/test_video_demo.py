"""fah-video (fah.video_demo): the render of a small synthetic results file (MANO + ffmpeg), and with weights the whole
command on the first frames of a public HOT3D clip (FAH_TEST_VIDEO: an .mp4 of HOT3D test frames)."""
import os
import pickle
import shutil
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

from fah import video_demo
from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS

MANO_DIR = os.environ.get("MANO_DIR", "data/mano")
HAVE_MANO = os.path.exists(os.path.join(MANO_DIR, "MANO_RIGHT.pkl"))


def frame_count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames,width,height",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip()
    w, h, n = (int(x) for x in out.split(","))
    return n, (w, h)


@unittest.skipUnless(HAVE_MANO and shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs MANO_DIR and ffmpeg")
class RenderTests(unittest.TestCase):
    def test_grid_and_timeline_from_results(self):
        from bench.common import ManoLayer
        from bench.seg.rle import encode

        mano = ManoLayer(MANO_DIR)
        W, H, n = 320, 240, 4
        K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], np.float32)
        frames = []
        for i in range(n):
            j, v = mano(True, np.zeros(10), np.zeros(45), np.zeros(3), np.array([0.0, 0.0, 0.5]))
            uv = j[:, :2] / j[:, 2:] * 300 + [160, 120]
            m = np.zeros((H, W), bool)
            m[90:160, 120:200] = True
            lab = np.zeros((40, 40), np.uint8)
            lab[10:30, 10:30] = 1 + i % 3
            hand = dict(is_right=True, score=0.9, bbox=np.array([120, 90, 200, 160.0]), joints2d=uv, joints3d=j, verts3d=v,
                        mask_det=encode(m), parts_roi=(130, 100, cv2.imencode(".png", lab)[1].tobytes()), contact=np.full(778, 0.2 * i, np.float16),
                        contact_score=0.2 * i, contact_video=np.full(778, 0.25 * i, np.float32), contact_score_video=0.25 * i,
                        contact_flag_video=i >= 2, track_id=0, filled=False)
            frames.append(dict(index=i, pred=[hand, dict(hand, score=0.1)], latency=0.01))  # the second hand is below the display rule
        with tempfile.TemporaryDirectory() as tmp:
            vid = os.path.join(tmp, "in.mp4")
            wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
            for i in range(n):
                wr.write(np.full((H, W, 3), 60 + 20 * i, np.uint8))
            wr.release()
            res = dict(video="in.mp4", fps=30.0, size=(W, H), K=K, model="fah", frames=frames)
            pkl, out = os.path.join(tmp, "r.pkl"), os.path.join(tmp, "out.mp4")
            pickle.dump(res, open(pkl, "wb"))
            video_demo.main([vid, "-o", out, "--load", pkl, "--tile", "160", "--jobs", "1", "--mano_dir", MANO_DIR])
            nf, (w, h) = frame_count(out)
            self.assertEqual(nf, n)
            self.assertEqual(w, 3 * 160)  # 3 tiles per row
            self.assertEqual(h, 2 * 120 + 300)  # 2 rows of tiles + the contact timeline
            self.assertTrue(os.path.exists(os.path.join(tmp, "out_poster.jpg")))
            video_demo.main([vid, "-o", out, "--load", pkl, "--tile", "160", "--jobs", "1", "--mano_dir", MANO_DIR, "--panels", "input,joints"])
            self.assertEqual(frame_count(out)[1], (2 * 160, 120))  # no contact tile: no timeline

    def test_unknown_panel(self):
        with self.assertRaises(SystemExit):
            video_demo.render(dict(frames=[], K=None), video_demo.parser().parse_args(["x.mp4", "--panels", "nope"]), "o.mp4")


class StaticCameraTests(unittest.TestCase):
    def test_static_camera(self):
        self.assertTrue(video_demo.static_camera(2.98, "offline"))  # the most-turning tripod clip, offline SLAM
        self.assertTrue(video_demo.static_camera(5.3, "realtime"))  # ... and its real-time camera
        self.assertFalse(video_demo.static_camera(6.7, "offline"))  # the least-turning dev clip with camera motion (HOT3D train)
        self.assertFalse(video_demo.static_camera(6.7, "realtime"))
        self.assertTrue(video_demo.static_camera(1.0, "offline", 0.13))  # a tripod clip's drift
        self.assertFalse(video_demo.static_camera(1.0, "offline", 0.6))  # a sliding camera that does not turn

    def test_rotation_extent(self):
        a = np.radians(np.array([0.0, 3.0, 8.0, 5.0]))
        T = np.tile(np.eye(4), (4, 1, 1))
        T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
        T[2, 0, 3] = np.nan  # a lost frame's translation does not matter
        self.assertAlmostEqual(video_demo.rotation_extent(T), 8.0, places=6)
        T[:, 1, 3] = [0.0, 0.1, 0.2, -0.3]
        self.assertAlmostEqual(video_demo.translation_extent(T), 0.3, places=6)  # the frame without a centre is skipped


@unittest.skipUnless(os.path.exists(os.environ.get("FAH_TEST_VIDEO", "")) and os.path.isdir(os.path.join(DEFAULT_WEIGHTS, "fah")) and
                     os.path.isdir(DEFAULT_SAM3D) and HAVE_MANO and shutil.which("ffmpeg"), "needs FAH_TEST_VIDEO, the weights, MANO, ffmpeg")
class HOT3DClipTests(unittest.TestCase):
    def test_first_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clip.mp4")
            video_demo.main([os.environ["FAH_TEST_VIDEO"], "-o", out, "--max_frames", "8", "--jobs", "1", "--mano_dir", MANO_DIR,
                             *(["--trt_dir", os.environ["FAH_TEST_TRT"]] if os.environ.get("FAH_TEST_TRT") else [])])
            res = pickle.load(open(os.path.join(tmp, "clip.pkl"), "rb"))
            self.assertEqual(len(res["frames"]), 8)
            self.assertEqual(frame_count(out)[0], 8)
            hands = [h for f in res["frames"] for h in f["pred"] if not h["filled"]]
            self.assertTrue(hands)
            for h in hands:
                for k in ("joints2d", "joints3d", "verts3d", "mask_det", "parts_roi", "contact", "track_id"):
                    self.assertIn(k, h)
                self.assertEqual(np.asarray(h["verts3d"]).shape, (778, 3))
                self.assertTrue(np.isfinite(np.asarray(h["joints3d"])).all())
            self.assertTrue(any("contact_video" in h and "contact_flag_video" in h for h in hands))  # the track hand of each side


if __name__ == "__main__":
    unittest.main()
