"""World timing harness (scripts/benchmark_world_e2e.py): the streaming frame source gives the frame cache's arrays, in
order, with the ring filled before iteration starts and look-ups of recent / other frames."""
import os
import tempfile
import unittest

import cv2
import numpy as np

from scripts.benchmark_world_e2e import FrameSource


class FrameSourceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.src, self.work = tempfile.mkdtemp(), tempfile.mkdtemp()
        self.files = []
        for i in range(7):
            f = os.path.join(self.src, f"{i:05d}.jpg")
            cv2.imwrite(f, rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
            self.files.append(f)
        self.cache = np.stack([cv2.imread(f) for f in self.files])  # the --frame_cache file of a --frames_dir clip

    def test_jpegs_and_raw_file_give_the_cache_arrays_in_order(self):
        open(os.path.join(self.work, "frames.txt"), "w").write("\n".join(self.files))
        s = FrameSource(self.work, ahead=3, keep=2)
        self.assertEqual(len(s), 7)
        for _ in range(2):  # re-iterable (the camera's warm-up pass, then the timed one)
            got = list(s)
            self.assertEqual(len(got), 7)
            for a, b in zip(got, self.cache):
                np.testing.assert_array_equal(a, b)
        raw = tempfile.mkdtemp()
        np.save(os.path.join(raw, "frames.npy"), self.cache)
        r = FrameSource(raw, ahead=4)
        got = list(r)
        self.assertTrue(all(isinstance(f, np.ndarray) and type(f) is np.ndarray and f.flags.writeable for f in got))
        np.testing.assert_array_equal(np.stack(got), self.cache)

    def test_ring_is_full_before_the_first_frame_and_lookups(self):
        open(os.path.join(self.work, "frames.txt"), "w").write("\n".join(self.files))
        s = FrameSource(self.work, ahead=3, keep=2)
        calls = []
        load = s.load
        s.load = lambda i: calls.append(i) or load(i)
        it = iter(s)
        self.assertGreaterEqual(len(calls), 3)  # decoded ahead before the start
        for _ in range(4):
            next(it)
        self.assertEqual(list(s.recent), [2, 3])  # the last `keep` frames
        n = len(calls)
        np.testing.assert_array_equal(s[3], self.cache[3])
        np.testing.assert_array_equal(s[-1], self.cache[6])  # not recent: decoded on demand
        self.assertEqual(calls[n:].count(3), 0)
        self.assertEqual(len(s[:3]), 3)
        np.testing.assert_array_equal(np.stack(s[1:5:2]), self.cache[1:5:2])


if __name__ == "__main__":
    unittest.main()
