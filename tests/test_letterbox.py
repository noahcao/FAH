"""fah.detector's letterbox equals ultralytics' auto letterbox (the input the YOLO weights were trained on) on frame sizes
of every aspect (needs ultralytics and cv2)."""
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("ultralytics") and importlib.util.find_spec("cv2"), "needs ultralytics")
class LetterboxTests(unittest.TestCase):
    def test_matches_ultralytics(self):
        from ultralytics.data.augment import LetterBox

        from fah.detector import WiLoRHandDetector

        det = WiLoRHandDetector.__new__(WiLoRHandDetector)
        det.imgsz, det.stride = 640, 32
        rng = np.random.default_rng(0)
        sizes = [(1080, 1920), (1080, 1620), (1080, 1440), (1080, 1080), (612, 1134), (1920, 1080), (480, 640), (1456, 1920),
                 (1408, 1408), (721, 1279)] + [tuple(int(v) for v in rng.integers(64, 2200, 2)) for _ in range(60)]
        for h, w in sizes:
            img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            ref = LetterBox(new_shape=(640, 640), auto=True, stride=32)(image=img)
            got, r, left, top = det._letterbox(img)
            self.assertEqual(got.shape, ref.shape, (h, w))
            self.assertTrue(np.array_equal(got, ref), (h, w))


if __name__ == "__main__":
    unittest.main()
