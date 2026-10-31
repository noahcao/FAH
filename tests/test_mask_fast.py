"""Host (Core ML) detector masks: box_logits (two matmuls per box) = torchvision roi_align at the box's frame pixels,
masks() = masks_crop() pasted, and the crop RLE (bench.seg.rle encode_crop) = encode of the pasted mask."""
import unittest

import numpy as np
import torch
import torchvision

from bench.seg.rle import decode, encode, encode_crop, paste_crop
from fah.detector import box_logits

R, S, PX, PY = 640 / 1620, 0.25, 0.0, (448 - 1080 * 640 / 1620) / 2  # 1620 x 1080 frame, 448 x 640 letterbox, 1/4 protos


class MaskFastTests(unittest.TestCase):
    def test_box_logits_match_roi_align(self):
        rng = np.random.default_rng(0)
        P = rng.standard_normal((32, 112, 160)).astype(np.float32)
        mc = rng.standard_normal(32).astype(np.float32)
        comb = torch.einsum("c,chw->hw", torch.from_numpy(mc), torch.from_numpy(P))[None, None]
        for _ in range(100):
            x0, y0 = int(rng.integers(0, 1600)), int(rng.integers(0, 1060))
            x1, y1 = min(1620, x0 + int(rng.integers(1, 500))), min(1080, y0 + int(rng.integers(1, 500)))
            b = torch.tensor([[0, (x0 * R + PX) * S, (y0 * R + PY) * S, (x1 * R + PX) * S, (y1 * R + PY) * S]])
            ref = torchvision.ops.roi_align(comb, b, (y1 - y0, x1 - x0), 1.0, 1, aligned=True)[0, 0].numpy()
            got = box_logits(mc, P, x0, y0, x1, y1, R, S, PX, PY)
            self.assertEqual(got.shape, ref.shape)
            self.assertLess(np.abs(got - ref).max(), 2e-3)

    def test_crop_rle_matches_full(self):
        rng = np.random.default_rng(1)
        for _ in range(2000):
            H, W = (int(v) for v in rng.integers(1, 12, 2))
            y0, x0 = int(rng.integers(0, H)), int(rng.integers(0, W))
            c = rng.random((int(rng.integers(1, H - y0 + 1)), int(rng.integers(1, W - x0 + 1)))) < rng.choice([0.0, 0.3, 0.7, 1.0])
            full = paste_crop(c, x0, y0, (H, W))
            self.assertEqual(encode_crop(c, x0, y0, (H, W)), encode(full))
            self.assertTrue((decode(encode_crop(c, x0, y0, (H, W))) == full).all())

    def test_masks_is_pasted_crops(self):
        from fah.detector import WiLoRHandDetector

        rng = np.random.default_rng(2)
        det = WiLoRHandDetector.__new__(WiLoRHandDetector)
        det.last = dict(boxes=np.array([[300, 300, 630, 680], [500, 320, 820, 700], [10, 10, 10, 50]], np.float32), hw=(1080, 1620),
                        mc=torch.from_numpy(rng.standard_normal((3, 32)).astype(np.float32)),
                        proto=torch.from_numpy(rng.standard_normal((32, 112, 160)).astype(np.float32)), scale=S, r=R, pad=(PX, PY))
        full, crops = det.masks(), det.masks_crop()
        self.assertIsNone(crops[2])
        self.assertFalse(full[2].any())
        for k in (0, 1):
            x0, y0, c = crops[k]
            self.assertTrue((paste_crop(c, x0, y0, (1080, 1620)) == full[k]).all())
        self.assertFalse((full[0] & full[1]).any())  # the overlap rule: one owner per pixel


if __name__ == "__main__":
    unittest.main()
