"""fah.yolo: the detector network without the ultralytics package. With the release weights: the default inference path
runs with ultralytics blocked from import, and (where ultralytics is installed) the network equals ultralytics' fused
model exactly (decoded outputs, prototypes and the raw head outputs the Core ML export runs)."""
import importlib.util
import os
import subprocess
import sys
import unittest

import numpy as np
import torch

from fah import api

DET = api.detector_weights()
HAVE = os.path.exists(DET) and os.path.isdir(api.DEFAULT_SAM3D)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AnchorTests(unittest.TestCase):
    def test_anchor_grid_and_box_decode(self):
        from fah.yolo import dist2box, make_anchors

        feats = [torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 1, 2)]
        a, s = make_anchors(feats, torch.tensor([8.0, 16.0]))
        self.assertEqual(a.tolist(), [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [0.5, 1.5], [1.5, 1.5], [2.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
        self.assertEqual(s[:, 0].tolist(), [8.0] * 6 + [16.0] * 2)
        box = dist2box(torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]), torch.tensor([[[10.0], [20.0]]]))
        self.assertEqual(box[0, :, 0].tolist(), [11.0, 21.0, 4.0, 6.0])  # cx, cy, w, h


def tiny_net():
    """A small segmentation network from fah.yolo's own blocks (no checkpoint)."""
    import torch.nn as nn

    from fah.yolo import ConvAct, Join, MaskBasis, SplitChain, YOLOHead, YOLONet, Residual

    torch.manual_seed(0)
    conv = lambda a, b, s=1: ConvAct(nn.Conv2d(a, b, 3, s, 1), nn.SiLU())
    stack = lambda c, out: nn.Sequential(conv(c, 8), conv(8, 8), nn.Conv2d(8, out, 1))
    layers = [conv(3, 8, 2), conv(8, 8, 2), SplitChain(conv(8, 8), conv(12, 8), nn.ModuleList([Residual(conv(4, 4), conv(4, 4), True)])),
              conv(8, 8), Join(1), conv(16, 8, 2)]
    integral = nn.Conv2d(4, 1, 1, bias=False).requires_grad_(False)
    integral.weight.data[:] = torch.arange(4.0).view(1, 4, 1, 1)
    head = YOLOHead(nn.ModuleList([stack(8, 16), stack(8, 16)]), nn.ModuleList([stack(8, 2), stack(8, 2)]),
                    nn.ModuleList([stack(8, 3), stack(8, 3)]), integral,
                    MaskBasis(conv(8, 8), nn.ConvTranspose2d(8, 8, 2, 2), conv(8, 8), conv(8, 3)), 2, 4, 3, torch.tensor([8.0, 16.0]))
    layers.append(head)
    for i, (m, f) in enumerate(zip(layers, [-1, -1, -1, -1, [-1, 2], -1, [2, 5]])):
        m.f, m.i = f, i
    return YOLONet(layers, [2, 5], {0: "left", 1: "right"}).eval()


class ExportTests(unittest.TestCase):
    def test_raw_outputs_survive_non_strict_torch_export(self):
        net = tiny_net()

        class Raw(torch.nn.Module):  # what the Core ML export traces (scripts/build_coreml.py det)
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, x):
                d = self.net(x, raw=True)
                return d["boxes"], d["scores"], d["mask_coefficient"], d["proto"]

        x = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            ref = Raw()(x)
            got = torch.export.export(Raw(), (x,), strict=False).module()(x)
            out, proto = net(x)
        for a, b in zip(ref, got):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(out.shape, (1, 4 + 2 + 3, 16 * 24 + 8 * 12))
        self.assertEqual(proto.shape, (1, 3, 32, 48))


@unittest.skipUnless(HAVE, "needs the weights")
class NoUltralyticsTests(unittest.TestCase):
    def test_default_inference_runs_with_ultralytics_blocked(self):
        code = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("ultralytics blocked")
sys.meta_path.insert(0, Block())
import numpy as np, cv2
import fah
m = fah.FAH("fah-s", "image", device="cpu", mesh="none")
img = np.full((480, 640, 3), 128, np.uint8)
m(img)
print("ok", "ultralytics" in sys.modules)
"""
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES=""))
        self.assertIn("ok False", r.stdout, r.stderr[-2000:])


@unittest.skipUnless(HAVE and importlib.util.find_spec("ultralytics"), "needs the weights and ultralytics")
class UltralyticsParityTests(unittest.TestCase):
    def test_network_equals_ultralytics(self):
        from ultralytics import YOLO

        from fah.yolo import load_yolo

        ref = YOLO(DET).model.fuse(verbose=False).float().eval()
        ours = load_yolo(DET)
        x = torch.from_numpy(np.random.default_rng(0).random((1, 3, 384, 640), np.float32))
        with torch.no_grad():
            (a, ap), raw_a = ref(x)
            b, bp = ours(x)
            raw_b = ours(x, raw=True)
        self.assertTrue(torch.equal(a, b) and torch.equal(ap, bp))
        for k in ("boxes", "scores", "mask_coefficient", "proto"):
            self.assertTrue(torch.equal(raw_a[k], raw_b[k]), k)
        self.assertEqual(ours.names, dict(ref.names))
        self.assertEqual(ours.stride.tolist(), ref.stride.tolist())


if __name__ == "__main__":
    unittest.main()
