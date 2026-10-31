"""CPU-only crop evaluation checks; synthetic metadata and images, no weights."""
import contextlib
import io
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts.crop_eval3d import main, sample
from train.build_crops import crop_keypoints, render_crop


def eager_sample(records, n, seed):
    """Reference sampling used for the existing benchmark results."""
    by = defaultdict(list)
    for r in records:
        if r.get("joints3d") is not None:
            by[r.get("episode", r["src"].split(":")[0])].append(r)
    rng = np.random.default_rng(seed)
    per = max(1, n // max(len(by), 1))
    out = []
    for e in sorted(by):
        rs = by[e]
        out += [rs[i] for i in sorted(rng.choice(len(rs), min(per, len(rs)), replace=False))]
    rng.shuffle(out)
    return out[:n]


class CropSamplingTests(unittest.TestCase):
    def test_sample_identity_and_order_match_existing_benchmarks(self):
        # Unequal, interleaved episodes, including src fallback and unlabelled rows.
        records = []
        for i, episode in enumerate(["z", "a", "b", "z", "b", "b", "z", "z", "b", "z"]):
            r = dict(src=f"{episode}:{i}", file=f"{i}.jpg", joints3d=[[i, 0, 1]])
            if episode != "b":
                r["episode"] = episode
            records.extend([dict(src="unlabelled:0", joints3d=None), r])
        records.append(dict(src="unlabelled:1"))
        with tempfile.TemporaryDirectory(prefix="fah-crop-eval-") as tmp:
            index = Path(tmp) / "index.jsonl"
            index.write_text("".join(json.dumps(r) + "\n" for r in records))
            for n in (0, 1, 2, 4, 8, 10, 100):
                for seed in (0, 7, 123):
                    with self.subTest(n=n, seed=seed):
                        self.assertEqual(sample(tmp, n, seed), eager_sample(records, n, seed))

    def test_no_3d_labels_returns_empty_sample(self):
        with tempfile.TemporaryDirectory(prefix="fah-crop-eval-") as tmp:
            index = Path(tmp) / "index.jsonl"
            for records in ([], [dict(src="a:0", joints3d=None)]):
                index.write_text("".join(json.dumps(r) + "\n" for r in records))
                self.assertEqual(sample(tmp, 4000), [])


class CropGeometryTests(unittest.TestCase):
    def test_original_left_and_right_hands_use_stored_crop_coordinates(self):
        width, height = 640, 480
        K = np.array([[500., 0, 280.], [0, 510., 220.], [0, 0, 1.]])
        rng = np.random.default_rng(12)
        joints = rng.uniform([.05, -.05, .45], [.13, .03, .6], (21, 3))
        projected = joints @ K.T
        pixels = projected[:, :2] / projected[:, 2:]
        keypoints = np.column_stack([pixels, np.ones(21)])
        box = np.r_[pixels.min(0) - 5, pixels.max(0) + 5]
        for is_right in (False, True):
            with self.subTest(is_right=is_right), \
                    tempfile.TemporaryDirectory(prefix="fah-crop-eval-") as tmp:
                crop, M = render_crop(np.zeros((height, width, 3), np.uint8), box, is_right, size=64)
                stored_joints, stored_K = joints.copy(), K.copy()
                if not is_right:
                    stored_joints[:, 0] *= -1
                    stored_K[0, 2] = width - 1 - stored_K[0, 2]
                record = dict(src="episode:0", file="crop.png", box=box.tolist(), is_right=is_right,
                              M=M.tolist(), K=stored_K.tolist(), joints3d=stored_joints.tolist(), hw=[height, width])
                Path(tmp, "index.jsonl").write_text(json.dumps(record) + "\n")
                self.assertTrue(cv2.imwrite(str(Path(tmp, "crop.png")), crop))
                est = Mock()
                est.process_image.return_value = [dict(keypoints_3d=stored_joints.copy())]
                method = SimpleNamespace(est=est, reset=Mock())
                args = SimpleNamespace(pool=tmp, n=1, seed=0, method_args="{}", out="")
                stdout = io.StringIO()
                with patch("bench.run.load_method", return_value=method), contextlib.redirect_stdout(stdout):
                    main(args)
                result = json.loads(stdout.getvalue())
                self.assertEqual((result["hands"], result["missed"]), (1, 0))
                self.assertAlmostEqual(result["pa_mpjpe"], 0)
                self.assertAlmostEqual(result["rr_mpjpe"], 0)
                method.reset.assert_called_once()
                est.process_image.assert_called_once()
                image_args, kwargs = est.process_image.call_args
                np.testing.assert_array_equal(image_args[0], crop)
                np.testing.assert_array_equal(kwargs["is_right"], [True])
                uv = stored_joints @ kwargs["cam_int"].T
                crop_pixels = uv[:, :2] / uv[:, 2:]
                expected = crop_keypoints(keypoints, M, is_right, width)[:, :2]
                np.testing.assert_allclose(crop_pixels, expected, atol=1e-4)
                supplied_box = kwargs["boxes"][0]
                self.assertTrue(np.all(crop_pixels >= supplied_box[:2]))
                self.assertTrue(np.all(crop_pixels <= supplied_box[2:]))
                self.assertTrue(np.all(supplied_box >= 0))
                self.assertTrue(np.all(supplied_box <= crop.shape[0]))


if __name__ == "__main__":
    unittest.main()
