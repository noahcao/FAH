"""Tiny CPU regression tests: invalid 2D poses in matching, ignored-hand exemption order,
training step guards, ROI mask sampling without float64 device tensors, and the world timing flag check."""
import ast
import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from bench.bench2d.evaluate import evaluate_dir, match
from bench.hardpose.evaluate import main as rescore
from fah.detector import WiLoRHandDetector

REPO = Path(__file__).resolve().parents[1]
JOINTS = np.column_stack((np.linspace(10, 70, 21), np.linspace(15, 75, 21)))
GT = dict(is_right=True, joints2d=JOINTS, vis=np.ones(21), bbox=np.array([0, 0, 100, 100]))
GOOD = dict(is_right=True, joints2d=JOINTS, bbox=GT["bbox"])
INVALID = dict(GOOD, joints2d=np.full((21, 2), np.nan))


def write_run(d, preds):
    frame = dict(key="fixture/0", pred=preds, gt=[GT], fully_annotated=True, latency=0.001)
    Path(d, "clip.pkl").write_bytes(pickle.dumps(dict(method="fixture", frames=[frame])))


class InvalidPoseMatchTests(unittest.TestCase):
    def test_non_finite_pose_never_matches_or_steals_a_valid_match(self):
        self.assertEqual(match([INVALID], [GT])[0], {})
        self.assertEqual(match([INVALID, GOOD], [GT])[0], {0: 1})

    def test_non_finite_pose_counts_against_precision(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d, [INVALID])
            m = evaluate_dir(d)["all"]
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["precision"], 0.0)


class FitProjectGuardTests(unittest.TestCase):
    """joints3d_fit: the 3D joints projected at the translation that best fits the GT 2D; unusable 3D stays unmatched."""
    K = np.array([[500.0, 0, 50], [0, 500.0, 50], [0, 0, 1]])

    def run_fit(self, joints3d):
        X = np.column_stack((JOINTS / 1000.0 - 0.04, np.full(21, 0.5)))  # 3D whose projection is the GT 2D
        gt = dict(GT, joints2d=(X @ self.K.T)[:, :2] / X[:, 2:])
        pred = dict(GOOD, joints2d=gt["joints2d"], joints3d=X if joints3d is None else joints3d)
        frame = dict(key="fixture/0", pred=[pred], gt=[gt], fully_annotated=True, latency=0.001, K=self.K, image_size=(100, 100))
        with tempfile.TemporaryDirectory() as d:
            Path(d, "clip.pkl").write_bytes(pickle.dumps(dict(method="fixture", frames=[frame])))
            return evaluate_dir(d, "joints3d_fit")["all"]

    def test_valid_3d_matches(self):
        m = self.run_fit(None)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["pck10"], 1.0)

    def test_non_finite_or_misshaped_3d_is_unmatched_and_counts_against_precision(self):
        for j3 in (np.full((21, 3), np.nan), np.zeros((20, 3))):
            with self.subTest(shape=j3.shape, finite=bool(np.isfinite(j3).all())):
                m = self.run_fit(j3)
                self.assertEqual(m["recall"], 0.0)
                self.assertEqual(m["precision"], 0.0)


class IgnoredHandOrderTests(unittest.TestCase):
    def rescored(self, preds, key=""):
        with tempfile.TemporaryDirectory() as d:
            pred_dir = Path(d, "preds")
            pred_dir.mkdir()
            write_run(pred_dir, preds)
            ann = dict(meta={"version": "fixture"}, images=[dict(id="fixture/0", hands=[
                dict(is_right=True, kp2d=JOINTS.tolist(), vis=[1] * 21, bbox=GT["bbox"].tolist()),
                dict(is_right=False, kp2d=JOINTS.tolist(), vis=[1] * 21, bbox=[20, 0, 120, 100], status="ignore")])])
            Path(d, "ann.json").write_text(json.dumps(ann))
            out = Path(d, "out.json")
            with contextlib.redirect_stdout(io.StringIO()):
                rescore(SimpleNamespace(ann=str(Path(d, "ann.json")), runs=str(pred_dir) + key, out=str(out)))
            return next(iter(json.loads(out.read_text())["runs"].values()))["all"]

    def test_prediction_overlapping_an_ignored_hand_still_matches_its_scored_hand(self):
        m = self.rescored([GOOD])
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["pck10_all"], 1.0)

    def test_matching_uses_the_selected_keypoint_field(self):
        pred = dict(GOOD, joints2d=JOINTS + 500.0, joints2d_alt=JOINTS)  # only the selected field lies on the hand
        m = self.rescored([pred], key=":joints2d_alt")
        self.assertEqual(m["recall"], 1.0)

    def test_unmatched_prediction_inside_an_ignored_box_is_exempt(self):
        stray = dict(is_right=False, joints2d=JOINTS + 30.0, bbox=np.array([20, 0, 120, 100]))
        m = self.rescored([GOOD, stray])
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["precision"], 1.0)


def finetune_ast():
    return ast.parse((REPO / "train" / "finetune.py").read_text())


def find_if(tree, text):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.If) and text in ast.unparse(n.test))


class TrainingStepGuardTests(unittest.TestCase):
    def test_non_finite_gradient_norm_is_skipped_regardless_of_the_threshold(self):
        test = ast.Expression(find_if(finetune_ast(), "isfinite(gn)").test)
        guard = lambda gn, skip_gn: eval(compile(test, "guard", "eval"), dict(torch=torch, gn=gn, args=SimpleNamespace(skip_gn=skip_gn)))
        p = torch.nn.Parameter(torch.tensor(0.0))
        loss = torch.sqrt(p.square())  # finite loss, NaN derivative
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p], 1.0)
        self.assertTrue(torch.isfinite(loss) and not torch.isfinite(gn))
        self.assertTrue(guard(gn, 1000.0))
        self.assertTrue(guard(gn, 0))
        self.assertTrue(guard(torch.tensor(2000.0), 1000.0))
        self.assertFalse(guard(torch.tensor(5.0), 1000.0))

    def test_non_finite_loss_skip_is_one_decision_across_ranks(self):
        tree = finetune_ast()
        body = next(b for b in (getattr(n, "body", None) for n in ast.walk(tree)) if isinstance(b, list)
                    and any(isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "finite" for s in b))
        i = next(k for k, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "finite")
        loop = ast.For(target=ast.Name("_", ast.Store()), iter=ast.parse("[0]", mode="eval").body,
                       body=body[i:i + 3] + [ast.parse("ran = True").body[0]], orelse=[])
        code = compile(ast.fix_missing_locations(ast.Module([loop], [])), "step", "exec")

        def run(loss, ddp, others_finite=True):
            def all_reduce(t, op=None):  # another rank's flag enters the MIN
                t.copy_(torch.minimum(t, torch.tensor(float(others_finite))))
            dist = SimpleNamespace(all_reduce=Mock(side_effect=all_reduce), ReduceOp=SimpleNamespace(MIN="min"))
            ns = dict(torch=torch, loss=loss, ddp=ddp, dist=dist, rank=1, step=0, opt=Mock())
            exec(code, ns)
            return ns.get("ran", False), dist.all_reduce

        ran, ar = run(torch.tensor(1.0), ddp=True)
        self.assertTrue(ran)
        self.assertEqual(ar.call_args.kwargs["op"], "min")
        self.assertFalse(run(torch.tensor(1.0), ddp=True, others_finite=False)[0])  # finite here, NaN on another rank
        self.assertFalse(run(torch.tensor(float("nan")), ddp=False)[0])


class DtypeRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.dtypes = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else [out]:
            if isinstance(t, torch.Tensor):
                self.dtypes.add(t.dtype)
        return out


class MaskRoiTests(unittest.TestCase):
    def detector(self, device="cpu"):
        rng = np.random.default_rng(5)
        d = WiLoRHandDetector.__new__(WiLoRHandDetector)
        d.last = dict(hw=(24, 32), boxes=np.array([[1.2, 2.5, 25.8, 22.4], [12.1, 0, 32, 19.7]], np.float32),
                      proto=torch.tensor(rng.normal(size=(3, 8, 10)), dtype=torch.float32).to(device),
                      mc=torch.tensor([[1.0, -0.4, 0.3], [0.2, 1.0, -0.5]]).to(device), scale=0.25, r=1.0, pad=(4, 4))
        return d

    def test_roi_masks_equal_point_sampling_including_padding_and_order(self):
        d = self.detector()
        for keep in ([0], [0, 1], [1, 0]):
            rois = [(-2, -2, 36, 28)] * len(keep)
            pts = [(np.tile(np.arange(x, x + w), h), np.repeat(np.arange(y, y + h), w)) for x, y, w, h in rois]
            for a, b in zip(d.mask_roi(keep, rois), d.mask_at(keep, pts)):
                np.testing.assert_array_equal(a.ravel(), b)

    def test_no_float64_tensors_are_created(self):  # MPS has no float64
        d, rec = self.detector(), DtypeRecorder()
        with rec:
            d.mask_roi([0, 1], [(0, 0, 32, 24)] * 2)
        self.assertNotIn(torch.float64, rec.dtypes)

    @unittest.skipUnless(torch.backends.mps.is_available(), "needs Apple MPS")
    def test_runs_on_mps_with_cpu_parity(self):
        rois = [(0, 0, 32, 24)] * 2
        for a, b in zip(self.detector("mps").mask_roi([0, 1], rois), self.detector().mask_roi([0, 1], rois)):
            np.testing.assert_array_equal(a, b)


class WorldTimingFlagTests(unittest.TestCase):
    def test_moge2_without_background_depth_is_rejected(self):
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "benchmark_world_e2e.py"), "--depth_model", "moge2"],
                           capture_output=True, text=True, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        self.assertEqual(r.returncode, 2)
        self.assertIn("needs --bg_depth", r.stderr)


if __name__ == "__main__":
    unittest.main()
