"""Tiny coverage fixtures for 2D matching, ignored hands, and projected 3D."""
import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench.bench2d.evaluate import evaluate_dir, match
from bench.hardpose.evaluate import main as rescore


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fah-eval-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "preds"
        self.run_dir.mkdir()
        self.K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1.]])
        xyz = np.random.default_rng(3).normal(0, .03, (21, 3)) + [0, 0, .5]
        xy = (xyz @ self.K.T)[:, :2] / xyz[:, 2:]
        self.gt = dict(is_right=True, joints2d=xy, vis=np.ones(21),
                       bbox=np.array([0., 0., 100., 100.]))
        self.good = dict(is_right=True, joints2d=xy, joints3d=xyz, bbox=self.gt["bbox"])

    def save(self, predictions):
        frames = [dict(key=f"fixture/{i}", latency=0., fully_annotated=True,
                       K=self.K, gt=[self.gt], pred=pred) for i, pred in enumerate(predictions)]
        (self.run_dir / "clip.pkl").write_bytes(pickle.dumps(dict(method="fixture", frames=frames)))

    def test_invalid_prediction_cannot_take_a_valid_match(self):
        for value in (np.nan, np.inf):
            bad = dict(self.good, joints2d=np.full((21, 2), value))
            mapping, fp = match([bad, self.good], [self.gt])
            self.assertEqual(mapping, {0: 1})
            self.assertEqual(fp, 1)
        mapping, fp = match([self.good], [dict(self.gt, vis=np.zeros(21))])
        self.assertEqual(mapping, {})
        self.assertEqual(fp, 1)

    def test_invalid_pose_remains_a_miss_and_false_positive(self):
        self.save([[dict(self.good, joints2d=np.full((21, 2), np.nan))]])
        metrics = evaluate_dir(str(self.run_dir))["all"]
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["pck10_all"], 0)

    def test_unusable_3d_is_charged_without_losing_valid_frames(self):
        for xyz in (None, np.full((21, 3), np.nan), np.full((21, 3), np.inf), np.zeros((20, 3))):
            self.save([[self.good], [dict(self.good, joints3d=xyz)]])
            metrics = evaluate_dir(str(self.run_dir), "joints3d_fit")["all"]
            self.assertEqual(metrics["recall"], .5)
            self.assertEqual(metrics["precision"], .5)
            self.assertEqual(metrics["pck10"], 1)
            self.assertEqual(metrics["pck10_all"], .5)

    def test_ignored_hand_does_not_remove_scored_match_for_an_alternate_key(self):
        prediction = dict(self.good, joints2d=self.good["joints2d"] + 1000, alternate=self.good["joints2d"])
        ignored_prediction = dict(prediction, alternate=self.good["joints2d"] + 1000)
        self.save([[prediction, ignored_prediction]])
        h = dict(is_right=True, kp2d=self.gt["joints2d"].tolist(), vis=[1] * 21,
                 bbox=self.gt["bbox"].tolist())
        annotation = dict(meta={"version": "fixture"}, images=[dict(
            id="fixture/0", hands=[h, dict(h, is_right=False, bbox=[20, 0, 120, 100], status="ignore")])])
        ann, out = self.root / "ann.json", self.root / "metrics.json"
        ann.write_text(json.dumps(annotation))
        with contextlib.redirect_stdout(io.StringIO()):
            rescore(SimpleNamespace(ann=str(ann), runs=f"{self.run_dir}:alternate", out=str(out)))
        metrics = json.loads(out.read_text())["runs"]["preds:alternate"]["all"]
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["pck10_all"], 1)


if __name__ == "__main__":
    unittest.main()
