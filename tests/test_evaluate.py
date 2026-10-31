"""Coverage accounting with tiny synthetic predictions, without MANO assets."""
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.evaluate import evaluate_dir, miss_charge


class CoverageTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        joints = rng.normal(0, .025, (21, 3)) + [0, 0, .5]
        verts = rng.normal(0, .025, (24, 3)) + [0, 0, .5]
        self.gt = dict(is_right=True, bbox=np.array([0., 0., 30., 30.]), joints3d=joints, verts3d=verts)
        self.canon = {True: (joints * 2, verts * 2)}
        self.charge = miss_charge(self.canon, self.gt)

    def evaluate(self, predictions, with_canon=True):
        frames = [dict(index=i, latency=0., gt=[self.gt], pred=pred) for i, pred in enumerate(predictions)]
        with tempfile.TemporaryDirectory(prefix="fah-test-") as tmp:
            with open(Path(tmp) / "clip.pkl", "wb") as f:
                pickle.dump(dict(method="synthetic", dataset="image", frames=frames), f)
            # No MANO files needed; the optional vertex joint metric is irrelevant here.
            with patch("bench.evaluate.canonical_hands", return_value=self.canon), \
                    patch("bench.evaluate.VertexJoints", return_value=lambda v: v[:21]):
                return evaluate_dir(tmp, mano_dir="unused" if with_canon else None)

    def test_invalid_mesh_keeps_joints_and_charges_mesh(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                bad = dict(self.gt, verts3d=np.full_like(self.gt["verts3d"], value))
                metrics = self.evaluate([[bad]])
                self.assertEqual(metrics["num_nonfinite_pred"], 1)
                self.assertEqual(metrics["num_matched"], 1)
                self.assertAlmostEqual(metrics["pa_mpjpe"], 0)
                self.assertAlmostEqual(metrics["pa_mpjpe_p"], 0)
                self.assertIsNone(metrics["pa_mpvpe"])
                self.assertIsNone(metrics["pa_mpjpe_v"])
                self.assertAlmostEqual(metrics["pa_mpvpe_p"], self.charge["v"])

    def test_invalid_or_missing_joints_charge_pose_once(self):
        for joints in (None, np.full((21, 3), np.nan), np.full((21, 3), np.inf)):
            with self.subTest(joints=joints is None):
                metrics = self.evaluate([[dict(self.gt, joints3d=joints)]])
                self.assertEqual(metrics["num_matched"], 1)
                self.assertEqual(metrics["num_nonfinite_pred"], int(joints is not None))
                self.assertIsNone(metrics["pa_mpjpe"])
                self.assertAlmostEqual(metrics["pa_mpjpe_p"], self.charge["j"])
                self.assertAlmostEqual(metrics["pa_mpvpe_p"], self.charge["v"])

    def test_mixed_valid_invalid_and_missed_predictions_keep_denominator(self):
        bad_mesh = dict(self.gt, verts3d=np.full_like(self.gt["verts3d"], np.nan))
        bad_joints = dict(self.gt, joints3d=np.full_like(self.gt["joints3d"], np.nan))
        metrics = self.evaluate([[self.gt], [bad_mesh], [bad_joints], []])
        self.assertEqual(metrics["num_gt_hands"], 4)
        self.assertEqual(metrics["num_matched"], 3)
        self.assertEqual(metrics["num_nonfinite_pred"], 2)
        self.assertAlmostEqual(metrics["pa_mpjpe"], 0)
        self.assertAlmostEqual(metrics["pa_mpvpe"], 0)
        self.assertAlmostEqual(metrics["pa_mpjpe_p"], self.charge["j"] * 2 / 4)
        self.assertAlmostEqual(metrics["pa_mpvpe_p"], self.charge["v"] * 3 / 4)

    def test_all_missed_hands_still_have_coverage_metrics(self):
        metrics = self.evaluate([[], []])
        self.assertIsNone(metrics["pa_mpjpe"])
        self.assertIsNone(metrics["pa_mpvpe"])
        self.assertAlmostEqual(metrics["pa_mpjpe_p"], self.charge["j"])
        self.assertAlmostEqual(metrics["pa_mpvpe_p"], self.charge["v"])

    def test_no_mesh_is_charged_even_without_any_valid_meshes(self):
        metrics = self.evaluate([[dict(self.gt, verts3d=None)]])
        self.assertEqual(metrics["num_nonfinite_pred"], 0)
        self.assertAlmostEqual(metrics["pa_mpjpe_p"], 0)
        self.assertAlmostEqual(metrics["pa_mpvpe_p"], self.charge["v"])

    def test_invalid_mesh_without_canonical_assets_keeps_joint_scores(self):
        bad = dict(self.gt, verts3d=np.full_like(self.gt["verts3d"], np.nan))
        metrics = self.evaluate([[bad]], with_canon=False)
        self.assertAlmostEqual(metrics["pa_mpjpe"], 0)
        self.assertIsNone(metrics["pa_mpvpe_p"])

    def test_valid_predictions_are_unchanged(self):
        metrics = self.evaluate([[self.gt]])
        self.assertEqual(metrics["num_nonfinite_pred"], 0)
        self.assertEqual(metrics["det_recall"], 1)
        for key in ("pa_mpjpe", "rr_mpjpe", "pa_mpvpe", "pa_mpjpe_p", "rr_mpjpe_p", "pa_mpvpe_p"):
            with self.subTest(metric=key):
                self.assertAlmostEqual(metrics[key], 0)


if __name__ == "__main__":
    unittest.main()
