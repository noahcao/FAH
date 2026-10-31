"""CPU-only checks of the baseline ports, no weights: the SAM 3D Body hand adapter's fixed settings, the segmentation
comparison's prompt / prediction plumbing and timing aggregation, and the detector-timing helpers."""
import unittest
from unittest.mock import patch

import numpy as np

from bench.methods import sam3db
from bench.seg.baselines import BOX_ARMS, build_arms, mask_box
from scripts import benchmark_detector, seg_compare


class SAM3DBodyAdapterTests(unittest.TestCase):
    def capture(self, **kw):
        seen = {}

        def init(self_, **args):
            seen.update(args)

        with patch.object(sam3db.FAHMethod, "__init__", init):
            m = sam3db.SAM3DBodyHandMethod(**kw)
        return m, seen

    def test_paper_settings(self):
        m, seen = self.capture()
        self.assertTrue(seen["base_weights"])
        self.assertFalse(seen["parts"])
        self.assertEqual((seen["dtype"], seen["cuda_graph"], seen["detector"], seen["det_conf"]), ("bfloat16", True, "wilor", 0.3))
        self.assertEqual((seen["mesh_fit"], seen["tip_extend"], seen["bbox_padding"]), (0, 0.0, 0.9))
        self.assertNotIn("input_size", seen)  # the checkpoint config's native crop size
        self.assertEqual(m.name, "sam3db-hand")

    def test_eager_name_and_rejects_fah_weights(self):
        m, _ = self.capture(cuda_graph=False)
        self.assertEqual(m.name, "sam3db-hand-eager")
        for k in ("finetune_ckpt", "pose_trt", "contact_ckpt", "refiner"):
            with self.assertRaises(ValueError):
                self.capture(**{k: "x"})

    def test_registered(self):
        from bench.run import METHODS

        self.assertEqual(METHODS["sam3db_hand"], "bench.methods.sam3db:SAM3DBodyHandMethod")


class SegCompareTests(unittest.TestCase):
    def test_mask_box(self):
        m = np.zeros((10, 20), bool)
        m[2:5, 3:9] = True
        np.testing.assert_array_equal(mask_box(m), [3, 2, 9, 5])
        self.assertIsNone(mask_box(np.zeros((4, 4), bool)))

    def test_unknown_arm(self):
        with self.assertRaises(ValueError):
            build_arms(["sam4"])

    def test_box_arms_keep_prompt_order_and_fields(self):
        ours = [dict(bbox=np.array([0, 0, 4, 4.0]), score=0.9, is_right=True, joints2d=np.zeros((21, 2))),
                dict(bbox=np.array([10, 10, 14, 14.0]), score=0.8, is_right=False, joints2d=np.ones((21, 2)))]
        fr = dict(gt=[dict(is_right=True), dict(is_right=False)])
        gt_boxes = [np.array([0, 0, 5, 5.0]), np.array([9, 9, 15, 15.0])]

        def arm(img, boxes):  # one mask per prompt box, in order
            return [np.full(img.shape[:2], i, bool) for i in range(len(boxes))]

        self.assertIn("sam2l_ours", BOX_ARMS)
        p = seg_compare.run_arm("sam2l_ours", arm, np.zeros((16, 16, 3), np.uint8), fr, ours, gt_boxes)
        self.assertEqual([h["score"] for h in p], [0.9, 0.8])
        self.assertEqual([h["is_right"] for h in p], [True, False])
        g = seg_compare.run_arm("sam2l_gt", arm, np.zeros((16, 16, 3), np.uint8), fr, ours, gt_boxes)
        np.testing.assert_array_equal(g[1]["bbox"], gt_boxes[1])
        self.assertEqual([h["score"] for h in g], [1.0, 1.0])
        joined = seg_compare.with_joints([dict(bbox=np.array([0, 0, 4, 4.0])), dict(bbox=np.array([30, 30, 31, 31.0]))], ours)
        np.testing.assert_array_equal(joined[0]["joints2d"], ours[0]["joints2d"])
        self.assertIsNone(joined[1]["joints2d"])

    def test_nms_keep(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30.0]])
        self.assertEqual(seg_compare.nms_keep(boxes, np.array([0.5, 0.9, 0.7])), [1, 2])


class DetectorTimingTests(unittest.TestCase):
    def test_label(self):
        self.assertEqual(benchmark_detector.label("runs/detector-a/weights/best.pt"), "detector-a")
        self.assertEqual(benchmark_detector.label("thirdparty/WiLoR/pretrained_models/detector.pt"), "WiLoR")
        self.assertEqual(benchmark_detector.label("weights/shared/detector.pt"), "detector.pt")

    def test_timed_is_a_median_in_ms(self):
        calls = []
        ms = benchmark_detector.timed(lambda: calls.append(1), "cpu", 7)
        self.assertEqual(len(calls), 7)
        self.assertGreaterEqual(ms, 0.0)


if __name__ == "__main__":
    unittest.main()
