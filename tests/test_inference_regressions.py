"""Small CPU-only checks; no model weights, downloads, or inference workloads."""
import unittest
import weakref
from contextvars import Context
from unittest.mock import patch

import numpy as np
import torch

from bench.methods.fah import FAHMethod
from fah.estimator import HandEstimator
from fah.frame_cache import frame_cache, frame_tensor
from fah.video import HandTracker, LearnedSmoother
from fah.vis import project


class FrameCacheTests(unittest.TestCase):
    def test_standalone_call_reads_reused_camera_buffer(self):
        image = np.zeros((2, 2, 3), np.uint8)
        first = frame_tensor(image, "cpu")
        image.fill(255)
        second = frame_tensor(image, "cpu")
        self.assertTrue(torch.all(first == 0))
        self.assertTrue(torch.all(second == 255))

    def test_nested_passes_share_upload_but_next_frame_is_fresh(self):
        image = np.zeros((2, 2, 3), np.uint8)
        with patch("fah.frame_cache.torch.from_numpy", wraps=torch.from_numpy) as upload:
            with frame_cache():
                first = frame_tensor(image, "cpu")
                with frame_cache():
                    self.assertIs(frame_tensor(image, "cpu"), first)
            image.fill(123)
            with frame_cache():
                second = frame_tensor(image, "cpu")
            self.assertEqual(upload.call_count, 2)
        self.assertTrue(torch.all(second == 123))

    def test_different_views_at_same_address_do_not_share_upload(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        view = image.transpose(1, 0, 2)
        self.assertEqual(image.ctypes.data, view.ctypes.data)
        with frame_cache():
            first = frame_tensor(image, "cpu")
            second = frame_tensor(view, "cpu")
        self.assertIsNot(first, second)
        np.testing.assert_array_equal(second.numpy(), view.transpose(2, 0, 1)[None])

    def test_exception_releases_cached_image_and_tensor(self):
        def fail():
            image = np.zeros((2, 2, 3), np.uint8)
            with frame_cache():
                tensor = frame_tensor(image, "cpu")
                refs.extend([weakref.ref(image), weakref.ref(tensor)])
                raise RuntimeError("inference failed")

        refs = []
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            fail()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_independent_contexts_do_not_share_cache(self):
        image = np.zeros((2, 2, 3), np.uint8)

        def other_context():
            with frame_cache():
                return frame_tensor(image, "cpu")

        with frame_cache():
            first = frame_tensor(image, "cpu")
            second = Context().run(other_context)
            self.assertIsNot(first, second)
            self.assertIs(frame_tensor(image, "cpu"), first)

    def test_inference_entry_points_refresh_and_share_across_pose_passes(self):
        # Exercise the real public wrappers with just the detector/model work stubbed.
        estimator = HandEstimator.__new__(HandEstimator)
        estimator.persp_crop = False
        estimator.refine_box_iters = estimator.recrop_border = 0
        boxes = np.array([[0, 0, 2, 2]], np.float32)
        scores, sides = np.ones(1), np.ones(1, bool)
        observed = []

        def detect(image):
            observed.append(frame_tensor(image, "cpu"))
            return boxes, scores, sides

        def crops(items):
            observed.append(frame_tensor(items[0][0], "cpu"))
            return {}

        estimator.detector = detect
        estimator._run_crops = crops
        estimator._results = lambda *args: []
        method = FAHMethod.__new__(FAHMethod)
        method.verify = True
        method.tracker = None
        method.mesh_fit = method.masks = method.det_masks = False
        method.tip_extend = method.parts_in_mask = False

        def verified(image, K):
            detect(image)
            # Verification can run the pose estimator again with flipped handedness.
            for side in (sides, ~sides):
                estimator.process_image(image, boxes=boxes, is_right=side, cam_int=K)
            return []

        method._verified = verified
        for infer in (estimator.process_image, lambda img: method(img, np.eye(3))):
            with self.subTest(entry_point=infer):
                image = np.zeros((2, 2, 3), np.uint8)
                for value in (10, 20):
                    image.fill(value)
                    observed.clear()
                    with patch("fah.frame_cache.torch.from_numpy", wraps=torch.from_numpy) as upload:
                        infer(image)
                    self.assertEqual(upload.call_count, 1)
                    self.assertGreaterEqual(len(observed), 2)
                    self.assertTrue(all(tensor is observed[0] for tensor in observed))
                    self.assertTrue(torch.all(observed[0] == value))


class FrameCacheDeviceTests(unittest.TestCase):
    def test_equivalent_device_spellings_share_one_upload(self):
        image = np.zeros((2, 2, 3), np.uint8)
        with frame_cache():
            with patch("fah.frame_cache.torch.from_numpy", wraps=torch.from_numpy) as upload:
                a = frame_tensor(image, "cpu")
                b = frame_tensor(image, torch.device("cpu"))
        self.assertEqual(upload.call_count, 1)
        self.assertIs(a, b)


class TF32ScopeTests(unittest.TestCase):
    def test_tf32_is_per_call_and_restored(self):
        prev = torch.backends.cuda.matmul.allow_tf32
        seen = []
        methods = []
        for on in (True, False):
            m = FAHMethod.__new__(FAHMethod)
            m.tf32 = on
            m._call = lambda image, K: seen.append(torch.backends.cuda.matmul.allow_tf32) or []
            methods.append(m)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            for m in methods + methods[::-1]:
                m(np.zeros((2, 2, 3), np.uint8), np.eye(3))
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(seen, [True, False, False, True])
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev


def hand_at(x):
    cam = np.array([x, 0, .5], np.float32)
    rel = np.random.default_rng(7).normal(0, .01, (21, 3)).astype(np.float32)
    rel[0] = 0
    joints = rel + cam
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], np.float32)
    return dict(
        is_right=True, score=1., bbox=np.array([0, 0, 100, 100]),
        cam_t=cam, cam_int=K, keypoints_3d=joints.copy(), keypoints_3d_rel=rel.copy(),
        keypoints_3d_mano=joints.copy(), keypoints_2d=project(joints, K),
        vertices=joints[:6].copy(), vertices_mano=joints[:6].copy(),
        vertices_dense=joints[:6].copy(),
    )


class TemporalOutputTests(unittest.TestCase):
    def test_gap_fill_does_not_reuse_previous_frame_pixel_predictions(self):
        smoother = LearnedSmoother.__new__(LearnedSmoother)
        smoother.T, smoother.blend, smoother.device = 4, .75, "cpu"
        smoother.gap, smoother.max_shift = 2, .05
        smoother.mesh_follow = smoother.anchor_2d = False
        smoother.keep_2d, smoother.euro, smoother.euro_2d = True, None, None
        smoother.reset()

        def forward(x):
            # Hold the last observed 3D pose through the missing frame.
            real = x[:, :1]
            return real[..., 2:65].reshape(1, 1, 21, 3), real[..., 107:110]

        smoother._forward = forward
        hand = hand_at(0)
        pixels = dict(
            parts_roi=(10, 20, np.ones((3, 4), np.uint8)),
            parts_fg_roi=(10, 20, np.ones((3, 4), np.uint8)),
            mask=np.ones((100, 100), bool),
            mask_arm=np.ones((100, 100), bool),
            contact=np.ones(778, np.float16), contact_dense=np.ones(400, np.float16), contact_score=.9,
            contact_fingers=np.ones(6, np.float32),
        )
        hand.update(pixels)
        observed = smoother.update([hand], 0)[0]
        filled = smoother.update([], 1 / 30)[0]
        for key, value in pixels.items():
            with self.subTest(field=key):
                self.assertIs(observed[key], value)
                self.assertNotIn(key, filled.keys())
        np.testing.assert_allclose(filled["keypoints_3d"], observed["keypoints_3d"], atol=1e-7)
        self.assertIn("vertices_dense", filled)
        self.assertEqual(filled["score"], 0.)

    def test_mask_export_handles_a_hand_without_pixel_predictions_first(self):
        method = FAHMethod.__new__(FAHMethod)
        method.verify = True
        method.tracker = None
        method.mesh_fit = method.masks = method.det_masks = False
        method.tip_extend = method.parts_in_mask = False
        method.vreg = None
        hands = [hand_at(0), hand_at(.1)]
        hands[1]["mask"] = np.ones((2, 2), bool)
        method._verified = lambda image, K: hands
        out = method(np.zeros((2, 2, 3), np.uint8), np.eye(3))
        self.assertNotIn("mask_head", out[0])
        from bench.seg.rle import decode
        np.testing.assert_array_equal(decode(out[1]["mask_head"]), hands[1]["mask"])

    def test_tracker_smooths_all_mesh_representations_with_translation(self):
        tracker = HandTracker()
        tracker.update([hand_at(0)], 0)
        hand = hand_at(.04)
        original_rel = hand["vertices_dense"] - hand["cam_t"]
        tracker.update([hand], 1 / 30)
        self.assertLess(hand["cam_t"][0], .04)
        for key in ("vertices", "vertices_mano", "vertices_dense"):
            with self.subTest(mesh=key):
                np.testing.assert_allclose(hand[key] - hand["cam_t"], original_rel, atol=1e-7)
        np.testing.assert_allclose(hand["keypoints_3d_rel"] + hand["cam_t"], hand["keypoints_3d"])

    def test_learned_refiner_rebuilds_relative_joints_and_keeps_requested_2d(self):
        # Only replace the network: use deterministic translation and articulation updates.
        smoother = LearnedSmoother.__new__(LearnedSmoother)
        smoother.T, smoother.blend, smoother.device = 4, .75, "cpu"
        smoother.gap, smoother.max_shift = 2, .05
        smoother.mesh_follow = smoother.anchor_2d = False
        smoother.keep_2d, smoother.euro, smoother.euro_2d = True, None, None
        smoother.reset()

        def forward(x):
            rel = x[..., 2:65].reshape(1, -1, 21, 3).clone()
            rel[..., 1:, 0] += .02  # 2 mm articulation change (network units: dm)
            root = x[..., 107:110].clone()
            root[..., 0] += .02  # 1 cm at z=.5
            return rel, root

        smoother._forward = forward
        for t, hands in ((0, [hand_at(0)]), (1 / 30, [])):
            original_2d = hands[0]["keypoints_2d"].copy() if hands else None
            hand = smoother.update(hands, t)[0]
            np.testing.assert_allclose(
                hand["keypoints_3d_rel"] + hand["cam_t"], hand["keypoints_3d"], atol=1e-7,
            )
            if original_2d is not None:
                np.testing.assert_array_equal(hand["keypoints_2d"], original_2d)


if __name__ == "__main__":
    unittest.main()
