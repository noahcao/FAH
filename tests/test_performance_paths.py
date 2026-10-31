"""Numerical and operation-count checks for optimized inference paths."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from fah.detector import WiLoRHandDetector
from fah.estimator import HandEstimator
from fah.hand_model import SAM3DHand
from fah.models.heads.keypoint_head import HandKeypointHead


class NonzeroCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.nonzero.default:
            self.count += 1
        return func(*args, **(kwargs or {}))


class DetectorPostprocessTests(unittest.TestCase):
    def make_detector(self, predictions, seg=False):
        detector = WiLoRHandDetector.__new__(WiLoRHandDetector)
        detector.device, detector.dtype = "cpu", torch.float32
        detector.conf, detector.iou, detector.seg = .1, .5, seg
        detector.model = SimpleNamespace(names={0: "left", 1: "right"})
        detector._letterbox = lambda image: (image, 1., 0, 0)
        detector.trt = lambda x: {
            "out": torch.tensor(predictions, dtype=torch.float32).T[None],
            "proto": torch.zeros(1, 2, 2, 2),
        }
        return detector

    def test_classwise_nms_and_mask_coefficients_keep_the_same_order(self):
        detector = self.make_detector([
            [5, 5, 8, 8, .9, .2, 10, 11],
            [5, 5, 8, 8, .8, .1, 20, 21],  # duplicate left hand
            [5, 5, 8, 8, .1, .85, 30, 31],  # overlapping right hand survives
            [9, 9, 6, 6, .7, .1, 40, 41],  # clipped at the image border
            [1, 1, 2, 2, .1, .05, 50, 51],  # threshold is strict
        ], seg=True)
        original_cpu = torch.Tensor.cpu
        transfers = []

        def cpu(tensor, *args, **kwargs):
            transfers.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with NonzeroCounter() as counts, patch.object(torch.Tensor, "cpu", cpu):
            boxes, scores, sides = detector._fast_call(np.zeros((10, 10, 3), np.uint8))
        np.testing.assert_array_equal(boxes, [[1, 1, 9, 9], [1, 1, 9, 9], [6, 6, 10, 10]])
        np.testing.assert_allclose(scores, [.9, .85, .7])
        np.testing.assert_array_equal(sides, [False, True, False])
        np.testing.assert_array_equal(detector.last["mc"].numpy(), [[10, 11], [30, 31], [40, 41]])
        self.assertEqual(counts.count, 1)
        self.assertEqual(transfers, [(3, 6)])
        self.assertEqual(boxes.dtype, np.float32)
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(sides.dtype, np.bool_)
        self.assertTrue(boxes.flags.c_contiguous)

    def test_no_detection_preserves_empty_shapes(self):
        detector = self.make_detector([[5, 5, 8, 8, .1, .05]], seg=False)
        boxes, scores, sides = detector._fast_call(np.zeros((10, 10, 3), np.uint8))
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(sides.shape, (0,))


class DenseDecodeTests(unittest.TestCase):
    def make_inputs(self, vertices, depth=True):
        rng = np.random.default_rng(314)
        estimator = HandEstimator.__new__(HandEstimator)
        estimator.input_size = [320, 320]
        estimator.lift_2d = False
        estimator.surface_smooth = estimator.surface_conf = 0
        estimator.model = SimpleNamespace(keypoint_head=SimpleNamespace(dense_vidx=None))
        n = 21 + vertices
        out = dict(
            pt_xyz=rng.normal(0, .02, (2, n, 3)).astype(np.float32),
            pt_uv=rng.uniform(.1, .9, (2, n, 2)).astype(np.float32),
            kp_dense=np.array([[0, vertices]] * 2),
            affine_trans=np.array([[[.8, 0, -2], [0, .8, -7]]] * 2, np.float32),
        )
        if depth:
            out["pt_depth"] = np.array([-1.2, -.7], np.float32)
        boxes = np.array([[0, 0, 40, 40]] * 2, np.float32)
        K = np.array([[650., 0, 17], [0, 440., 42], [0, 0, 1]], np.float32)
        args = (out, boxes, boxes, np.array([True, False]), np.ones(2), K, 640)
        return estimator, args

    def reference_translation(self, args, i):
        out, _, _, sides, _, K, width = args
        A = out["affine_trans"][i]
        uv = (out["pt_uv"][i] * 320 - A[:, 2]) @ np.linalg.inv(A[:, :2]).T
        xyz = out["pt_xyz"][i].astype(np.float64)
        if not sides[i]:
            uv[:, 0] = width - 1 - uv[:, 0]
            xyz[:, 0] *= -1
        u = uv - K[:2, 2]
        fx, fy = K[0, 0], K[1, 1]
        n = len(xyz)
        M = np.zeros((2 * n, 3))
        M[:n, 0], M[:n, 2] = fx, -u[:, 0]
        M[n:, 1], M[n:, 2] = fy, -u[:, 1]
        rhs = np.concatenate([u[:, 0] * xyz[:, 2] - fx * xyz[:, 0],
                              u[:, 1] * xyz[:, 2] - fy * xyz[:, 1]])
        if "pt_depth" not in out:
            return np.linalg.lstsq(M, rhs, rcond=None)[0]
        tz = float(np.exp(out["pt_depth"][i]) * fx * A[0, 0] / 320)
        return np.r_[np.linalg.lstsq(M[:, :2], rhs - M[:, 2] * tz, rcond=None)[0], tz]

    def test_learned_depth_matches_least_squares_without_a_solver(self):
        for vertices in (0, 4, 400, 778):
            with self.subTest(vertices=vertices):
                estimator, args = self.make_inputs(vertices)
                expected = [self.reference_translation(args, i) for i in range(2)]
                with patch("numpy.linalg.lstsq", side_effect=AssertionError("unnecessary solver")):
                    hands = estimator._kp_only_results(*args)
                for hand, translation in zip(hands, expected):
                    np.testing.assert_allclose(hand["cam_t"], translation, atol=1e-7, rtol=1e-7)

    def test_without_learned_depth_keeps_the_full_translation_fit(self):
        estimator, args = self.make_inputs(4, depth=False)
        expected = [self.reference_translation(args, i) for i in range(2)]
        hands = estimator._kp_only_results(*args)
        for hand, translation in zip(hands, expected):
            np.testing.assert_allclose(hand["cam_t"], translation, atol=1e-7, rtol=1e-7)

    def test_fixed_surface_indices_are_downloaded_once_and_exports_are_independent(self):
        estimator, args = self.make_inputs(4)

        class Indices:
            calls = 0

            def cpu(self):
                self.calls += 1
                return torch.tensor([0, 2, 4, 6])

        indices = Indices()
        estimator.model.keypoint_head.dense_vidx = indices
        estimator.sub_R = np.eye(7, 4, dtype=np.float32)
        first = estimator._kp_only_results(*args)
        first[0]["surface_vidx"][0] = 99
        second = estimator._kp_only_results(*args)
        self.assertEqual(indices.calls, 1)
        np.testing.assert_array_equal(first[1]["surface_vidx"], [0, 2, 4, 6])
        np.testing.assert_array_equal(second[0]["surface_vidx"], [0, 2, 4, 6])


class SharedHeadComputationTests(unittest.TestCase):
    def test_dense_forward_reuses_token_and_heatmap_probabilities(self):
        torch.manual_seed(10)
        head = HandKeypointHead(img_dim=16, tok_dim=12, dim=8, stem=4,
                                pool=True, depth=True, dense=(0, 4)).eval()
        verify = torch.nn.Linear(12, 2).eval()
        image = torch.rand(2, 1, 3, 32, 32)
        embedding, condition = torch.rand(2, 16, 2, 2), torch.rand(2, 3)
        model = SimpleNamespace(
            cfg=SimpleNamespace(MODEL=SimpleNamespace(
                BACKBONE=SimpleNamespace(TYPE="synthetic"), PROMPT_ENCODER={},
            )),
            _initialize_batch=lambda batch: None,
            _flatten_person=lambda x: x.flatten(0, 1),
            data_preprocess=lambda x, unit_range: x,
            get_ray_condition=lambda batch: torch.zeros(2, 1, 2, 2, 2),
            _get_decoder_condition=lambda batch: condition,
            ray_cond_emb_hand=lambda emb, rays: emb,
            kp_only=True, keypoint_head=head, verify_head=verify,
        )
        tokens = embedding.flatten(2).transpose(1, 2)
        with torch.no_grad():
            uv, xyz, logits = head(tokens, (2, 2), None, image[:, 0], cond=condition)
            pose = head.pool(torch.cat([tokens.mean(1), condition], -1))
            expected = dict(pt_uv=uv, pt_xyz=xyz, pt_conf=logits.flatten(2).softmax(-1).amax(-1),
                            pt_depth=head.depth(pose)[:, 0], pt_token=pose, verify_logits=verify(pose))

            class SoftmaxCounter(TorchDispatchMode):
                count = 0

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if func == torch.ops.aten._softmax.default:
                        self.count += 1
                    return func(*args, **(kwargs or {}))

            with SoftmaxCounter() as softmax, patch.object(head.pool, "forward", wraps=head.pool.forward) as pool:
                actual = SAM3DHand.forward(model, dict(img=image), image_embeddings=embedding)
            self.assertEqual(pool.call_count, 1)
            self.assertEqual(softmax.count, 1)
            for key, value in expected.items():
                with self.subTest(output=key):
                    torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
