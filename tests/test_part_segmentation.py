"""Small CPU contracts for part/mask agreement and stored-crop geometry."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from bench.methods.fah import FAHMethod
from fah.detector import WiLoRHandDetector
from fah.estimator import HandEstimator
from scripts.render_full_modality import Renderer
from train.build_crops import crop_box, crop_keypoints, render_crop
from train.part_masks import COLORS


class PartCropTests(unittest.TestCase):
    def test_left_and_right_crop_boxes_enclose_the_stored_labels(self):
        box = np.array([70., 50., 125., 135.])
        corners = np.array([[70, 50, 1], [125, 50, 1], [70, 135, 1], [125, 135, 1]])
        for right in (False, True):
            with self.subTest(right=right):
                _, M = render_crop(np.zeros((240, 400, 3), np.uint8), box, right, size=64)
                record = dict(box=box.tolist(), M=M.tolist(), is_right=right, hw=[240, 400])
                b = crop_box(record)
                kp = crop_keypoints(corners, M, right, 400)[:, :2]
                np.testing.assert_allclose(b, np.r_[kp.min(0), kp.max(0)], atol=1e-5)
                self.assertTrue(np.all(b >= 0))
                self.assertTrue(np.all(b <= 64))


class PartMaskAgreementTests(unittest.TestCase):
    def test_every_detector_mask_pixel_gets_a_part_even_outside_pose_crop(self):
        height, width = 36, 48
        box = np.array([6, 8, 40, 30], np.float32)
        mask = np.zeros((height, width), bool)
        mask[8:30, 6:40] = True
        mask[8:13, 18:22] = False  # background between fingers stays background

        class Detector:
            last = {"boxes": box[None]}

            def masks(self, keep):
                return [mask.copy() for _ in keep]

            def mask_at(self, keep, pts):
                if export_masks:
                    raise AssertionError("The already-exported mask must be reused")
                return [mask[ys, xs] for xs, ys in pts]

        for right, export_masks in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(right=right, export_masks=export_masks):
                estimator = HandEstimator.__new__(HandEstimator)
                estimator.input_size = [16, 16]
                # The hand extends beyond the pose crop on all sides. The raw
                # head calls everything background but has valid part rankings.
                logits = np.zeros((1, 17, 4, 4), np.float32)
                logits[:, 0] = 10
                logits[:, 2, :, :2] = 5
                logits[:, 7, :, 2:] = 5
                raw = dict(part_logits=logits, affine_trans=np.array([[[1., 0, -14], [0, 1., -12]]]))
                hand = dict(bbox=box, is_right=right, score=1., keypoints_2d=np.zeros((21, 2)),
                            keypoints_3d=np.zeros((21, 3)), cam_int=np.eye(3))
                estimator._paste_parts([hand], raw, [right], height, width)
                method = FAHMethod.__new__(FAHMethod)
                method.verify = method.mesh_fit = method.masks = method.tip_extend = False
                method.det_masks, method.parts_in_mask = export_masks, True
                method.schedule = method.tracker = method.vreg = None
                method.detector = Detector()
                method.est = SimpleNamespace(process_image=lambda *a, **kw: [hand])
                output = method._call(np.zeros((height, width, 3), np.uint8), np.eye(3))[0]
                x, y, roi = output["parts_roi"]
                parts = np.zeros(mask.shape, np.uint8)
                parts[y:y + roi.shape[0], x:x + roi.shape[1]] = roi
                np.testing.assert_array_equal(parts > 0, mask)
                self.assertTrue(set(np.unique(parts[mask])).issubset({2, 7}))
                self.assertFalse(output["parts_roi_head"][2].any())


class DensePartRenderingTests(unittest.TestCase):
    def test_surface_colors_skip_the_leading_bone_points(self):
        renderer = Renderer.__new__(Renderer)
        renderer.dense_part = np.array([2, 7])
        renderer.m = lambda img, K: [dict(
            joints2d=np.full((21, 2), [4., 40.]),
            surface2d_head=np.r_[np.tile([4., 50.], (40, 1)), [[20., 45.], [40., 45.]]],
        )]
        sheet, count = renderer.panels(np.zeros((64, 64, 3), np.uint8), np.eye(3))
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(sheet[45, 128 + 20], COLORS[2])
        np.testing.assert_array_equal(sheet[45, 128 + 40], COLORS[7])


class DetectorMaskSamplingTests(unittest.TestCase):
    def test_point_queries_match_exported_masks_with_overlap_and_padding(self):
        detector = WiLoRHandDetector.__new__(WiLoRHandDetector)
        rng = np.random.default_rng(5)
        height, width = 24, 32
        detector.last = dict(
            hw=(height, width), boxes=np.array([[1.2, 2.5, 25.8, 22.4], [12.1, 0, 32, 19.7]], np.float32),
            proto=torch.tensor(rng.normal(size=(3, 8, 10)), dtype=torch.float32),
            mc=torch.tensor([[1., -.4, .3], [.2, 1., -.5]]), scale=.25, r=1., pad=(4, 4),
        )
        ys, xs = np.mgrid[-2:height + 2, -2:width + 2]
        xs, ys = xs.ravel(), ys.ravel()
        ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        for keep in ([0], [0, 1], [1, 0]):
            with self.subTest(keep=keep):
                masks = detector.masks(keep)
                sampled = detector.mask_at(keep, [(xs, ys)] * len(keep))
                for i, got in enumerate(sampled):
                    want = np.zeros(len(xs), bool)
                    want[ok] = masks[i, ys[ok], xs[ok]]
                    np.testing.assert_array_equal(got, want)
                rois = [(-2, -2, width + 4, height + 4)] * len(keep)
                for got, want in zip(detector.mask_roi(keep, rois), sampled):
                    np.testing.assert_array_equal(got.ravel(), want)


class PartOutputTransferTests(unittest.TestCase):
    def test_byte_maps_share_one_transfer_with_exact_float_outputs_per_batch(self):
        estimator = HandEstimator.__new__(HandEstimator)
        estimator.input_size = [16, 16]
        estimator.max_batch_size, estimator.decoder_depth = 2, None
        estimator.fast_mhr, estimator.refine_mesh_iters, estimator.graphed = False, 0, None
        estimator.model = SimpleNamespace(decoder_hand=SimpleNamespace(), head_pose_hand=SimpleNamespace())
        values = torch.arange(3 * 3 * 21, dtype=torch.float32).reshape(3, 3, 21) / 7
        logits = torch.arange(3 * 17 * 4 * 4, dtype=torch.float32).reshape(3, 17, 4, 4).sin()

        def make_batch(crops):
            n = len(crops)
            return dict(img=torch.tensor(crops).float().reshape(n, 1, 1, 1, 1),
                        bbox_center=torch.zeros(n, 1, 2), bbox_scale=torch.ones(n, 1, 2),
                        affine_trans=torch.eye(3)[:2][None, None].expand(n, 1, -1, -1))

        def forward(batch):
            i = batch["img"][:, 0, 0, 0, 0].long()
            return dict(pt_xyz=values[i].transpose(1, 2), part_logits=logits[i],
                        kp_dense=torch.tensor([40, 400])[None].expand(len(i), -1))

        estimator._make_batch, estimator._forward = make_batch, forward
        estimator.fast_batch = False
        expected = estimator._run_crops([0, 1, 2])
        estimator.fast_batch = True
        transfers = []
        original_cpu = torch.Tensor.cpu

        def cpu(tensor, *args, **kwargs):
            transfers.append((tensor.dtype, tensor.numel() * tensor.element_size()))
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", cpu):
            got = estimator._run_crops([0, 1, 2])
        for key in expected:
            np.testing.assert_array_equal(got[key], expected[key])
        self.assertEqual(got["part_lab"].dtype, np.uint8)
        self.assertEqual(got["part_fg"].dtype, np.uint8)
        self.assertEqual(got["pt_xyz"].dtype, np.float32)
        self.assertEqual(len(transfers), 2)  # two crop batches, one copy each
        self.assertTrue(all(dtype == torch.uint8 for dtype, _ in transfers))
        self.assertEqual(sum(size for _, size in transfers), sum(a.nbytes for a in got.values()))

    def test_cpu_part_maps_equal_the_plain_argmax_and_switch_off(self):
        estimator = HandEstimator.__new__(HandEstimator)
        estimator.input_size = [16, 16]
        estimator.max_batch_size, estimator.decoder_depth = 4, None
        estimator.fast_mhr, estimator.refine_mesh_iters, estimator.graphed, estimator.fast_batch = False, 0, None, False
        estimator.model = SimpleNamespace(decoder_hand=SimpleNamespace(), head_pose_hand=SimpleNamespace())
        logits = torch.randn(3, 17, 4, 4, generator=torch.Generator().manual_seed(0))
        estimator._make_batch = lambda crops: dict(img=torch.zeros(len(crops), 1, 1, 1, 1), bbox_center=torch.zeros(len(crops), 1, 2),
                                                   bbox_scale=torch.ones(len(crops), 1, 2),
                                                   affine_trans=torch.eye(3)[:2][None, None].expand(len(crops), 1, -1, -1))
        estimator._forward = lambda batch: dict(pt_xyz=torch.ones(3, 21, 3), part_logits=logits)
        got = estimator._run_crops([0, 1, 2])
        lg = torch.nn.functional.interpolate(logits, size=(16, 16), mode="bilinear", align_corners=False)
        np.testing.assert_array_equal(got["part_lab"], lg.argmax(1).numpy())  # channels-last argmax on the CPU
        np.testing.assert_array_equal(got["part_fg"], (lg[:, 1:].argmax(1) + 1).numpy())
        estimator.parts = False  # fah-live until key m: no part maps, the other outputs unchanged
        off = estimator._run_crops([0, 1, 2])
        self.assertNotIn("part_lab", off)
        self.assertNotIn("part_fg", off)
        np.testing.assert_array_equal(off["pt_xyz"], got["pt_xyz"])


class PartSegLossTests(unittest.TestCase):
    def test_weighted_loss_and_gradients_match_cross_entropy(self):
        from train.finetune import part_seg_loss

        for finger_weight in (0.1, 2.0):
            logits = torch.linspace(-1, 1, 34 * 4).reshape(2, 17, 2, 2).requires_grad_()
            labels = torch.tensor([[[2, 255], [255, 255]], [[1, 1], [1, 1]]], dtype=torch.uint8)
            if finger_weight == 2.0:
                labels[0] = torch.tensor([[0, 1], [2, 3]])  # unequal class weights must preserve the weighted denominator
            has = torch.tensor([1., 0.])
            target = labels.long()
            target[1] = 255
            weight = torch.ones(17)
            weight[2:] = finger_weight
            reference = torch.nn.functional.cross_entropy(logits, target, weight=weight, ignore_index=255)
            actual = part_seg_loss(logits, labels, has, finger_weight)
            torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                       torch.autograd.grad(reference, logits)[0])

    def test_missing_and_all_ignored_labels_allow_zero_gradient_backward(self):
        from train.finetune import part_seg_loss

        for label, has in ((255, 1.), (2, 0.)):
            logits = torch.zeros(1, 17, 2, 2, requires_grad=True)
            loss = part_seg_loss(logits, torch.full((1, 4, 4), label, dtype=torch.uint8), torch.tensor([has]))
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


class DensePartLossTests(unittest.TestCase):
    def test_ignore_boundary_cannot_pull_points_away_from_their_part(self):
        from train.finetune import part_loss

        labels = torch.zeros(1, 32, 32, dtype=torch.uint8)
        labels[:, :, :12] = 255
        labels[:, 8:24, 23:30] = 2
        uv = torch.tensor([[[13., 16.]]], requires_grad=True)
        region, _ = part_loss(uv, labels, torch.ones(1), torch.tensor([2]), 32, res=32, n_cover=4)
        region.backward()
        self.assertLess(uv.grad[0, 0, 0].item(), 0)  # move right to the finger, not left into ignored pixels

    def test_absent_part_does_not_supply_a_region_target(self):
        from train.finetune import part_loss

        uv = torch.tensor([[[13., 16.]]], requires_grad=True)
        region, cover = part_loss(uv, torch.ones(1, 32, 32, dtype=torch.uint8), torch.ones(1),
                                  torch.tensor([2]), 32, res=32, n_cover=4)
        self.assertEqual((region + cover).item(), 0.)
        (region + cover).backward()
        torch.testing.assert_close(uv.grad, torch.zeros_like(uv))

    def test_region_moves_visible_points_toward_their_own_part(self):
        from train.finetune import part_loss

        labels = torch.zeros(1, 32, 32, dtype=torch.uint8)
        labels[:, 8:20, 20:28] = 2
        uv = torch.tensor([[[18., 14.]]], requires_grad=True)
        region, _ = part_loss(uv, labels, torch.ones(1), torch.tensor([2]), 32, res=32, sigmas=(1.,), n_cover=4)
        region.backward()
        self.assertLess(uv.grad[0, 0, 0].item(), 0)  # gradient descent moves right, toward this point's part
        self.assertTrue(torch.isfinite(uv.grad).all())

    def test_cover_spreads_points_to_the_visible_region_ends(self):
        from train.finetune import part_loss

        labels = torch.zeros(1, 32, 32, dtype=torch.uint8)
        labels[:, 14, [10, 28]] = 2
        uv = torch.tensor([[[18., 14.5], [20., 14.5]]], requires_grad=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            _, cover = part_loss(uv, labels, torch.ones(1), torch.tensor([2, 2]), 32,
                                 res=32, sigmas=(1.,), n_cover=64)
        cover.backward()
        self.assertGreater(uv.grad[0, 0, 0].item(), 0)  # left point moves left
        self.assertLess(uv.grad[0, 1, 0].item(), 0)  # right point moves right

    def test_hidden_points_and_unlabelled_samples_are_not_pulled(self):
        from train.finetune import part_loss

        labels = torch.full((1, 32, 32), 2, dtype=torch.uint8)
        for has, visible in ((0., 1.), (1., 0.)):
            with self.subTest(has=has, visible=visible):
                uv = torch.tensor([[[18., 14.]]], requires_grad=True)
                region, cover = part_loss(uv, labels, torch.tensor([has]), torch.tensor([2]), 32,
                                         vis=torch.tensor([[visible]]), res=32, sigmas=(1.,), n_cover=4)
                (region + cover).backward()
                self.assertEqual((region + cover).item(), 0)
                torch.testing.assert_close(uv.grad, torch.zeros_like(uv))


if __name__ == "__main__":
    unittest.main()
