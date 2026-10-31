"""Opt-in contact flag smoothing (contact_smooth=2, fah.api.CONTACT_SMOOTH_H): the video contact flag's hysteresis on the
mean of this and the previous frame's flag probability; the default is the shipped rule. With weights (FAH_TEST_VIDEO, a
short public HOT3D clip): the default equals contact_smooth=1, and contact_smooth=2 changes only the flags."""
import inspect
import os
import shutil
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from bench.methods.fah import FAHMethod
from fah import api, cli, video_demo
from test_coreml_paths import weights_dir


def video_contact(smooth, h=None):
    from fah.contact_video import ContactVideo, VideoContact

    net = ContactVideo(n_vert=6, dim=16, layers=1, heads=2, tok_dim=0)
    ck = dict(args=dict(T=4, dim=16, layers=1), n_vert=6, use_tok=False, tok_dim=0, model=net.state_dict(), h_on=0.9, h_off=0.1)
    vc = VideoContact(ck, np.eye(778, 6), device="cpu", smooth=smooth, h=h)
    p = [0.5]
    vc._forward = lambda x, tok: (x[..., 3:9] * 0, torch.full((1, vc.T), float(np.log(p[0] / (1 - p[0]))), dtype=torch.float64))
    return vc, p


def run(vc, p, seq):
    flags, scores = [], []
    for x in seq:
        if x is None:
            vc([])
            flags.append(None)
            continue
        p[0] = x
        h = dict(is_right=True, score=0.9, joints3d=np.random.rand(21, 3) + [0, 0, 0.5], contact_dense=np.full(6, 0.7), contact_score=0.8)
        vc([h])
        flags.append(h["contact_flag_video"])
        scores.append(h["contact_score_video"])
    return flags, scores


class RuleTests(unittest.TestCase):
    seq = [0.8, 0.1, 0.1, 0.9, None, 0.75, 0.6, 0.1, 0.25]

    def test_smooth_2_runs_the_hysteresis_on_the_mean_of_two_hand_frames(self):
        flags, scores = run(*video_contact(2, (0.7, 0.15)), self.seq)
        # means: .8 (no previous), .45, .1, .5, gap, .75 (the gap restarts the mean), .675, .35, .175
        self.assertEqual(flags, [True, True, False, False, None, True, True, True, True])
        np.testing.assert_allclose(scores, [x for x in self.seq if x is not None])  # contact_score_video: the raw probability

    def test_smooth_1_is_the_shipped_rule(self):
        vc, p = video_contact(1, (0.7, 0.15))
        self.assertEqual(run(vc, p, self.seq)[0], [True, False, False, True, None, True, True, False, False])
        vc, p = video_contact(1)  # the checkpoint's thresholds (0.9 / 0.1) without h
        self.assertEqual((vc.h_on, vc.h_off), (0.9, 0.1))
        self.assertEqual(run(vc, p, self.seq)[0], [False, False, False, False, None, False, False, False, False])

    def test_the_flag_restarts_after_the_window_emptied(self):
        vc, p = video_contact(2, (0.7, 0.15))
        self.assertEqual(run(vc, p, [0.9, 0.5, None, None, None, None, 0.5])[0], [True, True] + [None] * 4 + [False])


class ArgsTests(unittest.TestCase):
    def test_per_model_thresholds_with_the_video_contact_model_only(self):
        self.assertEqual(set(api.CONTACT_SMOOTH_H), set(api.MODELS))
        self.assertEqual(inspect.signature(api.method_args).parameters["contact_smooth"].default, 1)
        w, s = weights_dir()
        for m in api.MODELS[1:]:
            shutil.copytree(os.path.join(w, "fah-s"), os.path.join(w, m))
        for m in api.MODELS:
            open(os.path.join(w, m, "contact.pt"), "w").close()
            open(os.path.join(w, m, "contact_video.pt"), "w").close()
            a = api.method_args(m, "video", w, "cpu", s, contact=True, contact_smooth=2)
            self.assertEqual((a["contact_smooth"], a["contact_h"]), (2, api.CONTACT_SMOOTH_H[m]))
            for b in (api.method_args(m, "video", w, "cpu", s, contact=True),
                      api.method_args(m, "video", w, "cpu", s, contact=True, contact_smooth=1),
                      api.method_args(m, "video", w, "cpu", s, contact=True, contact_video=False, contact_smooth=2)):
                self.assertFalse({"contact_smooth", "contact_h"} & set(b))
        with self.assertRaises(AssertionError):
            api.method_args("fah", "video", w, "cpu", s, contact=True, contact_smooth=3)

    def test_fah_method_passes_the_rule_to_the_video_contact_model(self):
        p = inspect.signature(FAHMethod.__init__).parameters
        self.assertEqual((p["contact_smooth"].default, p["contact_h"].default), (1, None))
        kh = SimpleNamespace(contact_bias=None)
        for smooth, h in ((1, None), (2, (0.7, 0.15))):
            m = FAHMethod.__new__(FAHMethod)
            m.contact_video, m._vc, m.contact_smooth, m.contact_h = "contact_video.pt", None, smooth, h
            m.est = SimpleNamespace(model=SimpleNamespace(keypoint_head=kh, device="cpu"), sub_R=np.eye(778, 6))
            with mock.patch("fah.contact_video.VideoContact") as vc:
                m._video_contact([({}, {})], [{}], [{}])
            self.assertEqual((vc.call_args.kwargs["smooth"], vc.call_args.kwargs["h"]), (smooth, h))

    def test_command_line_flags(self):
        a = cli.parse_args(["--vid", "x.mp4", "--contact", "--contact_smooth", "2"])
        self.assertEqual(cli.model_kwargs(a)["contact_smooth"], 2)
        self.assertEqual(cli.meta(a)["contact_smooth"], 2)
        b = cli.parse_args(["--vid", "x.mp4", "--contact"])
        self.assertNotIn("contact_smooth", cli.model_kwargs(b))
        self.assertNotIn("contact_smooth", cli.meta(b))
        self.assertEqual(video_demo.parser().parse_args(["x.mp4", "--contact_smooth", "2"]).contact_smooth, 2)
        self.assertEqual(video_demo.parser().parse_args(["x.mp4"]).contact_smooth, 1)


@unittest.skipUnless(os.path.exists(os.environ.get("FAH_TEST_VIDEO", "")) and os.path.isdir(os.path.join(api.DEFAULT_WEIGHTS, "fah")) and
                     os.path.isdir(api.DEFAULT_SAM3D), "needs FAH_TEST_VIDEO and the weights")
class ClipTests(unittest.TestCase):
    def run_clip(self, **kw):
        import cv2

        m = api.FAH("fah", "video", trt_dir=os.environ.get("FAH_TEST_TRT", ""), contact=True, **kw)
        m.reset()
        cap, out = cv2.VideoCapture(os.environ["FAH_TEST_VIDEO"]), []
        for _ in range(16):
            ok, img = cap.read()
            if not ok:
                break
            out.append(m(img, None))
        return m, out

    def test_default_equals_smooth_1_and_smooth_2_changes_only_the_flags(self):
        _, a = self.run_clip()
        _, b = self.run_clip(contact_smooth=1)
        m, c = self.run_clip(contact_smooth=2)
        self.assertEqual((m.method._vc.smooth, (m.method._vc.h_on, m.method._vc.h_off)), (2, api.CONTACT_SMOOTH_H["fah"]))
        self.assertEqual([len(f) for f in a], [len(f) for f in b])
        self.assertEqual([len(f) for f in a], [len(f) for f in c])
        self.assertTrue(any("contact_score_video" in h for f in a for h in f))
        for fa, fb, fc in zip(a, b, c):
            for x, y, z in zip(fa, fb, fc):
                for k in ("joints3d", "contact", "contact_video", "contact_score_video", "contact_flag_video"):
                    if k in x:
                        np.testing.assert_array_equal(np.asarray(x[k]), np.asarray(y[k]))
                for k in ("joints3d", "contact", "contact_video", "contact_score_video"):
                    if k in x:
                        np.testing.assert_array_equal(np.asarray(x[k]), np.asarray(z[k]))
