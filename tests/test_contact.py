"""Hand-object contact head: CPU-only checks (no weights)."""
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import torch

from bench.methods.fah import FAHMethod
from fah import api
from fah.cli import to_json
from fah.models.heads.keypoint_head import HandKeypointHead
from test_coreml_paths import weights_dir
from test_inference_regressions import hand_at


class ContactHeadTests(unittest.TestCase):
    def test_contact_logits_and_unchanged_pose_outputs(self):
        torch.manual_seed(0)
        kw = dict(img_dim=16, tok_dim=12, dim=8, stem=4, pool=True, dense=(2, 6), vis=True)
        plain = HandKeypointHead(**kw).eval()
        head = HandKeypointHead(contact=2, **kw).eval()
        missing, unexpected = head.load_state_dict(plain.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing and all(k.startswith("contact_") for k in missing))
        args = (torch.rand(3, 4, 16), (2, 2), torch.rand(3, 12), torch.rand(3, 3, 32, 32))
        with torch.no_grad():
            a, b = plain(*args), head(*args)
        for x, y in zip(a, b):  # uv, xyz, logits: the contact head reads detached features only
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(tuple(head.last_contact.shape), (3, 6))  # one logit per dense surface point
        self.assertEqual(tuple(head.last_contact_hand.shape), (3,))
        self.assertFalse(hasattr(plain, "last_contact"))
        logits, hand = head.last_contact.clone(), head.last_contact_hand.clone()
        head.contact_bias.fill_(1.25)  # calibration offset: subtracted from the vertex logits, not from the hand logit
        with torch.no_grad():
            head(*args)
        self.assertEqual(tuple(head.last_contact_k.shape), (3, 6, 1))
        torch.testing.assert_close(head.last_contact, logits - 1.25)
        torch.testing.assert_close(head.last_contact_hand, hand, rtol=0, atol=0)

    def test_geometry_input_options(self):
        torch.manual_seed(0)
        kw = dict(img_dim=16, tok_dim=12, dim=8, stem=4, pool=True, dense=(2, 6), vis=True)
        plain = HandKeypointHead(**kw).eval()
        args = (torch.rand(3, 4, 16), (2, 2), torch.rand(3, 12), torch.rand(3, 3, 32, 32))
        img = HandKeypointHead(contact=2, contact_geo=False, **kw).eval()  # image tokens only: no per-point geometry input
        self.assertIsNone(img.contact_in)
        img.load_state_dict(plain.state_dict(), strict=False)
        with torch.no_grad():
            a, b = plain(*args), img(*args)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(tuple(img.last_contact.shape), (3, 6))
        ext = HandKeypointHead(contact=2, contact_geo="external", **kw).eval()  # the caller builds the point inputs
        with torch.no_grad():
            ext(*args)
            self.assertFalse(hasattr(ext, "last_contact"))
            logits, hand = ext.contact_from(torch.rand(3, 6, 2 * 8 + 4), ext.last_pose, ext.last_f)
        self.assertEqual((tuple(logits.shape), tuple(hand.shape)), ((3, 6), (3,)))

    def test_several_tolerances_keep_10mm_as_the_main_output(self):
        head = HandKeypointHead(img_dim=16, tok_dim=12, dim=8, stem=4, pool=True, dense=(2, 6), vis=True, contact=1,
                                contact_dists=(5.0, 10.0, 20.0), contact_dim=64).eval()
        with torch.no_grad():
            head(torch.rand(2, 4, 16), (2, 2), torch.rand(2, 12), torch.rand(2, 3, 32, 32))
        self.assertEqual((tuple(head.last_contact_k.shape), tuple(head.last_contact_hand_k.shape)), ((2, 6, 3), (2, 3)))
        torch.testing.assert_close(head.last_contact, head.last_contact_k[..., 1])
        self.assertEqual(head.contact_dists.tolist(), [5.0, 10.0, 20.0])
        with torch.no_grad():  # one hand: still dense tensors (the estimator packs every output into one byte copy)
            head(torch.rand(1, 4, 16), (2, 2), torch.rand(1, 12), torch.rand(1, 3, 32, 32))
        for t in (head.last_contact, head.last_contact_hand):
            self.assertEqual(t.stride()[-1], 1)
            t.reshape(1, -1).view(torch.uint8)

    def test_encoder_tokens_join_the_contact_memory_on_request(self):
        torch.manual_seed(0)
        kw = dict(img_dim=16, tok_dim=12, dim=8, stem=4, pool=True, dense=(2, 6), vis=True, contact=1, contact_dim=64)
        head = HandKeypointHead(**kw, contact_tokmem=True).eval()
        plain = HandKeypointHead(**kw).eval()
        plain.load_state_dict({k: v for k, v in head.state_dict().items() if "contact_tokmem" not in k})
        args = (torch.rand(2, 4, 16), (2, 2), torch.rand(2, 12), torch.rand(2, 3, 32, 32))
        with torch.no_grad():
            a, b = plain(*args), head(*args)
        for x, y in zip(a, b):  # pose outputs: unchanged
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(tuple(head.last_contact.shape), (2, 6))
        self.assertFalse(torch.equal(head.last_contact, plain.last_contact))  # the tokens reach the contact logits
        with torch.no_grad():
            tok = args[0].transpose(1, 2).reshape(2, -1, 2, 2)
            head.contact_from(torch.zeros(2, 6, 2 * 8 + 4), args[2], torch.rand(2, 8, 8, 8), tok)

    def test_branch_reads_the_encoder_hidden_state_and_leaves_pose_unchanged(self):
        from fah.models.heads.keypoint_head import attach_contact_branch
        from fah.student import StudentEncoder

        torch.manual_seed(0)
        enc = StudentEncoder("dinov3_vits16", out_dim=16, pretrained=False, size=64).eval()  # 4 x 4 tokens
        kw = dict(img_dim=16, tok_dim=12, dim=8, stem=4, pool=True, dense=(2, 6), vis=True, contact=1, contact_dim=64)
        head, plain = HandKeypointHead(**kw).eval(), HandKeypointHead(**kw).eval()
        plain.load_state_dict(head.state_dict())
        attach_contact_branch(enc, head, 2)
        self.assertEqual((head.contact_branch_block, len(head.contact_branch["blocks"])), (10, 2))
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            feat = enc(x)  # the pre-hook keeps block 10's input for the head
            hidden = head._br_in[0].clone()
            args = (feat.flatten(2).transpose(1, 2), (4, 4), torch.rand(2, 12), torch.rand(2, 3, 32, 32))
            a, b = plain(*args), head(*args)
            for u, v in zip(a, b):  # pose outputs: unchanged
                torch.testing.assert_close(u, v, rtol=0, atol=0)
            self.assertFalse(torch.equal(head.last_contact, plain.last_contact))  # the branch reaches the contact logits
            ref = head.last_contact.clone()
            head._br_in = (hidden, head.contact_branch_kw)  # the TensorRT path: the engine's hidden<N> + the kept kwargs
            head(*args)
        torch.testing.assert_close(head.last_contact, ref, rtol=0, atol=0)


class VideoContactModelTests(unittest.TestCase):
    def test_cpu_runs_eager_and_outputs_the_track_hand(self):
        from fah.contact_video import ContactVideo, VideoContact

        torch.manual_seed(0)
        net = ContactVideo(n_vert=6, dim=16, layers=1, heads=2, tok_dim=0)
        ck = dict(args=dict(T=4, dim=16, layers=1), n_vert=6, use_tok=False, tok_dim=0, model=net.state_dict(), thr_blend=0.5)
        vc = VideoContact(ck, np.eye(778, 6), device="cpu", cuda_graph=True)
        self.assertFalse(vc.cuda_graph)  # CUDA graphs on CUDA devices only
        h = dict(is_right=True, score=0.9, joints3d=np.random.rand(21, 3) + [0, 0, 0.5], contact_dense=np.full(6, 0.7), contact_score=0.8)
        weaker = dict(h, score=0.2)
        vc([h, weaker])
        self.assertEqual(h["contact_video"].shape, (778,))
        self.assertNotIn("contact_video", weaker)  # one track hand per side

    def test_the_flag_restarts_after_the_window_emptied_and_holds_over_short_gaps(self):
        from fah.contact_video import ContactVideo, VideoContact

        net = ContactVideo(n_vert=6, dim=16, layers=1, heads=2, tok_dim=0)
        ck = dict(args=dict(T=4, dim=16, layers=1), n_vert=6, use_tok=False, tok_dim=0, model=net.state_dict(), h_on=0.9, h_off=0.1)
        vc = VideoContact(ck, np.eye(778, 6), device="cpu")
        vc._forward = lambda x, tok: (x[..., 3:9] * 0, x[..., 2] * 0 + score[0])  # flag probability = score[0]
        hand = lambda: dict(is_right=True, score=0.9, joints3d=np.random.rand(21, 3) + [0, 0, 0.5], contact_dense=np.full(6, 0.7),
                            contact_score=0.8)
        score = [3.0]  # sigmoid(3) = 0.95 > h_on: on
        h = hand(); vc([h]); self.assertTrue(h["contact_flag_video"])
        score[0] = 0.0  # 0.5: between h_off and h_on, the state decides
        for _ in range(3):  # a gap shorter than the window: the flag holds
            vc([])
        h = hand(); vc([h]); self.assertTrue(h["contact_flag_video"])
        for _ in range(4):  # the whole window (T = 4) without the hand: a new track
            vc([])
        h = hand(); vc([h]); self.assertFalse(h["contact_flag_video"])


class PartHeadSkipTests(unittest.TestCase):
    def test_part_head_off_skips_its_compute_and_keeps_the_points(self):
        torch.manual_seed(1)
        head = HandKeypointHead(img_dim=16, tok_dim=12, dim=8, stem=4, pool=True, dense=(2, 6), vis=True, parts=17).eval()
        args = (torch.rand(2, 4, 16), (2, 2), torch.rand(2, 12), torch.rand(2, 3, 32, 32))
        with torch.no_grad():
            on = head(*args)
            self.assertEqual(tuple(head.last_part_logits.shape[:2]), (2, 17))
            head.parts_on = False  # part maps not requested (HandEstimator.parts): the part head is not run
            off = head(*args)
        self.assertIsNone(head.last_part_logits)
        for x, y in zip(on, off):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


class ContactArgsTests(unittest.TestCase):
    def test_contact_head_file_is_used_in_every_mode(self):
        w, s = weights_dir()
        path = os.path.join(w, "fah-s", "contact.pt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertNotIn("contact_ckpt", api.method_args("fah-s", "image", w, "cpu", s, contact=True))
        self.assertIn("contact.pt missing", str(caught[-1].message))
        open(path, "w").close()
        self.assertEqual(api.method_args("fah-s", "image", w, "cpu", s, contact=True)["contact_ckpt"], path)
        self.assertNotIn("contact_ckpt", api.method_args("fah-s", "image", w, "cpu", s))  # off by default
        self.assertEqual(api.method_args("fah-s", "video", w, "cpu", s, contact=True)["contact_ckpt"], path)

    def test_video_contact_model_in_video_modes_or_on_request(self):
        w, s = weights_dir()
        shutil.copytree(os.path.join(w, "fah-s"), os.path.join(w, "fah"))
        open(os.path.join(w, "fah", "contact.pt"), "w").close()
        video = os.path.join(w, "fah", "contact_video.pt")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertNotIn("contact_video", api.method_args("fah", "video", w, "cpu", s, contact=True))
        self.assertIn("contact_video.pt missing", str(caught[-1].message))
        open(video, "w").close()
        for mode in ("video", "world"):
            a = api.method_args("fah", mode, w, "cpu", s, contact=True)
            self.assertEqual((a["contact_video"], a["contact_ckpt"]), (video, os.path.join(w, "fah", "contact.pt")))
        self.assertNotIn("contact_video", api.method_args("fah", "image", w, "cpu", s, contact=True))  # images: independent
        self.assertEqual(api.method_args("fah", "image", w, "cpu", s, contact=True, contact_video=True)["contact_video"], video)
        self.assertNotIn("contact_video", api.method_args("fah", "video", w, "cpu", s, contact=True, contact_video=False))
        self.assertNotIn("contact_video", api.method_args("fah", "video", w, "cpu", s))  # contact off

    def test_json_holds_the_contact_outputs_on_request(self):
        h = dict(is_right=True, score=0.9, bbox=np.zeros(4), joints2d=np.zeros((21, 2)), joints3d=np.zeros((21, 3)),
                 contact=np.linspace(0, 1, 778).astype(np.float16), contact_dense=np.zeros(400, np.float16), contact_score=0.7,
                 contact_fingers=np.arange(6, dtype=np.float32) / 6)
        self.assertNotIn("contact", to_json([h])[0])
        d = to_json([h], contact=True)[0]
        self.assertEqual((len(d["contact"]), len(d["contact_fingers"]), d["contact_score"]), (778, 6, 0.7))
        np.testing.assert_allclose(d["contact"], h["contact"].astype(np.float32))
        self.assertNotIn("contact_video", d)
        h.update(contact_video=np.linspace(1, 0, 778).astype(np.float32), contact_score_video=0.8, contact_flag_video=True)
        d = to_json([h], contact=True)[0]
        self.assertEqual((len(d["contact_video"]), d["contact_score_video"], d["contact_flag_video"]), (778, 0.8, True))



class VideoContactInputTests(unittest.TestCase):
    def test_video_contact_sees_the_per_frame_hands_from_before_the_refiner(self):
        class Refiner:  # moves the joints and adds a gap-fill copy (no contact) of the hand
            def update(self, hands, t):
                for h in hands:
                    h["keypoints_3d_mano"] = h["keypoints_3d_mano"] + .1
                return hands + [{k: v for k, v in h.items() if not k.startswith("contact")} for h in hands[:1]]

        class Recorder:  # stands in for fah.contact_video.VideoContact
            seen = []

            def __call__(self, hands):
                self.seen.append([np.array(h["joints3d"]) for h in hands])
                for h in hands:
                    h.update(contact_video=np.full(778, .7, np.float32), contact_score_video=.8, contact_flag_video=True)

        m = FAHMethod.__new__(FAHMethod)
        m.verify, m.tracker, m.t, m.fps = True, Refiner(), 0, 30.
        m.mesh_fit = m.masks = m.det_masks = m.tip_extend = m.parts_in_mask = False
        m.vreg, m.contact_video, m._vc = None, "contact_video.pt", None
        m._verified = lambda image, K: []
        self.assertEqual(m(np.zeros((2, 2, 3), np.uint8), np.eye(3)), [])  # no hand yet: the model is not loaded
        self.assertIsNone(m._vc)
        m._vc = Recorder()
        h = hand_at(0)
        per_frame = h["keypoints_3d_mano"].copy()
        h.update(contact=np.zeros(778, np.float32), contact_dense=np.full(400, .3, np.float32), contact_score=.6,
                 contact_fingers=np.zeros(6, np.float32), token=np.ones(8, np.float32))
        m._verified = lambda image, K: [h]
        out = m(np.zeros((2, 2, 3), np.uint8), np.eye(3))
        np.testing.assert_array_equal(m._vc.seen[0][0], per_frame)  # the per-frame joints, not the refined ones
        np.testing.assert_allclose(out[0]["joints3d"], per_frame + .1)  # pose output: the refined joints
        self.assertEqual((out[0]["contact_score_video"], out[0]["contact_flag_video"]), (.8, True))
        self.assertNotIn("contact_video", out[1])  # gap fill: no observation, no contact
        self.assertNotIn("contact", out[1])


if __name__ == "__main__":
    unittest.main()
