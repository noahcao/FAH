"""The fah command: flags map to the Python API's arguments, and its JSON holds exactly the API's outputs (stub model:
no weights)."""
import contextlib
import inspect
import io
import json
import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from bench.seg.rle import encode
from fah import api, build_trt, cli, world
from test_coreml_paths import weights_dir


def fake_hands(h, w, mesh="mano", seed=0):
    rng = np.random.default_rng(seed)
    hands = []
    for k in range(2):
        roi = rng.integers(0, 17, (11, 13)).astype(np.uint8)
        mask = np.zeros((h, w), bool)
        mask[5:20, 8 + k:30] = True
        verts = rng.normal(size=(778, 3)).astype(np.float32) * 0.01 + np.float32([0, 0, 0.5])
        hand = dict(
            is_right=bool(k), score=float(0.2 + 0.7 * k), bbox=rng.random(4).astype(np.float32) * 40,
            joints2d=rng.random((21, 2)).astype(np.float32) * 60, joints3d=rng.normal(size=(21, 3)).astype(np.float32),
            verts3d=verts, verts3d_dense=verts + np.float32(1e-3), surface2d_head=rng.random((440, 2)).astype(np.float32),
            surface3d_head=rng.normal(size=(440, 3)).astype(np.float32), surface_visible=rng.random(440) > 0.3,
            parts_roi=(int(3 + k), int(4), roi), mask_det=encode(mask), token=rng.random(8).astype(np.float16))
        if mesh in ("mhr", "both"):
            hand.update(verts3d_mhr=rng.normal(size=(2445, 3)).astype(np.float32) * 0.01 + np.float32([0, 0, 0.5]),
                        verts3d_mhr_mano=verts + np.float32(2e-3), joints3d_mhr_fit=rng.normal(size=(21, 3)).astype(np.float32),
                        joints3d_mhr_rig=rng.normal(size=(21, 3)).astype(np.float32),
                        mhr_params=dict(hand_pose=rng.normal(size=27).astype(np.float32), hand_pca=rng.normal(size=54).astype(np.float32),
                                        scale=rng.normal(size=28).astype(np.float32), shape=rng.normal(size=45).astype(np.float32),
                                        global_rot=rng.normal(size=3).astype(np.float32), global_rotmat=np.eye(3, dtype=np.float32),
                                        cam_t=rng.normal(size=3).astype(np.float32), model_params=rng.normal(size=204).astype(np.float32),
                                        mirrored=not k))
        if mesh in ("none", "mhr"):  # as fah.FAH: no MANO mesh
            del hand["verts3d"]
        hands.append(hand)
    return hands


class StubFAH:
    calls = []

    def __init__(self, model, mode, **kwargs):
        self.model, self.mode, self.kwargs = model, mode, kwargs
        StubFAH.calls.append(dict(kwargs, model=model, mode=mode))
        self.faces = np.array([[0, 1, 2]])
        self.faces_mhr = np.array([[0, 1, 2], [2, 1, 3]])

    def __call__(self, img, K=None):
        return fake_hands(*img.shape[:2], mesh=self.kwargs.get("mesh", "mano"))


def run_cli(*flags):
    d = tempfile.mkdtemp()
    img = os.path.join(d, "hand.png")
    cv2.imwrite(img, np.full((48, 64, 3), 128, np.uint8))
    StubFAH.calls = []
    with patch("fah.api.FAH", StubFAH), contextlib.redirect_stdout(io.StringIO()):
        cli.main(["--img", img, "--output_dir", os.path.join(d, "out"), *flags])
    assert os.path.exists(os.path.join(d, "out", "hand.jpg"))
    with open(os.path.join(d, "out", "hand.json")) as f:
        return json.load(f), StubFAH.calls[0]


def parse_error(argv):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            cli.parse_args(argv)
    except SystemExit:
        return err.getvalue()
    raise AssertionError(f"{argv} parsed without an error")


class CliJsonTests(unittest.TestCase):
    def test_all_outputs_round_trip_exactly(self):
        out, call = run_cli("--dense", "--parts", "--masks")
        self.assertTrue(call["det_masks"])
        api = fake_hands(48, 64)
        self.assertEqual(len(out["hands"]), len(api))
        for j, a in zip(out["hands"], api):
            self.assertEqual(j["is_right"], a["is_right"])
            self.assertEqual(j["score"], a["score"])
            for k in ("bbox", "joints2d", "joints3d", "verts3d", "verts3d_dense", "surface2d_head", "surface3d_head", "surface_visible"):
                v = np.asarray(j[k], a[k].dtype)
                self.assertEqual(v.shape, a[k].shape, k)
                np.testing.assert_array_equal(v, a[k], err_msg=k)
            x0, y0, roi = cli.decode_parts(j["parts"])
            self.assertEqual((x0, y0), a["parts_roi"][:2])
            np.testing.assert_array_equal(roi, a["parts_roi"][2])
            self.assertEqual(j["mask_det"], a["mask_det"])

    def test_default_json_fields(self):
        out, call = run_cli()
        self.assertNotIn("det_masks", call)
        for j, a in zip(out["hands"], fake_hands(48, 64)):
            self.assertEqual(set(j), {"is_right", "score", "bbox", "joints2d", "joints3d", "verts3d"})
            np.testing.assert_array_equal(np.asarray(j["joints3d"], np.float32), a["joints3d"])
        self.assertEqual((out["model"], out["mode"], out["mesh"], out["tips"]), ("fah", "image", "mano", True))
        self.assertNotIn("preset", out)

    def test_mesh_none_writes_no_mesh(self):
        out, call = run_cli("--mesh", "none")
        self.assertEqual(call["mesh"], "none")
        self.assertTrue(all("verts3d" not in j for j in out["hands"]))

    def test_mhr_meshes_round_trip_exactly(self):
        mhr = ("verts3d_mhr", "verts3d_mhr_mano", "joints3d_mhr_fit", "joints3d_mhr_rig")
        for flag, mesh in (("mhr", "mhr"), ("hmr", "mhr"), ("both", "both")):
            out, call = run_cli("--mesh", flag)
            self.assertEqual((call["mesh"], out["mesh"]), (mesh, mesh))
            for j, a in zip(out["hands"], fake_hands(48, 64, mesh)):
                self.assertEqual("verts3d" in j, mesh == "both")
                for k in mhr:
                    np.testing.assert_array_equal(np.asarray(j[k], np.float32), a[k], err_msg=k)
                self.assertEqual(set(j["mhr_params"]), set(a["mhr_params"]))
                for k, v in a["mhr_params"].items():
                    if k == "mirrored":
                        self.assertIs(j["mhr_params"][k], v)
                    else:
                        np.testing.assert_array_equal(np.asarray(j["mhr_params"][k], np.float32), v, err_msg=k)
        out, _ = run_cli()
        self.assertTrue(all(k not in j for j in out["hands"] for k in mhr + ("mhr_params",)))

    def test_mesh_both_draws_mano_and_mhr_panels(self):
        img = np.full((48, 64, 3), 128, np.uint8)
        hands = fake_hands(48, 64, "both")
        for mesh, width in (("mano", 64), ("none", 64), ("mhr", 64), ("both", 128)):
            args = cli.parse_args(["--img", "x.jpg", "--mesh", mesh])
            self.assertEqual(cli.overlay(img, hands, np.eye(3), np.array([[0, 1, 2]]), args, np.array([[0, 1, 2]])).shape, (48, width, 3))

    def test_min_score_filters_the_overlay_only(self):
        args = cli.parse_args(["--img", "x.jpg", "--min_score", "2"])
        img = np.full((48, 64, 3), 128, np.uint8)
        np.testing.assert_array_equal(cli.overlay(img, fake_hands(48, 64), np.eye(3), np.array([[0, 1, 2]]), args), img)
        out, _ = run_cli("--min_score", "2")
        self.assertEqual(len(out["hands"]), 2)

    def test_part_overlay_paints_only_part_pixels(self):
        img = np.zeros((48, 64, 3), np.uint8)
        hands = fake_hands(48, 64)
        out = cli.draw_parts(img, hands)
        x0, y0, roi = hands[0]["parts_roi"]
        painted = out.any(-1)
        self.assertFalse(painted[:y0].any())
        self.assertTrue(painted[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]][roi > 1].all())


class CliFlagTests(unittest.TestCase):
    def kwargs(self, *argv):
        args = cli.parse_args(list(argv))
        return args, cli.model_kwargs(args)

    def test_backbones_and_modes(self):
        for b, model in (("s", "fah-s"), ("b", "fah-b"), ("l", "fah"), ("h", "fah-h")):
            self.assertEqual(cli.parse_args(["--img", "x.jpg", "--backbone", b]).model, model)
        self.assertEqual(cli.parse_args(["--img", "x.jpg"]).model, "fah")
        self.assertEqual(cli.parse_args(["--img", "x.jpg"]).mode, "image")
        self.assertEqual(cli.parse_args(["--vid", "x.mp4"]).mode, "video")
        self.assertEqual(cli.parse_args(["--vid", "x.mp4", "--mode", "world"]).mode, "world")
        self.assertEqual(cli.parse_args(["--vid", "x.mp4", "--mode", "image"]).mode, "image")

    def test_model_arguments(self):
        _, kw = self.kwargs("--img", "x.jpg", "--weights", "/w", "--sam3d", "/s", "--device", "cpu", "--trt", "--no_tips",
                            "--mesh", "none", "--masks")
        self.assertEqual(kw, dict(weights="/w", sam3d="/s", device="cpu", trt_dir=os.path.join("/w", "trt"), mano_dir=None,
                                  mesh="none", tips=False, det_masks=True, parts=False))
        self.assertTrue(self.kwargs("--img", "x.jpg", "--parts")[1]["parts"])  # part maps only when requested
        _, kw = self.kwargs("--img", "x.jpg")
        self.assertEqual((kw["trt_dir"], kw["mesh"], kw["tips"], kw["device"]), ("", "mano", True, "auto"))
        self.assertNotIn("dtype", kw)
        self.assertEqual(self.kwargs("--img", "x.jpg", "--device", "mps", "--dtype", "float32")[1]["dtype"], "float32")

    def test_empty_frame_overlay_does_not_load_mesh_assets(self):
        from fah.cli import overlay_for_model

        class NoMeshAssets:
            @property
            def faces(self):
                raise AssertionError("MANO should not be loaded for an empty frame")

            @property
            def faces_mhr(self):
                raise AssertionError("MHR should not be loaded for an empty frame")

        from types import SimpleNamespace
        image = np.zeros((8, 8, 3), np.uint8)
        for mesh in ("none", "mano", "mhr", "both"):
            args = SimpleNamespace(mesh=mesh, min_score=0.3, masks=False, parts=False, dense=False, contact=False)
            out = overlay_for_model(image, [], np.eye(3), NoMeshAssets(), args)
            self.assertEqual(out.shape, (8, 16 if mesh == "both" else 8, 3))

    def test_device_presets(self):
        from fah.api import EXECUTION

        self.assertEqual(EXECUTION["mps"]["dtype"], "float32")  # avoid bf16 Metal matmul aborts and fp16 overflow
        self.assertEqual(EXECUTION["cuda"]["dtype"], "bfloat16")
        self.assertEqual(EXECUTION["cpu"]["dtype"], "float32")

    def test_mesh_presets(self):
        w, s = weights_dir()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            a = api.method_args("fah-s", "image", w, "cpu", s, mesh="mhr")
        self.assertEqual((a["mesh_fit"], a["mhr_fit"], a.get("mhr_fit_init")), (0, 2, None))  # no initialiser: closed-form init, 2 steps
        self.assertIn("mhr_fitter.pt missing", str(caught[-1].message))
        init = api.mhr_fitter(w)
        open(init, "w").close()
        a = {m: api.method_args("fah-s", "image", w, "cpu", s, mesh=m) for m in ("none", "mano", "mhr", "hmr", "both")}
        self.assertEqual({m: (x["mesh_fit"], x.get("mhr_fit", 0), x.get("mhr_fit_init")) for m, x in a.items()},
                         dict(none=(0, 0, None), mano=(1, 0, None), mhr=(0, 1, init), hmr=(0, 1, init), both=(1, 1, init)))
        self.assertEqual(cli.parse_args(["--img", "x.jpg", "--mesh", "hmr"]).mesh, "mhr")

    def test_configuration(self):
        w, s = weights_dir()
        a = lambda mode: api.method_args("fah-s", mode, w, "cpu", s)
        for mode in ("image", "video", "world"):  # the one detector: shared/detector.pt (the weights' config.json)
            self.assertEqual(a(mode)["detector_weights"], os.path.join(w, "shared", "detector.pt"))
        self.assertEqual(api.detector_weights(w), os.path.join(w, "shared", "detector.pt"))
        self.assertEqual(a("image")["pair_wrist"], 0.5)
        self.assertEqual((a("image")["mesh_fit_centre"], a("image")["mesh_fit_fd_eps"]), (True, 1e-2))
        self.assertNotIn("vd_propose", a("image"))
        video = a("video")
        self.assertEqual((video["pair_wrist"], video["vd_propose"], video["vd_gate_fill"], video["vd_min_inside"], video["vd_min_hits"]),
                         (0.5, True, True, 0.6, 10))
        self.assertEqual(a("world")["vd_propose"], True)
        self.assertEqual(api.method_args("fah-s", "video", w, "cpu", s, pair_wrist=0.0)["pair_wrist"], 0.0)  # overrides win
        self.assertEqual((a("image")["parts"], api.method_args("fah-s", "image", w, "cpu", s, parts=True)["parts"]),
                         (False, True))  # part maps on request (the quoted speeds are without them)
        self.assertNotIn("preset", inspect.signature(api.method_args).parameters)

    def test_world_placement_includes_the_mhr_mesh(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        h = dict(joints3d=np.zeros((21, 3)), verts3d_mhr=np.ones((2445, 3)))
        world.place([h], T)
        np.testing.assert_allclose(h["verts3d_mhr_world"], np.ones((2445, 3)) + [1.0, 2.0, 3.0])
        self.assertNotIn("verts3d_world", h)

    def test_invalid_combinations(self):
        self.assertIn("not allowed with", parse_error(["--img", "x.jpg", "--vid", "y.mp4"]))
        self.assertIn("one of the arguments", parse_error([]))
        self.assertIn("need --vid", parse_error(["--img", "x.jpg", "--mode", "video"]))
        self.assertIn("--cameras needs", parse_error(["--vid", "x.mp4", "--cameras", "c.npz"]))
        self.assertIn("invalid choice", parse_error(["--img", "x.jpg", "--backbone", "x"]))

    def test_build_trt_resolves_relative_paths(self):
        seen = []
        with patch.object(build_trt, "commands", lambda a: seen.append(a) or []), contextlib.redirect_stdout(io.StringIO()):
            build_trt.main(["--weights", "w", "--sam3d", "s", "--calib_images", "c", "--work", "k"])
        self.assertEqual((seen[0].weights, seen[0].sam3d, seen[0].calib_images, seen[0].work, seen[0].out),
                         tuple(os.path.abspath(x) for x in ("w", "s", "c", "k")) + ("",))

    def test_build_trt_commands(self):
        w = tempfile.mkdtemp()
        args = SimpleNamespace(backbone="s", weights=w, sam3d="/s", calib_images="/c", out="", work="/tmp/wk", trt_python="py", force=False)
        cmds = build_trt.commands(args)
        self.assertEqual([os.path.basename(c[1]) for c in cmds], ["build_pose_engine.py", "build_pose_engine.py", "build_det_engine.py"])
        self.assertEqual(cmds[0][2], "export")
        self.assertIn(os.path.join(w, "fah-s", "model.pt"), cmds[0])
        self.assertEqual(cmds[1][0], "py")
        self.assertEqual(cmds[1][-1], os.path.join(w, "trt", "fah-s-pose-fp16.engine"))
        self.assertEqual(cmds[2][-1], os.path.join(w, "trt", "detector-640-fp16.engine"))
        self.assertEqual(cmds[2][cmds[2].index("--weights") + 1], os.path.join(w, "shared", "detector.pt"))
        self.assertEqual(cmds[0][cmds[0].index("--detector") + 1], os.path.join(w, "shared", "detector.pt"))
        self.assertNotIn("--contact_ckpt", cmds[0])
        os.makedirs(os.path.join(w, "fah-s"))
        open(os.path.join(w, "fah-s", "contact.pt"), "w").close()  # a contact head: the export gets it (a branch -> 2 outputs)
        self.assertEqual(build_trt.commands(args)[0][-2:], ["--contact_ckpt", os.path.join(w, "fah-s", "contact.pt")])
        args.no_contact_branch = True
        self.assertNotIn("--contact_ckpt", build_trt.commands(args)[0])


if __name__ == "__main__":
    unittest.main()
