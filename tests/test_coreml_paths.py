"""Apple-silicon paths without a Mac: Core ML batch chunking, preset selection and fallbacks, the batched finite-difference
Jacobian, the build / live command lines. The Core ML parity itself runs only on macOS with exports present."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fah import api, build_coreml, live
from fah.coreml import CoreMLNet, CoreMLPose, compiled_path, fit_letterbox


def weights_dir():
    w, s = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(os.path.join(w, "fah-s"))
    os.makedirs(os.path.join(w, "shared"))
    cfg = dict(model="model.pt", detector="../shared/detector.pt", mesh_fitter="../shared/mesh_fitter.pt",
               video_refiner="../shared/video_refiner.pt", image=dict(det_size=640, mesh_fit=1, mesh_fit_jac="fd", tip_extend=0.6),
               video=dict(refiner_euro=[12, 1], refiner_keep_2d=True, side_memory=0.8, refiner_euro_2d=[30, 0.05]))
    json.dump(cfg, open(os.path.join(w, "fah-s", "config.json"), "w"))
    os.makedirs(os.path.join(s, "assets"))
    for f in ("model_config.yaml", "assets/mhr_model.pt"):
        open(os.path.join(s, f), "w").close()
    return w, s


class CoreMLChunkTests(unittest.TestCase):
    def test_batches_run_in_exported_sizes_with_padding(self):
        calls = []
        net = CoreMLPose.__new__(CoreMLPose)
        net.sizes, net.heads = [2, 4], True

        def run(name, **inputs):
            calls.append((name, len(inputs["img"])))
            return {"pt_uv": inputs["img"][:, :1] * 2}

        net.run = run
        x = np.arange(7, dtype=np.float32).reshape(7, 1)
        out = net(img=x)["pt_uv"]
        np.testing.assert_array_equal(out, x * 2)
        self.assertEqual(calls, [("b4", 4), ("b2", 2), ("b2", 2)])  # the last single crop is zero-padded to the smallest size

    def test_nonfinite_pose_is_reported_instead_of_returned(self):
        net = CoreMLPose.__new__(CoreMLPose)
        net.sizes = [1]
        net.run = lambda name, **inputs: {"pt_uv": np.full((1, 21, 2), np.nan, np.float32)}
        with self.assertRaisesRegex(FloatingPointError, "pose_precision float32"):
            net(img=np.zeros((1, 3, 8, 8), np.float32))

    def test_every_batch_function_is_warmed_up_front(self):
        ran = []

        def init(self, path, units="ALL"):
            self.functions, self.outputs = ["b1", "b4", "b2", "b3"], ["pt_uv"]
            self.shapes = {f"b{b}": {"img": (b, 3, 8, 8)} for b in (1, 2, 3, 4)}

        with patch.object(CoreMLNet, "__init__", init), \
                patch.object(CoreMLNet, "run", lambda self, name, **x: ran.append((name, x["img"].shape, float(x["img"].sum())))):
            net = CoreMLPose("pose.mlpackage")
        self.assertEqual(net.sizes, [1, 2, 3, 4])  # no compile / first-predict stall at a first 3-4 crop pass
        self.assertEqual(ran, [(f"b{b}", (b, 3, 8, 8), 0.0) for b in (1, 2, 3, 4)])


class LetterboxFitTests(unittest.TestCase):
    FUNCS = ["s384x640", "s448x640", "s480x640", "s640x640"]

    def test_a_built_shape_is_unchanged(self):
        img = np.zeros((384, 640, 3), np.uint8)
        out, px, py = fit_letterbox(img, 0, 12, self.FUNCS)
        self.assertIs(out, img)
        self.assertEqual((px, py), (0, 12))

    def test_a_missing_shape_is_padded_to_the_nearest_larger_one(self):
        img = np.full((352, 640, 3), 7, np.uint8)  # a 1134x612 frame (aspect 1.85) letterboxed with auto padding
        out, px, py = fit_letterbox(img, 0, 3, self.FUNCS)
        self.assertEqual(out.shape, (384, 640, 3))
        self.assertEqual((px, py), (0, 3 + 16))
        self.assertTrue((out[16:368] == 7).all() and (out[:16] == 114).all() and (out[368:] == 114).all())

    def test_no_larger_shape(self):
        self.assertIsNone(fit_letterbox(np.zeros((640, 672, 3), np.uint8), 0, 0, self.FUNCS))


class CompileOnceTests(unittest.TestCase):
    def test_compiled_model_is_kept_next_to_the_package_and_redone_after_a_rebuild(self):
        compiled = []

        def compile_model(src, out):
            compiled.append(src)
            os.makedirs(out)

        ct = SimpleNamespace(utils=SimpleNamespace(compile_model=compile_model))
        d = tempfile.mkdtemp()
        pkg = os.path.join(d, "fah-s-pose.mlpackage")
        os.makedirs(os.path.join(pkg, "Data"))
        weight = os.path.join(pkg, "Data", "weight.bin")
        open(weight, "w").close()
        os.utime(weight, (1e9, 1e9))
        with patch.dict(sys.modules, coremltools=ct):
            out = compiled_path(pkg)
            self.assertEqual((out, len(compiled)), (os.path.join(d, "fah-s-pose.mlmodelc"), 1))
            self.assertEqual(compiled_path(pkg), out)  # later launches: the same compiled model, no compile
            self.assertEqual(len(compiled), 1)
            os.utime(weight, None)  # the package was rebuilt after the compile
            os.utime(out, (1e9 - 10, 1e9 - 10))
            self.assertEqual((compiled_path(pkg), len(compiled)), (out, 2))

            def read_only(src, out):
                raise PermissionError(out)

            ct.utils.compile_model = read_only
            pkg2 = os.path.join(d, "other.mlpackage")
            os.makedirs(pkg2)
            self.assertIsNone(compiled_path(pkg2))  # read-only: load the package itself


class CoreMLPresetTests(unittest.TestCase):
    def setUp(self):
        self.w, self.s = weights_dir()

    def args(self, device, **kw):
        return api.method_args("fah-s", "image", self.w, device, self.s, **kw)

    def test_cuda_defaults_have_no_apple_settings(self):
        a = self.args("cuda")
        for k in ("pose_coreml", "det_coreml", "mesh_fit_device"):
            self.assertNotIn(k, a)
        self.assertEqual(a["mesh_fit_jac"], "fd")

    def test_mps_preset(self):
        a = self.args("mps")
        self.assertEqual((a["dtype"], a["mesh_fit_device"], a["mesh_fit_jac"]), ("float32", "cpu", "fdb"))
        self.assertNotIn("pose_coreml", a)
        self.assertNotIn("refiner_device", a)  # image mode: no refiner

    def test_video_refiner_on_the_cpu_on_apple_silicon_only(self):
        video = lambda device: api.method_args("fah-s", "video", self.w, device, self.s)
        self.assertEqual(video("mps")["refiner_device"], "cpu")  # 0.8 vs 2.4 ms per hand on MPS
        self.assertNotIn("refiner_device", video("cuda"))

    def test_coreml_falls_back_with_a_warning(self):
        with patch.object(sys, "platform", "linux"), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertNotIn("pose_coreml", self.args("mps", coreml=True))
        self.assertIn("macOS only", str(w[-1].message))
        with patch.object(sys, "platform", "darwin"), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertNotIn("pose_coreml", self.args("mps", coreml=True))
        self.assertIn("fah-build-coreml", str(w[-1].message))

    def test_coreml_exports_are_used_when_present(self):
        pose, det = api.coreml_paths("fah-s", self.w)
        for p in (pose, det):
            os.makedirs(p)
        with patch.object(sys, "platform", "darwin"):
            a = self.args("mps", coreml=True)
        self.assertEqual((a["pose_coreml"], a["det_coreml"], a["mesh_fit_device"], a["mesh_fit_jac"]), (pose, det, "cpu", "fdb"))
        self.assertEqual((a["device"], a["fast_batch"], a["det_half"]), ("cpu", True, True))  # PyTorch on the CPU, MPS host paths

    def test_part_free_pose_export_while_the_part_maps_are_off(self):
        pose, det = api.coreml_paths("fah-s", self.w)
        for p in (pose, det):
            os.makedirs(p)
        with patch.object(sys, "platform", "darwin"):
            self.assertEqual(self.args("mps", coreml=True)["pose_coreml"], pose)  # not built: the full export
            noparts = api.coreml_noparts(pose)
            os.makedirs(noparts)
            self.assertEqual(os.path.basename(noparts), "fah-s-pose-noparts.mlpackage")
            self.assertEqual(self.args("mps", coreml=True)["pose_coreml"], noparts)  # parts off (the default)
            self.assertEqual(self.args("mps", coreml=True, parts=True)["pose_coreml"], pose)
        self.assertEqual(api.coreml_paths("fah-s", self.w, parts=False), (noparts, det))
        self.assertEqual(api.coreml_paths("fah-s", self.w), (pose, det))

    def test_build_commands(self):
        args = SimpleNamespace(backbone="s", weights=self.w, sam3d=self.s, frames="1920x1080,1440x1080,1620x1080,1080x1080",
                               batch=[1, 2, 3, 4], parity_frames="/frames", mesh="mano", focal=610.0, force=False)
        cmds = build_coreml.commands(args)
        self.assertEqual([c[2] for c in cmds], ["pose", "pose", "det", "parity"])
        pose, det = api.coreml_paths("fah-s", self.w)
        self.assertEqual((cmds[0][-1], cmds[1][-1]), (pose, api.coreml_noparts(pose)))  # with / without the part head
        self.assertEqual(("--no_parts" in cmds[0], "--no_parts" in cmds[1]), (False, True))
        self.assertIn("--heads", cmds[0])
        i = cmds[0].index("--batch")
        self.assertEqual(cmds[0][i + 1:i + 5], ["1", "2", "3", "4"])  # crossed hands: 3-4 crops per pass
        self.assertEqual(cmds[2][cmds[2].index("--frames") + 1], "1920x1080,1440x1080,1620x1080,1080x1080")
        base = json.loads(cmds[3][cmds[3].index("--base_args") + 1])
        test = json.loads(cmds[3][cmds[3].index("--args") + 1])
        self.assertEqual((base["dtype"], base["det_half"], base["mesh_fit_device"]), ("float32", False, None))
        self.assertEqual((test["pose_coreml"], test["det_coreml"]), (api.coreml_noparts(pose), det))  # the default runtime
        self.assertEqual((os.path.basename(det), base["detector_weights"]), ("detector-640.mlpackage",
                                                                          os.path.join(self.w, "shared", "detector.pt")))
        self.assertEqual((cmds[2][-1], cmds[2][cmds[2].index("--weights") + 1]), (os.path.join(self.w, "coreml", "detector-640.mlpackage"),
                                                                                 os.path.join(self.w, "shared", "detector.pt")))

    def test_live_real_time_preset(self):
        parse = lambda *x: live.parser().parse_args(["--weights", self.w, "--sam3d", self.s, *x])
        path = os.path.join(self.w, "shared", "video_refiner.pt")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a, coreml, refiner = live.preset(parse("--device", "cpu"))
        self.assertFalse(coreml)
        self.assertEqual((a["mesh_fit"], a["tip_extend"], a["device"]), (0, 0.6, "cpu"))
        self.assertEqual((a["refiner"], refiner), (None, None))  # no refiner checkpoint: per-frame image mode, with a warning
        self.assertIn("per-frame image mode", str(w[-1].message))
        open(path, "w").close()
        args = parse("--device", "mps")
        a, _, refiner = live.preset(args)
        self.assertEqual((args.backbone, args.video_mode), ("s", 1))  # default: FAH-S, online video mode
        self.assertEqual((a["refiner"], refiner, a["refiner_device"], a["side_memory"], a["refiner_euro"]), (path, path, "cpu", 0.8, [12, 1]))
        self.assertEqual((a["pair_wrist"], a["vd_propose"], a["mesh_fit_centre"]), (0.5, True, True))  # as fah --mode video
        self.assertEqual(a["detector_weights"], os.path.join(self.w, "shared", "detector.pt"))
        a, _, refiner = live.preset(parse("--device", "mps", "--video_mode", "0"))
        self.assertEqual((a["refiner"], refiner), (None, path))  # key v switches it on

    def test_build_defaults(self):
        with patch.object(sys, "platform", "darwin"), patch.object(build_coreml, "commands", return_value=[]) as c, \
                contextlib.redirect_stdout(io.StringIO()):
            build_coreml.main(["--weights", self.w, "--sam3d", self.s])
        args = c.call_args[0][0]
        self.assertEqual((args.batch, args.frames), ([1, 2, 3, 4], build_coreml.DEFAULT_FRAMES))

    def test_default_letterbox_shapes(self):
        from scripts.build_coreml import letterbox_shape

        shapes = {letterbox_shape(*map(int, f.split("x")[::-1]), 640) for f in build_coreml.DEFAULT_FRAMES.split(",")}
        for w, h in ((1134, 612), (2532, 1170), (1920, 1080), (1080, 1920), (1170, 2532), (1280, 960), (960, 1280)):
            self.assertIn(letterbox_shape(h, w, 640), shapes, (w, h))  # e.g. the 1134 x 612 user video, iPhone landscape / portrait
        self.assertEqual(len(shapes), 11)


@unittest.skipUnless(os.path.exists(os.path.join(os.environ.get("MANO_DIR", "data/mano"), "MANO_RIGHT.pkl")), "needs MANO")
class BatchedJacobianTests(unittest.TestCase):
    def test_fdb_matches_fd(self):
        from fah.fast_mano import FastMano, MeshFitter

        torch.manual_seed(0)
        with torch.no_grad():  # a MANO hand (random articulation and shape) as the target surface + joints
            verts, joints = FastMano(device="cpu")(torch.randn(2, 45) * 0.3, torch.randn(2, 10) * 0.5)
        tV, tJ = (verts + torch.tensor([0, 0, 0.5])).numpy(), (joints + torch.tensor([0, 0, 0.5])).numpy()
        out = {jac: MeshFitter(device="cpu", iters=2, solver="gn", jac=jac, graph=False, compile=False).fit(
            tV, tJ, np.tile(np.eye(3), (2, 1, 1)), np.zeros((2, 45))) for jac in ("fd", "fdb")}
        np.testing.assert_allclose(out["fdb"], out["fd"], atol=5e-5)  # 0.05 mm (the same Gauss-Newton steps; fp32 differences)


@unittest.skipUnless(sys.platform == "darwin" and all(os.path.exists(p) for p in api.coreml_paths("fah-s")) and
                     os.environ.get("FAH_TEST_IMAGE"), "macOS with fah-s Core ML exports and $FAH_TEST_IMAGE")
class CoreMLParityTests(unittest.TestCase):
    def test_coreml_matches_pytorch_fp32(self):
        import cv2

        img = cv2.imread(os.environ["FAH_TEST_IMAGE"])
        ref = api.FAH("fah-s", "image", device="mps", dtype="float32", det_half=False, mesh="none")(img)
        got = api.FAH("fah-s", "image", device="mps", coreml=True, mesh="none")(img)
        self.assertEqual(len(got), len(ref))
        for a, b in zip(sorted(ref, key=lambda h: h["bbox"][0]), sorted(got, key=lambda h: h["bbox"][0])):
            self.assertLess(np.abs(np.asarray(a["joints3d"]) - np.asarray(b["joints3d"])).max(), 0.025)  # 25 mm worst joint (parity max)


if __name__ == "__main__":
    unittest.main()
