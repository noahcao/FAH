"""Core ML reuse must follow actual inputs and artifacts, not just package filenames."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fah import build_coreml


class ExportCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(backbone="s", weights=str(self.root / "weights"),
                                    sam3d=str(self.root / "sam"), batch=[1, 2], frames="320x320",
                                    pose_precision="float32", parity_frames="", force=False)
        for relative, data in [("weights/fah-s/model.pt", b"pose-A"),
                               ("weights/shared/detector.pt", b"det-A"),
                               ("sam/model_config.yaml", b"config"), ("sam/assets/mhr_model.pt", b"rig")]:
            p = self.root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def complete_exports(self):
        for command in build_coreml.commands(self.args):
            out = command[command.index("--out") + 1]
            p = Path(out)
            p.mkdir(parents=True, exist_ok=True)
            (p / "Manifest.json").write_text("simulated exported package")
            build_coreml._write_manifest(out, self.args._build_identities[out])

    def test_identical_build_reuses_all_exports(self):
        self.complete_exports()
        self.assertEqual(build_coreml.commands(self.args), [])

    def test_precision_change_rebuilds_both_pose_packages_only(self):
        self.complete_exports()
        self.args.pose_precision = "float16"
        commands = build_coreml.commands(self.args)
        self.assertEqual([c[2] for c in commands], ["pose", "pose"])
        self.assertTrue(all(c[c.index("--precision") + 1] == "float16" for c in commands))

    def test_weight_content_change_is_detected_even_with_same_size_and_timestamp(self):
        self.complete_exports()
        p = self.root / "weights/fah-s/model.pt"
        stat = p.stat()
        p.write_bytes(b"pose-B")
        os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["pose", "pose"])

    def test_detector_weights_and_shapes_do_not_rebuild_pose(self):
        self.complete_exports()
        (self.root / "weights/shared/detector.pt").write_bytes(b"det-B")
        self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["det"])
        self.complete_exports()
        self.args.frames = "1920x1080"
        self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["det"])

    def test_batch_change_rebuilds_pose(self):
        self.complete_exports()
        self.args.batch = [1, 2, 3]
        self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["pose", "pose"])

    def test_upstream_asset_change_rebuilds_pose_only(self):
        self.complete_exports()
        (self.root / "sam/assets/mhr_model.pt").write_bytes(b"new rig")
        self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["pose", "pose"])

    def test_toolchain_change_rebuilds_all_exports(self):
        self.complete_exports()
        with patch.object(build_coreml.importlib.metadata, "version", return_value="new-version"):
            self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["pose", "pose", "det"])

    def test_exporter_code_change_rebuilds_all_exports(self):
        code = self.root / "code"
        for name in ("scripts/build_coreml.py", "bench/methods/fah.py", "fah/encoder.py"):
            p = code / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("original code")
        with patch.object(build_coreml, "REPO", str(code)):
            self.complete_exports()
            (code / "fah/encoder.py").write_text("updated code")
            self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["pose", "pose", "det"])

    def test_overwritten_package_is_not_blessed_by_an_old_manifest(self):
        self.complete_exports()
        pose, _ = build_coreml.coreml_paths("fah-s", self.args.weights)
        Path(pose, "Manifest.json").write_text("different export")
        commands = build_coreml.commands(self.args)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][-1], pose)

    def test_legacy_or_corrupt_manifests_trigger_rebuild(self):
        self.complete_exports()
        pose, det = build_coreml.coreml_paths("fah-s", self.args.weights)
        Path(build_coreml._manifest_path(pose)).unlink()
        Path(build_coreml._manifest_path(det)).write_text("invalid JSON")
        self.assertEqual([c[2] for c in build_coreml.commands(self.args)], ["pose", "det"])

    def test_failed_export_does_not_write_success_metadata(self):
        with patch.object(sys, "platform", "darwin"), patch.object(build_coreml.subprocess, "run",
                side_effect=subprocess.CalledProcessError(1, "export")):
            with self.assertRaises(subprocess.CalledProcessError):
                build_coreml.main(["--weights", self.args.weights, "--sam3d", self.args.sam3d])
        self.assertEqual(list(self.root.rglob("*.fah-build.json")), [])


if __name__ == "__main__":
    unittest.main()
