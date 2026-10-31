"""Recipe scripts: checkpoint flow, configuration, command boundaries and preview behaviour."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
from scripts import _recipes as shared


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load("fah_train_recipe", "train.py")
infer = load("fah_infer_recipe", "infer.py")


def settings():
    # Arbitrary test values, not a training allocation.
    return dict(data_root="/tmp/data with spaces", ckpt_root="/tmp/checkpoints", weights="/tmp/weights",
                train_batch_size=7, train_processes=3, train_devices="test-device", model="fah",
                pose_tracks="/tmp/pose.npz", contact_tracks="hot3d:/tmp/tracks", mesh_data="/tmp/mesh.npz",
                mhr_data="/tmp/mhr.npz", output_dir="/tmp/output with spaces", mano_dir="/tmp/mano")


def value(command, flag):
    return command[command.index(flag) + 1]


class RecipeTests(unittest.TestCase):
    def test_pose_is_one_run_resuming_the_selected_encoder_and_heads(self):
        commands = train.commands(settings(), "pose")
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(value(cmd, "--steps"), "30000")
        self.assertEqual(value(cmd, "--lr"), "0.0001")
        self.assertEqual(value(cmd, "--init_ckpt"), "/tmp/checkpoints/finetune/fah/heads/final.pt")
        self.assertEqual(value(cmd, "--student"), value(cmd, "--init_ckpt"))
        self.assertEqual(value(cmd, "--out"), "/tmp/checkpoints/finetune/fah/pose")
        self.assertEqual(value(cmd, "--crops"), "/tmp/data with spaces/crops")
        self.assertNotIn("--external_kd", cmd)

    def test_parts_resume_the_single_pose_checkpoint(self):
        cmd = train.commands(settings(), "parts")[0]
        expected = "/tmp/checkpoints/finetune/fah/pose/final.pt"
        self.assertEqual(value(cmd, "--student"), expected)
        self.assertEqual(value(cmd, "--init_ckpt"), expected)

    def test_head_initialisation_uses_the_selected_encoder_for_every_model(self):
        for model in shared.MODELS:
            with self.subTest(model=model):
                cmd = train.commands(settings(), "heads", model)[0]
                self.assertEqual(value(cmd, "--student"), f"/tmp/checkpoints/distill/{model}/best.pt")
                self.assertEqual(value(cmd, "--out"), f"/tmp/checkpoints/finetune/{model}/heads")
                self.assertEqual(value(cmd, "--student_size"), "448" if model == "fah-h" else "320")
                self.assertEqual("--grad_ckpt" in cmd, model == "fah-h")
                pose = train.commands(settings(), "pose", model)[0]
                self.assertEqual(value(pose, "--student"), value(cmd, "--out") + "/final.pt")
                self.assertEqual(value(pose, "--init_ckpt"), value(pose, "--student"))

    def test_h_model_uses_its_training_resolution(self):
        cmd = train.commands(settings(), "pose", "fah-h")[0]
        self.assertEqual(value(cmd, "--size"), "448")
        self.assertEqual(value(cmd, "--student_size"), "448")
        self.assertIn("--grad_ckpt", cmd)

    def test_training_requires_user_resource_choices(self):
        c = settings()
        c["train_batch_size"] = None
        with self.assertRaisesRegex(ValueError, "train_batch_size"):
            train.commands(c, "pose")
        c = settings()
        c["train_processes"] = None
        with self.assertRaisesRegex(ValueError, "train_processes"):
            train.commands(c, "pose")

    def test_inference_needs_no_training_settings_and_preserves_paths(self):
        c = dict(weights="/tmp/weights", output_dir="/tmp/output with spaces",
                 inference=dict(backend="coreml", parts=True, contact=True))
        cmd = infer.commands(c, "/tmp/image with spaces.jpg", model="fah-s")[0]
        self.assertEqual(value(cmd, "--img"), str(Path("/tmp/image with spaces.jpg").resolve()))
        self.assertEqual(value(cmd, "--backbone"), "s")
        self.assertIn("--coreml", cmd)
        self.assertIn("--parts", cmd)
        self.assertIn("--contact", cmd)

    def test_invalid_backend_and_world_image_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            infer.commands(dict(inference=dict(backend="unknown")), "image.jpg")
        with self.assertRaisesRegex(ValueError, "requires a video"):
            infer.commands({}, "image.jpg", mode="world")

    def test_preview_never_executes_and_failed_commands_stop_the_recipe(self):
        with patch.object(shared.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            shared.execute([["python", "first"], ["python", "second"]], settings(), dry_run=True)
            run.assert_not_called()
        failure = subprocess.CalledProcessError(1, ["python", "first"])
        with patch.object(shared.subprocess, "run", side_effect=failure) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                shared.execute([["python", "first"], ["python", "second"]], settings())
            self.assertEqual(run.call_count, 1)
            self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
