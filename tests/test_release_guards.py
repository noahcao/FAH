"""Release training safeguards; no models, datasets or ML dependencies required."""
import json
import unittest
from pathlib import Path

from train.release_guards import egodex_camera_holdout, egodex_episode_key, finetuned_teacher_state


class TeacherTests(unittest.TestCase):
    def test_finetuned_backbone_and_heads_replace_base_tensors(self):
        base = {"backbone.block.weight": "base", "head.weight": "base", "other": "base"}
        checkpoint = {"student_name": None, "trainable": {
            "model.backbone.block.weight": "fine-tuned", "model.head.weight": "head", "encoder.unused": "unused"}}
        base.update(finetuned_teacher_state(checkpoint, base))
        self.assertEqual(base, {"backbone.block.weight": "fine-tuned", "head.weight": "head", "other": "base"})

    def test_rejects_student_head_only_and_unknown_tensors(self):
        for checkpoint in [
            {"student_name": "vit", "trainable": {"model.backbone.weight": 1}},
            {"trainable": {"model.head.weight": 1}},
            {"trainable": {"model.backbone.unknown": 1}},
        ]:
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                finetuned_teacher_state(checkpoint, {"backbone.weight"})


class CameraSplitTests(unittest.TestCase):
    def test_all_scored_episodes_are_excluded_in_source_and_cache_forms(self):
        split = Path(__file__).resolve().parents[1] / "bench/splits/egodex_world_p2.json"
        for episode in json.loads(split.read_text())["episodes"]:
            short = episode.removeprefix("part2/")
            for name in [episode, "/data/" + episode + ".hdf5", short.replace("/", "__")]:
                with self.subTest(name=name):
                    self.assertTrue(egodex_camera_holdout(name))
                    self.assertEqual(egodex_episode_key(name), episode)

    def test_reserves_unscored_candidates_without_excluding_every_episode(self):
        import zlib

        for i in range(200):
            key = f"part2/synthetic_task/{i}"
            self.assertEqual(egodex_camera_holdout(key), zlib.crc32(key.encode()) % 1000 < 50)


if __name__ == "__main__":
    unittest.main()
