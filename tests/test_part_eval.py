"""Small CPU contracts of the part evaluations (bench.parts; scripts/part_seg_eval.py, scripts/dense_part_eval.py):
held-out selection and split files, part-head scoring, dense-point metrics / bootstrap, prediction caches."""
import copy
import json
import os
import pickle
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bench.parts import heldout, seg
from bench.parts.dense import bootstrap, mean_defined, metrics, score_hand, stack
from bench.seg.rle import encode
from scripts import dense_part_eval as dense_cli

REPO = Path(__file__).resolve().parents[1]
HELD = next(str(i) for i in range(10000) if zlib.crc32(str(i).encode()) % heldout.VAL_MOD == 0)
TRAIN = next(str(i) for i in range(10000) if zlib.crc32(str(i).encode()) % heldout.VAL_MOD != 0)


def make_pool(root, records, routes):
    (root / "pool").mkdir()
    (root / "pool/index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (root / "pool/masks_parts_index.0.jsonl").write_text("".join(json.dumps(dict(file=f, route=rt)) + "\n" for f, rt in routes.items()))


class HeldoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        recs = [dict(file="a.jpg", src=f"{HELD}:0"), dict(file="b.jpg", src=f"{HELD}:1"), dict(file="c.jpg", src=f"{TRAIN}:0"),
                dict(file="d.jpg", src=f"{HELD}:2")]
        make_pool(self.root, recs, {"a.jpg": "mesh+mask", "b.jpg": "mesh+mask", "c.jpg": "mesh+mask", "d.jpg": "mesh"})

    def tearDown(self):
        self.tmp.cleanup()

    def split(self, files):
        return {"pools": {"pool": {"part_dir": "masks_parts", "files": files}}}

    def test_only_validation_crops_with_accepted_labels_are_candidates(self):
        self.assertEqual([r["file"] for r in heldout.candidates(self.tmp.name, "pool", "masks_parts")], ["a.jpg", "b.jpg"])

    def test_sampling_is_seeded_and_split_keeps_its_order(self):
        s1 = heldout.sample(self.tmp.name, ["pool"], {}, 2)
        s2 = heldout.sample(self.tmp.name, ["pool"], {}, 2)
        self.assertEqual([r["file"] for r in s1["pool"]], [r["file"] for r in s2["pool"]])
        sel = heldout.from_split(self.tmp.name, self.split(["b.jpg", "a.jpg"]))
        self.assertEqual([r["file"] for r in sel["pool"][2]], ["b.jpg", "a.jpg"])
        self.assertEqual(sel["pool"][:2], ("masks_parts", "masks"))

    def test_split_rejects_training_unlabelled_missing_and_duplicate_crops(self):
        for files in (["c.jpg"], ["d.jpg"], ["missing.jpg"]):
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, "listed crops"):
                heldout.from_split(self.tmp.name, self.split(files))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            heldout.from_split(self.tmp.name, self.split(["a.jpg", "a.jpg"]))
        with self.assertRaisesRegex(ValueError, "not in the split"):
            heldout.from_split(self.tmp.name, self.split(["a.jpg"]), pools=["other"])

    def test_written_split_round_trips(self):
        path = self.root / "split.json"
        heldout.write_split(path, heldout.sample(self.tmp.name, ["pool"], {}, 2), {}, {"pool": "sam/masks"}, version="t")
        sel = heldout.from_split(self.tmp.name, heldout.load_split(path))
        self.assertEqual(sel["pool"][1], "sam/masks")
        self.assertEqual(len(sel["pool"][2]), 2)


class SplitFileTests(unittest.TestCase):
    def test_paper_splits_are_well_formed(self):
        expect = {"parts_heldout.json": (5, 200), "dense_parts.json": (3, 300), "parts_hard.json": (1, 109)}
        for name, (n_pools, n) in expect.items():
            with self.subTest(name=name):
                split = heldout.load_split(REPO / "bench" / "splits" / name)
                self.assertEqual(len(split["pools"]), n_pools)
                for pool, e in split["pools"].items():
                    self.assertEqual(len(e["files"]), n if name != "parts_hard.json" else 109)
                    self.assertEqual(len(set(e["files"])), len(e["files"]), pool)
                    self.assertTrue(all(not f.startswith("/") for f in e["files"]))


class PartHeadScoreTests(unittest.TestCase):
    def label(self):
        lab = np.zeros((16, 16), np.uint8)
        lab[4:12, 4:8], lab[4:12, 8:12] = 1, 5  # palm, index proximal
        lab[0, 0] = 255
        return lab

    def test_perfect_box_and_detector_predictions(self):
        lab = self.label()
        roi = (2, 2, lab[2:14, 2:14].copy())
        roi[2][lab[2:14, 2:14] == 255] = 0
        sam = ((lab > 0) & (lab < 255)).astype(np.uint8) * 255
        pred = dict(box=roi, dets=[dict(mask_det=encode(sam > 127), parts_roi=roi)])
        acc = seg.new_acc()
        seg.score_crop(acc, lab, sam, pred)
        r = seg.report(acc)
        self.assertEqual(r["n"], 1)
        self.assertEqual(r["n_detected"], 1)
        self.assertAlmostEqual(r["iou_lab"], 1.0)
        self.assertAlmostEqual(r["detbox_union_iou_sam"], 1.0)
        self.assertAlmostEqual(r["det_iou_sam"], 1.0)
        self.assertAlmostEqual(r["pixel_acc"], 1.0)
        iou = r["detbox_iou_per_class"]
        self.assertEqual((iou[1], iou[5]), (1.0, 1.0))

    def test_missed_hand_counts_as_background_end_to_end(self):
        lab = self.label()
        sam = ((lab > 0) & (lab < 255)).astype(np.uint8) * 255
        acc = seg.new_acc()
        seg.score_crop(acc, lab, sam, dict(box=None, dets=[]))
        r = seg.report(acc)
        self.assertEqual((r["n_mask_evaluable"], r["n_detected"]), (1, 0))
        self.assertEqual(r["e2e_union_iou_sam"], 0.0)
        self.assertIsNone(r["det_iou_sam"])
        self.assertEqual(r["iou_lab"], 0.0)

    def test_no_sam_mask_skips_the_detector_metrics(self):
        acc = seg.new_acc()
        seg.score_crop(acc, self.label(), None, dict(box=None, dets=None))
        r = seg.report(acc)
        self.assertEqual((r["n"], r["n_mask_evaluable"]), (1, 0))
        json.dumps(r)


class DenseMetricTests(unittest.TestCase):
    def hand(self, part=4, visible=True, predicted_part=None, ignore=False):
        label = np.full((8, 8), part, np.uint8)
        if predicted_part is not None:
            label[-1, -1] = predicted_part  # its own part exists, but the point lands on another finger
        if ignore:
            label[3, 3] = 255
        return score_hand(label, np.array([[3., 3.]]), np.array([visible]), np.array([part if predicted_part is None else predicted_part]), 8.)

    def test_wrong_finger_and_strict_accuracy(self):
        right, wrong = metrics(stack([self.hand()])), metrics(stack([self.hand(predicted_part=7)]))
        self.assertEqual((right["wrong_finger"], right["part_acc_fingers"]), (0.0, 1.0))
        self.assertEqual((wrong["wrong_finger"], wrong["finger_acc"]), (1.0, 0.0))
        ign = metrics(stack([self.hand(ignore=True)]))
        self.assertIsNone(ign["part_acc_fingers"])  # the point lands on ignore: skipped ...
        self.assertEqual(ign["part_acc_fingers_strict"], 0.0)  # ... but counted as not inside in the strict variant

    def test_no_visible_finger_points_is_undefined_not_zero_error(self):
        for hand in (self.hand(visible=False), self.hand(part=1)):
            result = metrics(stack([hand]))
            self.assertIsNone(result["wrong_finger"])
            self.assertIsNone(result["finger_acc"])
            json.dumps(result, allow_nan=False)

    def test_empty_pool_has_no_invented_metrics(self):
        result = metrics(stack([]))
        self.assertEqual(result["n"], 0)
        for key in ("wrong_finger", "part_acc", "finger_acc", "cover", "sil_iou", "vis_frac"):
            self.assertIsNone(result[key], key)

    def test_paired_bootstrap(self):
        a = {"p": stack([self.hand(), self.hand(), self.hand(), self.hand()])}
        b = {"p": stack([self.hand(predicted_part=7), self.hand(predicted_part=7), self.hand(), self.hand()])}
        r = bootstrap(a, b, ["p"], B=64)["wrong_finger"]
        self.assertEqual(r["diff"], 0.5)
        clip = bootstrap(a, b, ["p"], B=64, clips={"p": ["x", "x", "y", "y"]})["wrong_finger"]
        self.assertTrue({clip["lo"], clip["hi"]} <= {0.0, 0.5, 1.0})  # whole-clip mixtures only
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            bootstrap(a, {"p": stack([])}, ["p"], B=4)
        self.assertEqual(mean_defined([None, .5, 1.]), .75)


class DenseCacheTests(unittest.TestCase):
    def args(self, tmp, **kw):
        return SimpleNamespace(method="fah_pipeline", method_args='{"input_size": 320}', crops=tmp, mano_dir=tmp, **kw)

    def test_options_geometry_and_checkpoint_change_the_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "model.pt"
            ck.write_bytes(b"test checkpoint")
            args = self.args(tmp)
            sel = {"pool": ("parts", [dict(file="a.jpg", box=[1, 2, 10, 12], hw=[32, 32], is_right=True, M=[[1, 0, 0], [0, 1, 0]])])}
            base = dense_cli.inference_signature(str(ck), sel, args)
            args.method_args = '{"input_size": 224}'
            self.assertNotEqual(dense_cli.inference_signature(str(ck), sel, args), base)
            args.method_args = '{"input_size": 320}'
            moved = copy.deepcopy(sel)
            moved["pool"][1][0]["box"][0] += 1
            self.assertNotEqual(dense_cli.inference_signature(str(ck), moved, args), base)
            st = ck.stat()
            os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
            self.assertNotEqual(dense_cli.inference_signature(str(ck), sel, args), base)
            relabelled = {"pool": ("parts_new", sel["pool"][1])}  # labels are not an inference input
            self.assertEqual(dense_cli.inference_signature(str(ck), relabelled, args), dense_cli.inference_signature(str(ck), sel, args))

    def test_foreign_cache_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "model.pt"
            ck.write_bytes(b"test checkpoint")
            sel = {"pool": ("parts", [dict(file="a.jpg", box=[1, 2, 10, 12])])}
            pickle.dump(dict(files={"pool": ["a.jpg"]}), open(Path(tmp) / "preds_arm.pkl", "wb"))
            args = self.args(tmp, out_dir=tmp, arms=f"arm={ck}", preds="")
            with patch.object(dense_cli, "infer") as infer, self.assertRaisesRegex(ValueError, "another model"):
                dense_cli.load_predictions(args, sel)
            infer.assert_not_called()

    def test_stored_predictions_must_cover_the_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            pf = Path(tmp) / "p.pkl"
            pickle.dump(dict(files={"pool": ["b.jpg"]}, pools={"pool": [None]}), open(pf, "wb"))
            args = self.args(tmp, out_dir=tmp, arms="", preds=f"x={pf}")
            with self.assertRaisesRegex(ValueError, "other crops"):
                dense_cli.load_predictions(args, {"pool": ("parts", [dict(file="a.jpg")])})


if __name__ == "__main__":
    unittest.main()
