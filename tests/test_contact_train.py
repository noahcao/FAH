"""Contact training / evaluation code: CPU-only checks on toy data (no weights, no datasets)."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from bench import contact_ci, contact_eval as E
from train import contact_head as H
from train import contact_labels as L
from train import contact_video as V


class EvalMetricTests(unittest.TestCase):
    def test_ap_groups_tied_scores(self):
        # one precision / recall point per distinct score: scores (0.9, 0.5, 0.5, 0.1), labels (1, 1, 0, 0)
        # -> points (P 1, R .5) and (P 2/3, R 1): AP = .5 * 1 + .5 * 2/3
        self.assertAlmostEqual(E.ap_score([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0]), 0.5 + 0.5 * 2 / 3, places=9)
        self.assertEqual(E.ap_score([0.3, 0.2], [0, 0]), 0.0)

    def test_scores_pooled_and_per_hand(self):
        lab = np.zeros((2, 778), bool)
        lab[0, :10] = True
        prob = np.zeros((2, 778), np.float32)
        prob[0, :5] = 0.9  # 5 TP, 5 FN on hand 0
        prob[1, :5] = 0.9  # 5 FP on hand 1 (no GT contact)
        s = E.scores(prob, np.array([0.9, 0.2], np.float32), lab, 0.5)
        self.assertAlmostEqual(s["vF1"], 2 * 5 / (2 * 5 + 5 + 5))
        self.assertAlmostEqual(s["hF1"], 2 * 1.0 * 0.5 / 1.5)  # per-hand mean over hands with GT contact (hand 0 only)
        self.assertAlmostEqual(s["frameF1"], 1.0)

    def test_runs_and_switches(self):
        self.assertEqual(E.runs([3, 4, 5, 9, 10]), [slice(0, 3), slice(3, 5)])
        c = np.array([0] * 6 + [1] * 6, bool)
        self.assertEqual(E.switches(c, "on"), [6])
        self.assertEqual(E.switches(c, "off"), [])
        self.assertEqual(E.hysteresis(np.array([0.2, 0.7, 0.5, 0.3]), 0.6, 0.4).tolist(), [False, True, True, False])


class VideoTrainTests(unittest.TestCase):
    def test_release_flag_rule(self):
        f = np.array([0.95, 0.5, 0.5, 0.5, 0.5, 0.95])
        present = np.array([True, False, False, False, True, True])
        # off -> on at frame 0; frames without a hand never update the state; back after >= T = 3 missing frames: starts off
        self.assertEqual(V.hysteresis_track(f, present, 0.9, 0.1, 3).tolist(), [True, False, False, False, False, True])

    def test_model_and_windows(self):
        net = V.ContactVideo(n_vert=6, dim=16, layers=1, heads=2, tok_dim=0)
        x = torch.zeros(2, 4, 3 + 6 + 66)
        x[..., 0] = 1
        lg, fg = net(x)
        self.assertEqual(tuple(lg.shape), (2, 4, 6))
        self.assertEqual(tuple(fg.shape), (2, 4))
        self.assertTrue(torch.allclose(lg, x[..., 3:9]))  # zero-initialised residuals: starts at the per-frame logits
        tr = dict(x=np.zeros((10, 75), np.float32), lab=np.zeros((10, 6), np.float32), flag=np.zeros(10, np.float32),
                  has=np.ones(10, np.float32), pred_idx=np.zeros(10, np.int64))
        self.assertEqual(len(V.windows([tr], 4, (1,))), 4)  # starts 0, 2, 4, 6 (half-window stride)

    def test_folds_by_recording(self):
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "dexycb.json")
            clips = [f"20200820_135508_{c}.tar" for c in ("836212060125", "839512060362")] + ["20200903_112137_932122060861.tar"]
            json.dump(dict(clips=clips), open(sp, "w"))
            V.folds(SimpleNamespace(sets=f"dexycb:{sp}", clip_meta="", out=d))
            f0, f1 = (json.load(open(os.path.join(d, f"dexycb_train_f{k}.json")))["clips"] for k in (0, 1))
            self.assertEqual(sorted(f0 + f1), sorted(clips))
            for f in (f0, f1):  # every camera view of a recording in one fold
                recs = {c.rsplit("_", 1)[0] for c in f}
                self.assertFalse(recs & {c.rsplit("_", 1)[0] for c in (f1 if f is f0 else f0)})


class HeadAndLabelTests(unittest.TestCase):
    def test_fold_key_and_metrics(self):
        self.assertEqual(H.fold_key({"file": "a/b.jpg", "src": "clip-001:0005"}), "clip-001")
        self.assertEqual(H.fold_key({"file": "a/b.jpg", "src": "x:1", "sequence_id": "P0001"}, "sequence"), "P0001")
        lab = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        P, R, f1 = H.metrics(torch.tensor([[5.0, 5.0, -5.0], [-5.0, -5.0, -5.0]]), lab, torch.ones(2))
        self.assertAlmostEqual(f1[0].item(), 2 / 3, places=5)
        self.assertEqual(f1[1].item(), 1.0)  # no contact, none predicted

    def test_make_root_filters_hoi4d(self):
        with tempfile.TemporaryDirectory() as d:
            lab = os.path.join(d, "labels")
            for s in ("hot3d_lab_mesh", "hoi4d"):
                os.makedirs(os.path.join(lab, s))
            np.save(os.path.join(lab, "hoi4d", "sdf.npy"), np.zeros((3, 778), np.float16))
            rows = [dict(file="a.jpg", valid=True, min_mm=-3.0, obj="Mug"), dict(file="b.jpg", valid=True, min_mm=-20.0, obj="Mug"),
                    dict(file="c.jpg", valid=True, min_mm=1.0, obj="Chair")]
            with open(os.path.join(lab, "hoi4d", "contact.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            L.make_root(os.path.join(d, "root"), lab, sources="hot3d_lab_mesh")
            got = [json.loads(line) for line in open(os.path.join(d, "root", "hoi4d", "contact.jsonl"))]
            self.assertEqual([g["valid"] for g in got], [True, False, False])
            self.assertTrue(os.path.islink(os.path.join(d, "root", "hot3d_lab_mesh")))

    def test_labels_use_the_shared_distance_rule(self):
        from train import contact_sdf

        self.assertIs(L.signed_mm, contact_sdf.signed_mm)
        self.assertIs(L.MeshSDF, contact_sdf.MeshSDF)
        self.assertEqual((L.THRESH_MM, L.CLIP_MM), (contact_sdf.THRESH_MM, contact_sdf.CLIP_MM))


class CITests(unittest.TestCase):
    def test_gt_ci_on_toy_set(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "toy")
            os.makedirs(root)
            files = [f"clip-{k // 2}/{k:04d}_r.jpg" for k in range(6)]
            with open(os.path.join(root, "contact.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(dict(file=x, obj="Chair" if k == 5 else "Mug")) for k, x in enumerate(files)) + "\n")
            with open(os.path.join(root, "index.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(dict(file=x, src=f"seq/{x.split('/')[0]}:{k}")) for k, x in enumerate(files)) + "\n")
            sdf = np.full((6, 778), 100.0, np.float16)
            sdf[:, :20] = 0
            np.save(os.path.join(root, "sdf.npy"), sdf)
            p = np.zeros((6, 778), np.float16)
            p[:, :20] = 0.9
            np.savez(os.path.join(root, "pred_a.npz"), files=np.asarray(files), prob=p, hand=np.full(6, 0.9))
            np.savez(os.path.join(root, "pred_b.npz"), files=np.asarray(files), prob=p * 0, hand=np.zeros(6))
            out = os.path.join(d, "ci.json")
            kw = dict(test_root=d, set="toy", boot=20, ap_boot=5, seed=0, out=out, vidx="", objs="", pairs="", arms="a=a@0.5,b=b@0.5")
            contact_ci.gt(SimpleNamespace(subset="", **kw))
            r = json.load(open(out))
            self.assertEqual((r["hands"], r["sequences"]), (6, 3))
            self.assertEqual(r["arms"]["a"]["vF1"], 1.0)
            self.assertEqual(r["arms"]["b"]["minus_a"]["vF1"][0], -1.0)
            contact_ci.gt(SimpleNamespace(subset="nochair", **kw))
            self.assertEqual(json.load(open(out))["hands"], 5)
            kw.update(arms="a=a@0.5,ab=a@0.5+b@0.5", pairs="ab-a")  # a seed group: metrics averaged over its runs
            contact_ci.gt(SimpleNamespace(subset="", **kw))
            r = json.load(open(out))["arms"]
            self.assertEqual((r["ab"]["vF1"], r["ab"]["runs"]), (0.5, 2))
            self.assertEqual(r["ab"]["minus_a"]["vF1"][0], -0.5)


if __name__ == "__main__":
    unittest.main()
