"""train.fetch_qc: sha256-checked QC lists land uncompressed where train.finetune reads them; a wrong file stops it."""
import contextlib
import gzip
import hashlib
import io
import json
import os
import tempfile
import unittest

from train import fetch_qc


def _asset(root, rows):
    os.makedirs(os.path.join(root, "pool_a"))
    name = "pool_a/qc.0.jsonl.gz"
    with gzip.open(os.path.join(root, name), "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    sha = hashlib.sha256(open(os.path.join(root, name), "rb").read()).hexdigest()
    return name, sha


class FetchQcTests(unittest.TestCase):
    def test_lists_land_in_the_pool(self):
        rows = [{"file": "a/0.jpg", "err2d": 0.05}, {"file": "a/1.jpg", "err2d": 0.3}]
        with tempfile.TemporaryDirectory() as tmp:
            src, crops = os.path.join(tmp, "src"), os.path.join(tmp, "crops")
            name, sha = _asset(src, rows)
            os.makedirs(os.path.join(crops, "pool_a"))
            man = os.path.join(tmp, "m.json")
            json.dump({"files": {name: {"sha256": sha, "rows": 2}}}, open(man, "w"))
            with contextlib.redirect_stdout(io.StringIO()):
                out = fetch_qc.fetch(src, crops, man)
            self.assertEqual(out, [os.path.join(crops, "pool_a", "qc.0.jsonl")])
            self.assertEqual([json.loads(x) for x in open(out[0])], rows)

    def test_wrong_sha_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            name, _ = _asset(src, [{"file": "a/0.jpg", "err2d": 0.05}])
            man = os.path.join(tmp, "m.json")
            json.dump({"files": {name: {"sha256": "0" * 64, "rows": 1}}}, open(man, "w"))
            with self.assertRaises(ValueError):
                fetch_qc.fetch(src, tmp, man)

    def test_manifest_matches_the_finetune_readers(self):
        files = json.load(open(fetch_qc.MANIFEST))["files"]
        for name in files:
            base = os.path.basename(name)
            self.assertTrue(base.startswith("qc.") or base == "mask_qc.jsonl.gz", name)


if __name__ == "__main__":
    unittest.main()
