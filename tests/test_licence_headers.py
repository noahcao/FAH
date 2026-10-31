"""Licence markings: every file with a Meta copyright header carries an SPDX identifier (SAM License or DINOv3 License),
every SPDX-marked file is listed in THIRD_PARTY_NOTICES.md, and the licence texts the headers point to exist."""
import os
from pathlib import Path
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def python_files():
    for root, _, files in os.walk(os.path.join(REPO, "fah")):
        for f in files:
            if f.endswith(".py") and not f.startswith("._"):  # (macOS resource-fork copies)
                yield os.path.relpath(os.path.join(root, f), REPO)


class LicenceHeaderTests(unittest.TestCase):
    def test_meta_files_are_marked_and_listed(self):
        notices = (Path(REPO) / "THIRD_PARTY_NOTICES.md").read_text()
        marked = 0
        for f in python_files():
            head = "".join((Path(REPO) / f).read_text().splitlines(keepends=True)[:6])
            spdx = re.search(r"SPDX-License-Identifier: (\S+)", head)
            if head.startswith("# Copyright (c) Meta Platforms, Inc. and affiliates.") or "dinov3_vit" in f:
                self.assertIsNotNone(spdx, f)
            if spdx:
                marked += 1
                self.assertIn(spdx.group(1), ("LicenseRef-SAM-License", "LicenseRef-DINOv3-License"), f)
                self.assertIn(f"`{f}`", notices, f)
                if spdx.group(1) == "LicenseRef-DINOv3-License":
                    self.assertIn("dinov3_vit", f)
        self.assertEqual(marked, 36)

    def test_licence_texts_exist(self):
        for f in ("LICENSE", "NOTICE", "WEIGHTS_LICENSE", "third_party_licenses/SAM-License.txt", "third_party_licenses/DINOv3-License.md"):
            self.assertTrue(os.path.exists(os.path.join(REPO, f)), f)
        self.assertIn("Apache License", (Path(REPO) / "LICENSE").read_text()[:200])
        w = (Path(REPO) / "WEIGHTS_LICENSE").read_text()
        self.assertIn((Path(REPO) / "third_party_licenses/SAM-License.txt").read_text().strip(), w)  # verbatim pass-through
        self.assertIn((Path(REPO) / "third_party_licenses/DINOv3-License.md").read_text().strip(), w)
        self.assertIn("Built with DINOv3", w)
        self.assertIn("RESEARCH AND NON-COMMERCIAL USE ONLY", w)
        for term in ("shared/detector.pt", "AGPL-3.0", "Ultralytics",
                     "TensorRT", "ONNX", "Core ML"):
            self.assertIn(term, w)
        detector_terms = w.split("PART 4. DETECTOR PROVENANCE AND UPSTREAM TERMS", 1)[1]
        self.assertNotIn("Lineage: WiLoR", detector_terms)
        self.assertIn("credit FAH and Ultralytics", detector_terms)



if __name__ == "__main__":
    unittest.main()
