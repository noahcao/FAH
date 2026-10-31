"""fah.api.fetch_mhr_model: assets/mhr_model.pt taken from the MHR release archive, checked by sha256 (offline: a local
archive through a file:// URL)."""
import hashlib
import os
import tempfile
import unittest
import zipfile

from fah import api


class FetchMhrTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.data = os.urandom(1000)
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.zip = os.path.join(self.d, "assets.zip")
        with zipfile.ZipFile(self.zip, "w") as z:
            z.writestr("assets/LICENSE.txt", "Apache License")
            z.writestr("assets/mhr_model.pt", self.data)
        self.url = "file://" + self.zip

    def test_extracts_only_the_model_and_checks_its_hash(self):
        out = os.path.join(self.d, "sam3d", "assets", "mhr_model.pt")
        self.assertEqual(api.fetch_mhr_model(out, self.url, self.sha), out)
        self.assertEqual(open(out, "rb").read(), self.data)
        self.assertEqual(sorted(os.listdir(os.path.dirname(out))), ["mhr_model.pt"])  # no archive or temporary files left
        os.remove(self.zip)
        self.assertEqual(api.fetch_mhr_model(out, self.url, self.sha), out)  # right hash already there: no download

    def test_wrong_hash_fails_and_leaves_no_file(self):
        out = os.path.join(self.d, "x", "mhr_model.pt")
        with self.assertRaises(AssertionError):
            api.fetch_mhr_model(out, self.url, "0" * 64)
        self.assertFalse(os.path.exists(out))

    def test_release_hash(self):
        self.assertEqual(api.MHR_SHA256, "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc")
        self.assertTrue(api.MHR_ASSETS.startswith("https://github.com/facebookresearch/MHR/releases/download/v1.0.1/"))


if __name__ == "__main__":
    unittest.main()
