"""fah.api.relaunch_once: one relaunch when the child aborts before loaded(), none after it."""
import os
import tempfile
import unittest
from unittest.mock import patch

from fah import api

CHILD = """import os, sys
n = os.path.join(os.environ["FAKE_DIR"], "count")
k = int(open(n).read()) + 1 if os.path.exists(n) else 1
open(n, "w").write(str(k))
plan = os.environ["FAKE_PLAN"].split(",")[k - 1]  # per launch: abort | load_abort | ok
if plan != "abort":
    open(os.environ["FAH_LOADED_MARK"], "w").close()
if plan != "ok":
    os.abort()
"""


def run(plan):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "fake_child.py"), "w").write(CHILD)
        env = dict(FAKE_DIR=d, FAKE_PLAN=plan, PYTHONPATH=d + os.pathsep + os.environ.get("PYTHONPATH", ""))
        with patch.dict(os.environ, env), patch("sys.stderr"):
            rc = api.relaunch_once("fake_child", [])
        return rc, int(open(os.path.join(d, "count")).read())


class RelaunchTests(unittest.TestCase):
    def test_abort_before_loading_is_relaunched_once(self):
        self.assertEqual(run("abort,ok"), (0, 2))
        self.assertEqual(run("abort,abort"), (134, 2))  # at most one relaunch

    def test_abort_after_loading_and_clean_runs_are_not_relaunched(self):
        self.assertEqual(run("load_abort,ok"), (134, 1))
        self.assertEqual(run("ok"), (0, 1))


if __name__ == "__main__":
    unittest.main()
