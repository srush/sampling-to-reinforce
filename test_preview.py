"""Check that preview failures recover without losing the last good page."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import preview


class PreviewRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.previous_version = preview.VERSION
        self.previous_error = preview.ERROR

    def tearDown(self):
        preview.VERSION = self.previous_version
        preview.ERROR = self.previous_error

    def test_failed_build_then_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "index.html"
            page.write_text("last successful build")
            builder = root / "build_html.py"
            builder.write_text('raise ValueError("bad notebook cell")\n')
            with patch.object(preview, "ROOT", root), redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                version = preview.VERSION
                self.assertFalse(preview.rebuild())
                self.assertIn("bad notebook cell", preview.ERROR)
                self.assertEqual(preview.VERSION, version)
                self.assertEqual(page.read_text(), "last successful build")
                builder.write_text('from pathlib import Path\nPath("index.html").write_text("recovered")\n')
                self.assertTrue(preview.rebuild())
                self.assertEqual(preview.ERROR, "")
                self.assertEqual(preview.VERSION, version + 1)
                self.assertEqual(page.read_text(), "recovered")

    def test_builder_launch_failure(self):
        with patch.object(preview.subprocess, "run", side_effect=OSError("cannot launch")), redirect_stderr(StringIO()):
            self.assertFalse(preview.rebuild())
            self.assertIn("cannot launch", preview.ERROR)

    def test_watcher_continues_after_error(self):
        snapshots = [{Path("puzzle.py"): 1}, {Path("puzzle.py"): 2}]
        with patch.object(preview, "sources", side_effect=snapshots), \
                patch.object(preview, "rebuild", side_effect=[RuntimeError("temporary error"), True]) as rebuild, \
                patch.object(preview.time, "sleep", side_effect=[None, None, KeyboardInterrupt]), \
                redirect_stderr(StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                preview.watch(previous={})
            self.assertEqual(rebuild.call_count, 2)


if __name__ == "__main__":
    unittest.main()
