"""Tests for metrics: python3 scripts/test_metrics_collect.py"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as m  # noqa: E402
import metrics_collect as mc  # noqa: E402
from metrics_fixtures import SUMMARY  # noqa: E402,F401





class Collect(unittest.TestCase):
    def test_test_paths(self):
        for p in ["a/tests/x.rs", "tests/t.rs", "a/src/tests.rs", "a/src/ext_tests.rs"]:
            self.assertTrue(mc.is_test_path(p), p)
        for p in ["a/src/lib.rs", "a/src/testsuite.rs", "a/mytests/x.rs"]:
            self.assertFalse(mc.is_test_path(p), p)

    def test_lines_and_the_production_test_split(self):
        self.assertEqual(mc.count_lines("a\n\n  \nb\n"), 2)
        self.assertEqual(mc.split_rs("fn a() {}\n\nfn b() {}\n"), (2, 0))
        self.assertEqual(mc.split_rs("fn a() {}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {}\n}\n"), (1, 5))
        self.assertEqual(mc.split_rs(""), (0, 0))

    def test_crates_members_and_packages(self):
        self.assertEqual(mc.crate_of("ferrum_a/src/x.rs"), "ferrum_a")
        self.assertEqual(mc.crate_of("readme.md"), "(root)")
        self.assertEqual(mc.workspace_members('[workspace]\nmembers = ["a", "b/c"]\n'), ["a", "b/c"])
        self.assertEqual(mc.workspace_members("[package]\n"), [])
        self.assertEqual(mc.lock_packages('[[package]]\nname="a"\n[[package]]\nname="b"\n'), 2)

    def test_the_tables_over_a_small_tree(self):
        with tempfile.TemporaryDirectory() as d:
            for path, text in {
                "c/src/lib.rs": "fn a() {}\n#[cfg(test)]\nmod t {\n#[test]\nfn x() {}\n#[test]\n#[ignore]\nfn y() {}\n}\n",
                "c/tests/it.rs": "#[test]\nfn z() {}\n",
                "c/s.wgsl": "fn main() {}\n\n",
                "notes.md": "# t\n",
            }.items():
                os.makedirs(os.path.dirname(f"{d}/{path}"), exist_ok=True)
                open(f"{d}/{path}", "w").write(text)
            files = ["c/src/lib.rs", "c/tests/it.rs", "c/s.wgsl", "notes.md", ".claude/x.rs"]
            rust, sizes = mc.rust_table(d, files)
            self.assertEqual(rust["c"], dict(files=2, prod=1, test=10, tests=3, ignored=1))
            self.assertEqual(sizes, {"c/src/lib.rs": 1, "c/tests/it.rs": 0})
            self.assertEqual(mc.language_table(d, files), {"Rust": (2, 11), "WGSL shaders": (1, 1), "Markdown": (1, 1)})
            self.assertEqual(mc.read(d, "missing"), "")


if __name__ == "__main__":
    unittest.main()
