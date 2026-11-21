"""Tests for metrics.py and metrics_collect.py: python3 scripts/test_metrics.py"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as m  # noqa: E402
import metrics_collect as mc  # noqa: E402

SUMMARY = """Filename            Regions  Missed Regions  Cover  Functions  Missed Functions  Executed  Lines  Missed Lines  Cover  Branches  Missed Branches  Cover
-------------------------------------------------------------------------------------------------------------------------------------------
ferrum_a/src/x.rs        10               1  90.00%          4                 0   100.00%     40             4  90.00%         0                0      -
ferrum_a/src/y.rs        20               0 100.00%          2                 0   100.00%     60             0 100.00%         0                0      -
ferrum_b/src/z.rs        30               6  80.00%          6                 2    66.67%    100            10  90.00%         0                0      -
-------------------------------------------------------------------------------------------------------------------------------------------
TOTAL                    60               7  88.33%         12                 2    83.33%    200            14  93.00%         0                0      -
"""


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


class Render(unittest.TestCase):
    def test_summary_parsing(self):
        rows, total = m.parse_summary(SUMMARY)
        self.assertEqual([r["name"] for r in rows], ["ferrum_a/src/x.rs", "ferrum_a/src/y.rs", "ferrum_b/src/z.rs"])
        self.assertEqual((total["lines"], total["missed_lines"], total["regions"]), (200, 14, 60))
        self.assertEqual(m.parse_summary("nothing\nuseful"), ([], None))
        self.assertEqual(m.parse_summary("a.rs x y z 1 2 3 4 5 6")[0], [])

    def test_coverage_block(self):
        b = m.coverage_block(SUMMARY, "98", "abc1234", "2026-11-18 10:00 UTC")
        self.assertIn("| lines | 186 | 200 | 14 | 93.00% |", b)
        self.assertIn("| ferrum_a | 100 | 4 | 96.00% | 96.67% |", b)
        self.assertIn("| ferrum_b | 100 | 10 | 90.00% | 80.00% |", b)
        self.assertIn("**98%**", b)
        self.assertIn("`abc1234`", b)
        # the file with the most missed lines is listed first, fully covered files are not listed
        self.assertLess(b.index("ferrum_b/src/z.rs"), b.index("ferrum_a/src/x.rs"))
        self.assertNotIn("y.rs", b)
        self.assertEqual(m.coverage_block("garbage", "98", "x", "now"), m.NONE_YET)
        absolute = SUMMARY.replace("ferrum_a/", "/repo/ferrum_a/").replace("ferrum_b/", "/repo/ferrum_b/")
        self.assertIn("| ferrum_a | 100 | 4 |", m.coverage_block(absolute, "98", "c", "t", "/repo"))
        full = SUMMARY.replace("            4  90.00%", "            0 100.00%")
        self.assertIsNotNone(m.coverage_block(full.replace("ferrum_b/src/z.rs", "ferrum_b/src/zz.rs"), "100", "c", "t"))

    def test_gate_block(self):
        text = "start rc=0 10:00:00\nfmt rc=0 10:00:30\nxe rc=1 10:05:30\ncoverage rc=0 00:10:30\n"
        b = m.gate_block(text, "FAIL\n")
        self.assertIn("| fmt | pass | 0m 30s |", b)
        self.assertIn("| xe | FAIL (1) | 5m 00s |", b)
        self.assertIn("| coverage | pass | ", b)
        self.assertIn("| **total** | FAIL | ", b)
        self.assertEqual(m.gate_block("start rc=0 10:00:00\n", "PASS"), m.NONE_YET)
        self.assertEqual(m.seconds("01:02:03"), 3723)

    def test_blocks_wrap_and_unwrap(self):
        w = m.wrap(m.GATE_MARK, "body")
        self.assertEqual(m.get_block("x\n" + w + "\ny", m.GATE_MARK), "body")
        self.assertIsNone(m.get_block("no markers", m.GATE_MARK))
        self.assertEqual(m.pct(1, 4), "25.00%")
        self.assertEqual(m.pct(0, 0), "-")


def git(d, *a):
    subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)


class Cli(unittest.TestCase):
    def repo(self, d):
        git(d, "init", "-q")
        os.makedirs(f"{d}/c/src")
        open(f"{d}/c/src/lib.rs", "w").write("fn a() {}\n")
        open(f"{d}/Cargo.toml", "w").write('[workspace]\nmembers = ["c"]\n')
        git(d, "add", "-A")
        git(d, "commit", "-qm", "first")

    def test_a_run_writes_the_file_keeps_measured_sections_and_is_stable(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d)
            out = f"{d}/metrics.md"
            self.assertEqual(m.main(["--root", d]), 0)
            text = open(out).read()
            self.assertIn("**0** `#[test]`", text)
            self.assertIn("| commits |", text)
            self.assertIn(m.NONE_YET, text)
            # a coverage run and a gate run replace their sections
            open(f"{d}/cov.txt", "w").write(SUMMARY)
            open(f"{d}/gate.txt", "w").write("start rc=0 10:00:00\nfmt rc=0 10:01:00\n")
            m.main(["--root", d, "--coverage", f"{d}/cov.txt", "--floor", "98", "--gate", f"{d}/gate.txt", "--verdict", "PASS"])
            text = open(out).read()
            self.assertIn("93.00%", text)
            self.assertIn("| fmt | pass | 1m 00s |", text)
            # a recorded run can name the commit and time it measured
            m.main(["--root", d, "--coverage", f"{d}/cov.txt", "--commit", "74190dc", "--when", "2026-11-18 13:42 UTC"])
            self.assertIn("2026-11-18 13:42 UTC on commit `74190dc`", open(out).read())
            # a plain run afterwards keeps both
            m.main(["--root", d])
            again = open(out).read()
            self.assertIn("93.00%", again)
            self.assertIn("| fmt | pass | 1m 00s |", again)
            # the pre-commit run counts the commit being made
            m.main(["--root", d, "--next-commit"])
            self.assertIn("| 2 |", open(out).read())
            self.assertIn("plus the commit being made", open(out).read())

    def test_the_gate_verdict_is_read_from_gate_done(self):
        with tempfile.TemporaryDirectory() as d:
            self.repo(d)
            os.makedirs(f"{d}/.gate")
            open(f"{d}/.gate/gate.txt", "w").write("start rc=0 10:00:00\nfmt rc=0 10:00:10\n")
            open(f"{d}/.gate/gate.done", "w").write("PASS\n")
            m.main(["--root", d, "--gate", f"{d}/.gate/gate.txt"])
            self.assertIn("| **total** | PASS | 0m 10s |", open(f"{d}/metrics.md").read())


if __name__ == "__main__":
    unittest.main()
