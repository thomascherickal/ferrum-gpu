"""Tests for metrics: python3 scripts/test_metrics_cli.py"""
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
