"""Tests for gate_ok.py: python3 scripts/test_gate_ok.py"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_ok as g  # noqa: E402


def git(d, *a):
    subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)


class GateOk(unittest.TestCase):
    def repo(self):
        d = tempfile.mkdtemp()
        git(d, "init", "-q")
        open(f"{d}/a.rs", "w").write("fn a() {}\n")
        open(f"{d}/metrics.md", "w").write("m1\n")
        git(d, "add", "-A")
        git(d, "commit", "-qm", "one")
        return d

    def record(self, d, done, commit):
        os.makedirs(f"{d}/.gate", exist_ok=True)
        open(f"{d}/.gate/gate.done", "w").write(done + "\n")
        open(f"{d}/.gate/commit", "w").write(commit + "\n")

    def head(self, d):
        return g.git(d, "rev-parse", "HEAD")[1].strip()

    def test_a_pass_on_this_commit_allows_the_push(self):
        d = self.repo()
        self.assertFalse(g.passed_on_this_tree(d))  # nothing recorded
        self.record(d, "PASS", self.head(d))
        self.assertTrue(g.passed_on_this_tree(d))

    def test_generated_reports_may_change_after_the_gate_but_code_may_not(self):
        d = self.repo()
        self.record(d, "PASS", self.head(d))
        open(f"{d}/metrics.md", "w").write("m2\n")
        git(d, "commit", "-qam", "metrics only")
        self.assertTrue(g.passed_on_this_tree(d))
        open(f"{d}/a.rs", "w").write("fn b() {}\n")
        git(d, "commit", "-qam", "code")
        self.assertFalse(g.passed_on_this_tree(d))

    def test_a_failed_or_dirty_gate_never_allows_it(self):
        d = self.repo()
        self.record(d, "FAIL", self.head(d))
        self.assertFalse(g.passed_on_this_tree(d))
        self.record(d, "PASS", "dirty")
        self.assertFalse(g.passed_on_this_tree(d))
        self.assertEqual(g.gate_state(tempfile.mkdtemp()), (None, None))


if __name__ == "__main__":
    unittest.main()
